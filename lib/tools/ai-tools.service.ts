import { Inject, Injectable, Logger, type OnModuleInit } from '@nestjs/common';
import { DiscoveryService, ExternalContextCreator, MetadataScanner } from '@nestjs/core';
import { isObservable, lastValueFrom } from 'rxjs';
import type { Schema, ToolSet } from 'ai';
import { AGENT_FUNCTION_METADATA, AI_CONTEXT_TYPE, AI_MODULE_OPTIONS, AI_PARAMS_METADATA } from '../ai.constants.js';
import { AiParamsFactory, recordHandlerParams } from '../context/ai-params.js';
import { AiConfigurationError, AiToolError } from '../errors/ai.errors.js';
import type { AgentFunctionContext, AgentFunctionOptions } from '../interfaces/agent-function.interface.js';
import type { AiModuleOptions } from '../interfaces/ai-module-options.interface.js';
import type { AiCaller, AiToolSelection } from '../interfaces/ai-tool.interface.js';
import { toToolError } from '../utils/errors.util.js';
import { assertProviderToolName } from '../utils/names.util.js';
import { EMPTY_OBJECT_SCHEMA, toModelSchema } from '../utils/schema.util.js';
import { toToolOutput } from './tool-output.util.js';

/** A method behind Nest's enhancers: called as `(input, context)`. */
export type AiToolHandler = (input: unknown, context: AgentFunctionContext) => Promise<unknown>;

/** An agent function of the application: a provider method with `@AgentFunction()`, offered to models as a tool. */
export interface AiToolEntry {
  name: string;
  description?: string;
  scopes?: string[];
  /** The `ai` schema of its arguments. */
  inputSchema: Schema<unknown>;
  handler: AiToolHandler;
  /** The provider class it is a method of. */
  host: Function;
  /** `Class.method`, for messages. */
  where: string;
}

const NEVER_ABORTED = new AbortController().signal;

/** An agent function's progress (`ctx.reportProgress()`), which a streamed answer sends as a transient data chunk. */
export interface AiToolProgress {
  toolCallId: string;
  toolName: string;
  progress: number;
  total?: number;
  message?: string;
}

/**
 * The application's agent functions: the `@AgentFunction()` methods of its providers (or of the `include` modules),
 * found once at startup, each wrapped by Nest's enhancers with the context type `'ai'`, and called with an
 * `AgentFunctionContext`. `select()` turns some of them into `ai` tools for one caller.
 */
@Injectable()
export class AiTools implements OnModuleInit {
  private readonly logger = new Logger('AiModule');
  private readonly entries = new Map<string, AiToolEntry>();
  private readonly paramsFactory = new AiParamsFactory();
  private readonly contextCreator: ExternalContextCreator;
  private scanned = false;

  constructor(
    private readonly discovery: DiscoveryService,
    private readonly scanner: MetadataScanner,
    externalContextCreator: ExternalContextCreator,
    @Inject(AI_MODULE_OPTIONS) private readonly options: AiModuleOptions,
  ) {
    this.contextCreator = withOwnHandlerMetadata(externalContextCreator);
  }

  onModuleInit() {
    this.scan();
  }

  /** Every agent function, by name. */
  get tools(): ReadonlyMap<string, AiToolEntry> {
    this.scan();
    return this.entries;
  }

  /**
   * `ai` tools for `names` (agent function names, or provider classes for all their agent functions), whose calls run
   * as `caller`: what `AiService` calls pass the model, and what code that calls `ai` itself can pass `generateText()`.
   */
  select(
    selection: AiToolSelection,
    {
      caller = {},
      onExecute,
      onProgress,
      execute = true,
    }: {
      caller?: AiCaller;
      /** @internal */
      onExecute?: () => void;
      /** @internal Where `ctx.reportProgress()` goes: a streamed answer. */
      onProgress?: (progress: AiToolProgress) => void;
      /** @internal `false`: the tools are offered to the model, and their calls returned, not run (durable runs). */
      execute?: boolean;
    } = {},
  ): ToolSet {
    this.scan();
    const tools: ToolSet = {};
    for (const entry of this.resolve(selection)) {
      // An `ai` tool is a plain object; `ai` validates the input against the schema before `execute`.
      tools[entry.name] = {
        ...(entry.description !== undefined && { description: entry.description }),
        inputSchema: entry.inputSchema,
        ...(execute && {
          execute: (input: unknown, options: { toolCallId: string; messages: unknown[]; abortSignal?: AbortSignal }) => {
            onExecute?.();
            return this.call(entry, input, caller, { ...options, ...(onProgress && { onProgress }) });
          },
        }),
      } as unknown as ToolSet[string];
    }
    return tools;
  }

  /**
   * Runs the agent function `name` with `input` (validated by its schema) as `caller`, through its enhancers, as an
   * agent's call does. Resolves with what the model would read; rejects with the `AiToolError` it would read.
   */
  async call(
    entryOrName: AiToolEntry | string,
    input: unknown,
    caller: AiCaller = {},
    options: {
      toolCallId?: string;
      messages?: unknown[];
      abortSignal?: AbortSignal;
      idempotencyKey?: string;
      onProgress?: (progress: AiToolProgress) => void;
    } = {},
  ): Promise<unknown> {
    const entry = typeof entryOrName === 'string' ? this.resolve([entryOrName])[0]! : entryOrName;
    if (typeof entryOrName === 'string') {
      const validated = await entry.inputSchema.validate?.(input);
      if (validated && !validated.success) {
        throw new AiToolError(`Invalid input: ${validated.error.message}`, entry.name);
      }
      input = validated ? validated.value : input;
    }
    if (entry.scopes?.length) {
      const held = new Set(caller.scopes ?? []);
      const missing = entry.scopes.filter((scope) => !held.has(scope));
      if (missing.length > 0) {
        throw new AiToolError(`Forbidden: this tool needs the scope(s) ${missing.join(', ')}`, entry.name);
      }
    }
    const toolCallId = options.toolCallId ?? '';
    // Where `@nestjs/authentication` and `@nestjs/authorization` look on an `'ai'` call, as on an `'mcp'` one: the
    // request it runs in, else the context itself, with its `user`.
    const context: AgentFunctionContext = {
      name: entry.name,
      ...(caller.request !== undefined && { request: caller.request }),
      ...(caller.user !== undefined && { user: caller.user }),
      ...(caller.scopes !== undefined && { scopes: caller.scopes }),
      signal: options.abortSignal ?? NEVER_ABORTED,
      reportProgress: async (progress, total, message) =>
        options.onProgress?.({
          toolCallId,
          toolName: entry.name,
          progress,
          ...(total !== undefined && { total }),
          ...(message !== undefined && { message }),
        }),
      toolCallId,
      ...(options.idempotencyKey !== undefined && { idempotencyKey: options.idempotencyKey }),
      messages: (options.messages ?? []) as AgentFunctionContext['messages'],
    };
    let value: unknown;
    try {
      value = await entry.handler(input ?? {}, context);
      if (isObservable(value)) {
        value = await lastValueFrom(value);
      }
    } catch (error) {
      // Nest's exception filters have run; map the failure to the error the model reads.
      throw toToolError(error, entry.name);
    }
    try {
      return toToolOutput(value, entry.name, entry.where);
    } catch (error) {
      if (!(error instanceof AiToolError)) {
        this.logger.error(`${entry.where} returned a value that is not JSON`, (error as Error)?.stack);
      }
      throw toToolError(error, entry.name);
    }
  }

  /** The agent functions a selection names, each once, in order. Unknown names throw. */
  resolve(selection: AiToolSelection): AiToolEntry[] {
    this.scan();
    if (!Array.isArray(selection)) {
      throw new AiConfigurationError(
        "`tools` lists agent function names and provider classes, e.g. ['find-order', OrdersTools].",
      );
    }
    const picked = new Map<string, AiToolEntry>();
    for (const item of selection) {
      if (typeof item === 'string') {
        const entry = this.entries.get(item);
        if (!entry) {
          throw new AiConfigurationError(`There is no agent function named "${item}". ${this.known()}`);
        }
        picked.set(entry.name, entry);
      } else if (typeof item === 'function') {
        const matches = [...this.entries.values()].filter((entry) => entry.host === item || entry.host.prototype instanceof item);
        if (matches.length === 0) {
          throw new AiConfigurationError(
            `${item.name} has no agent functions, or is not a provider of the application. ${this.known()}`,
          );
        }
        for (const entry of matches) {
          picked.set(entry.name, entry);
        }
      } else {
        throw new AiConfigurationError(`\`tools\` lists agent function names and provider classes; got ${String(item)}.`);
      }
    }
    return [...picked.values()];
  }

  /** Finds the agent functions, once. */
  scan(): void {
    if (this.scanned) {
      return;
    }
    this.scanned = true;

    const seen = new Set<object>();
    const wrappers = this.discovery.getProviders(this.options.include ? { include: this.options.include } : undefined);
    for (const wrapper of wrappers) {
      const { instance } = wrapper;
      if (!instance || typeof instance !== 'object' || wrapper.isAlias || seen.has(instance)) {
        continue;
      }
      seen.add(instance);

      const prototype = Object.getPrototypeOf(instance);
      for (const methodName of this.scanner.getAllMethodNames(prototype)) {
        const method = prototype[methodName];
        const definition = Reflect.getMetadata(AGENT_FUNCTION_METADATA, method) as AgentFunctionOptions | undefined;
        if (!definition) {
          continue;
        }

        const where = `${wrapper.metatype?.name ?? instance.constructor.name}.${methodName}`;
        if (!wrapper.isDependencyTreeStatic()) {
          throw new AiConfigurationError(
            `${where}: agent functions must be on singleton providers, and this one is request-scoped or transient (or ` +
              'depends on a request-scoped provider). Pass what a call needs through `caller` instead.',
          );
        }
        const name = definition.name ?? methodName;
        assertProviderToolName(name, where);
        const existing = this.entries.get(name);
        if (existing) {
          throw new AiConfigurationError(
            `${where}: there is already an agent function named "${name}" (${existing.where}). Give one of them ` +
              'another { name }.',
          );
        }

        recordHandlerParams(instance.constructor, methodName);
        const handler = this.contextCreator.create(
          instance,
          method,
          methodName,
          AI_PARAMS_METADATA,
          this.paramsFactory,
          undefined,
          undefined,
          undefined,
          AI_CONTEXT_TYPE,
        ) as AiToolHandler;
        this.entries.set(name, {
          name,
          ...(definition.description !== undefined && { description: definition.description }),
          ...(definition.scopes?.length && { scopes: [...definition.scopes] }),
          inputSchema: definition.inputSchema
            ? toModelSchema(definition.inputSchema, `${where} inputSchema`, { object: true })
            : (EMPTY_OBJECT_SCHEMA as Schema<unknown>),
          handler,
          host: instance.constructor,
          where,
        });
      }
    }
  }

  private known(): string {
    const names = [...this.entries.keys()];
    return names.length
      ? `Agent functions: ${names.join(', ')}.`
      : 'The application has no agent functions (@AgentFunction() methods).';
  }
}

/**
 * Nest's `ExternalContextCreator`, with a cache of handler metadata of its own. Nest's keeps what it reads of a
 * method's parameters by class and method name, and builds the context type of the first `create()` for the method into
 * it: the `ExecutionContext` that `createParamDecorator()` decorators receive. A method that is an `@nestjs/mcp` tool
 * too is wrapped by both packages, and its parameter decorators would see the type of whichever wrapped it first,
 * `'ai'` or `'mcp'`, on every call. Guards, interceptors and filters get the type of each `create()` either way. Each
 * agent function is wrapped once, at startup, so nothing needs caching here.
 */
function withOwnHandlerMetadata(creator: ExternalContextCreator): ExternalContextCreator {
  // `handlerMetadataStorage` is the private field `getMetadata()` reads, in every 12.x release.
  return Object.create(creator, {
    handlerMetadataStorage: { value: { get: () => undefined, set: () => {} } },
  }) as ExternalContextCreator;
}
