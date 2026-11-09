import type { ResiliencePolicy } from '@nestjs/resilience';
import type { ModelMessage, ToolSet } from 'ai';
import { AiConfigurationError } from '../errors/ai.errors.js';
import type { AiCallOptions, AiModelReference, AiToolChoice } from '../interfaces/ai-call-options.interface.js';
import type { AiBudget, AiModuleOptions } from '../interfaces/ai-module-options.interface.js';
import type { AiCaller } from '../interfaces/ai-tool.interface.js';
import type { AiUsageEvent } from '../interfaces/ai-usage.interface.js';
import type { AiModels, ResolvedModel } from '../models/ai-models.service.js';
import type { AiToolProgress, AiTools } from '../tools/ai-tools.service.js';
import { modelPolicy, type PolicyPlan, type StepAttempt } from './model-policy.js';
import { validateBudget } from '../utils/usage.util.js';

const DEFAULT_MAX_STEPS = 10;

/** The generation settings of one model call, as `generateText()` and `streamText()` take them. */
export interface GenerationSettings {
  maxOutputTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  seed?: number;
  stopSequences?: string[];
  presencePenalty?: number;
  frequencyPenalty?: number;
  providerOptions?: Record<string, Record<string, any>>;
  headers?: Record<string, string | undefined>;
}

/** Everything a call needs, resolved once from its options, the models and the module's defaults. Internal. */
export interface CallPlan extends PolicyPlan {
  label?: string;
  maxSteps: number;
  budget?: AiBudget;
  instructions?: string;
  messages: ModelMessage[];
  tools?: ToolSet;
  toolChoice?: AiToolChoice;
  caller: AiCaller;
  listeners: Array<(event: AiUsageEvent) => void | Promise<void>>;
  settingsFor(model: ResolvedModel): GenerationSettings;
  /**
   * The attempt whose tools may run, set by the step. A tool that starts means the model has answered: the attempt
   * commits (no timeout, retry or fallback from then on), so that no tool runs twice.
   */
  running?: StepAttempt;
}

/**
 * Resolves a call's options. Mistakes (an unknown model or agent function, both `prompt` and `messages`) throw here,
 * before anything is sent. `signal` is the call's: the caller's, or one combined with it.
 */
export function createCallPlan(
  options: AiCallOptions,
  {
    models,
    tools,
    moduleOptions,
    signal,
    preset,
    toolExecution = true,
  }: {
    models: AiModels;
    tools: AiTools;
    moduleOptions: AiModuleOptions;
    signal: AbortSignal;
    /** A preset of `ResilienceService`, by name. */
    preset: (name: string) => ResiliencePolicy;
    /** `false`: tool calls are returned, not run (durable runs). */
    toolExecution?: boolean;
  },
): CallPlan {
  if (typeof options !== 'object' || options === null) {
    throw new AiConfigurationError('Pass the call its options: { prompt } or { messages }, and the rest.');
  }
  const defaults = moduleOptions.defaults ?? {};
  const primary = models.resolve(options.model);
  const identity = (model: ResolvedModel) => model.info.name ?? model.model;
  const chain = [primary, ...fallbacks(options.fallback, primary, models)].filter(
    (model, index, all) => all.findIndex((other) => identity(other) === identity(model)) === index,
  );

  const caller: AiCaller = { ...options.caller };
  const plan = {} as CallPlan;
  const onExecute = () => plan.running?.commit();
  // A streamed answer carries the agent functions' progress, as transient data chunks `useChat()` hands to `onData`.
  const onProgress = (progress: AiToolProgress) =>
    plan.running?.emit({ type: 'data-tool-progress', id: progress.toolCallId, data: progress, transient: true });
  const selected = options.tools?.length
    ? tools.select(options.tools, { caller, onExecute, onProgress, execute: toolExecution })
    : undefined;
  const maxSteps = options.maxSteps ?? defaults.maxSteps ?? DEFAULT_MAX_STEPS;
  if (!Number.isInteger(maxSteps) || maxSteps < 1) {
    throw new AiConfigurationError(`\`maxSteps\` must be a whole number of at least 1. Got ${String(maxSteps)}.`);
  }

  const budget = options.budget ?? defaults.budget;
  validateBudget(budget);

  const policies = new Map<ResolvedModel, ResiliencePolicy | undefined>();
  const pick = <K extends keyof GenerationSettings>(model: ResolvedModel, key: K): GenerationSettings[K] =>
    (options as GenerationSettings)[key] ?? (model.settings as GenerationSettings)[key] ?? (defaults as GenerationSettings)[key];

  return Object.assign(plan, {
    ...(options.label !== undefined && { label: options.label }),
    chain,
    signal,
    maxSteps,
    ...(budget && { budget }),
    ...(options.instructions !== undefined && { instructions: options.instructions }),
    messages: messagesOf(options),
    ...(selected && { tools: selected }),
    ...(options.toolChoice !== undefined && { toolChoice: options.toolChoice }),
    caller,
    listeners: [moduleOptions.onUsage, options.onUsage].filter((listener) => typeof listener === 'function'),
    source: options.label ?? 'AiService',
    fallbackIf: options.fallbackIf ?? (() => true),
    policyFor: (model) => {
      let policy = policies.get(model);
      if (!policies.has(model)) {
        const own = options.resilience ?? model.settings.resilience ?? defaults.resilience;
        policy =
          own === undefined
            ? modelPolicy(model, {
                retry: options.retry ?? model.settings.retry ?? defaults.retry,
                timeout: options.timeout ?? model.settings.timeout ?? defaults.timeout,
              })
            : typeof own === 'string'
              ? preset(own)
              : own;
        policies.set(model, policy);
      }
      return policy;
    },
    settingsFor: (model) => {
      const settings: GenerationSettings = {};
      for (const key of [
        'maxOutputTokens',
        'temperature',
        'topP',
        'topK',
        'seed',
        'stopSequences',
        'presencePenalty',
        'frequencyPenalty',
        'providerOptions',
        'headers',
      ] as const) {
        const value = pick(model, key);
        if (value !== undefined) {
          (settings as Record<string, unknown>)[key] = value;
        }
      }
      return settings;
    },
  } satisfies Omit<CallPlan, 'running'>);
}

function fallbacks(
  option: AiCallOptions['fallback'],
  primary: ResolvedModel,
  models: AiModels,
): ResolvedModel[] {
  if (option === false) {
    return [];
  }
  if (option === undefined) {
    return primary.fallback.map((name) => models.resolve(name));
  }
  const references: AiModelReference[] = Array.isArray(option) ? [...option] : [option as AiModelReference];
  return references.map((reference) => models.resolve(reference));
}

function messagesOf(options: AiCallOptions): ModelMessage[] {
  const { prompt, messages } = options;
  if (prompt !== undefined && messages !== undefined) {
    throw new AiConfigurationError('Pass `prompt` or `messages`, not both.');
  }
  if (typeof prompt === 'string') {
    return [{ role: 'user', content: prompt }];
  }
  if (Array.isArray(messages) && messages.length > 0) {
    return [...messages];
  }
  throw new AiConfigurationError('Pass `prompt` (a string) or `messages` (a non-empty array of model messages).');
}
