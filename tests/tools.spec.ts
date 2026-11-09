/**
 * The registry of agent functions outside HTTP: discovery and its mistakes, selection, `ai` tools for code that calls
 * `ai` itself, MCP results, callers without a request (a queue job), the context of a call and its `'ai'` type, and a
 * method that is an MCP tool too, each decorator with its own options.
 */
import {
  Catch,
  createParamDecorator,
  ForbiddenException,
  Injectable,
  Module,
  Scope,
  UseFilters,
  UseGuards,
  type CanActivate,
  type DynamicModule,
  type ExecutionContext,
} from '@nestjs/common';
import { Test } from '@nestjs/testing';
import { McpModule, McpRegistry, Tool } from '@nestjs/mcp';
import { connectMcpClient } from '@nestjs/mcp/testing';
import { generateText } from 'ai';
import { EMPTY, of, throwError } from 'rxjs';
import * as z from 'zod';
import {
  AgentFunction,
  AiConfigurationError,
  AiExecutionContext,
  AiModule,
  AiTools,
  type AgentFunctionContext,
  type AiContextType,
} from '../lib/index.js';
import { callTool, FakeModel } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

/** One schema for both decorators of `search`. */
const SearchInput = z.object({ query: z.string() });

@Catch()
class ReturnsObservableFilter {
  catch() {
    return of({ refunded: false, reason: 'Not your order' });
  }
}

@Catch()
class ReturnsErrorObservableFilter {
  catch() {
    return throwError(() => new ForbiddenException('Not your order'));
  }
}

@Catch()
class ReturnsEmptyObservableFilter {
  catch() {
    return EMPTY;
  }
}

@Injectable()
class ObservableFilterTools {
  @AgentFunction({ name: 'observable-result' })
  @UseFilters(ReturnsObservableFilter)
  result() {
    throw new Error('The order belongs to another customer');
  }

  @AgentFunction({ name: 'observable-error' })
  @UseFilters(ReturnsErrorObservableFilter)
  error() {
    throw new Error('The order belongs to another customer');
  }

  @AgentFunction({ name: 'observable-empty' })
  @UseFilters(ReturnsEmptyObservableFilter)
  empty() {
    throw new Error('The order belongs to another customer');
  }
}

@Injectable()
class CatalogTools {
  @AgentFunction({ description: 'Look a product up', inputSchema: z.object({ sku: z.string() }) })
  product({ sku }: { sku: string }) {
    return { sku, name: 'Salmon kibble', restocked: new Date('2026-11-01T00:00:00Z') };
  }

  @AgentFunction({ description: 'Takes no arguments' })
  stock() {
    return undefined;
  }

  // An MCP tool with a dotted name, which model providers refuse, and an agent function with a name they accept.
  @Tool({ name: 'catalog.search', description: 'Search the catalog (MCP)', inputSchema: SearchInput })
  @AgentFunction({ name: 'catalog-search', description: 'Search the catalog', inputSchema: SearchInput })
  search({ query }: { query: string }) {
    return `found ${query}`;
  }

  // An MCP tool only: never an agent function, whatever its options.
  @Tool({ name: 'mcp-only', description: 'For MCP clients' })
  mcpOnly() {
    return 'mcp';
  }

  @AgentFunction({ name: 'structured', description: 'Returns a CallToolResult' })
  structured() {
    return { content: [{ type: 'text', text: '{"count":2}' }], structuredContent: { count: 2 } };
  }

  @AgentFunction({ name: 'texts', description: 'Returns text blocks' })
  texts() {
    return { content: [{ type: 'text', text: 'one' }, { type: 'image', data: 'AAAA', mimeType: 'image/png' }, { type: 'text', text: 'two' }] };
  }

  @AgentFunction({ name: 'failed', description: 'Returns isError' })
  failed() {
    return { content: [{ type: 'text', text: 'Out of stock' }], isError: true };
  }
}

@Injectable()
class JobTools {
  @AgentFunction({ description: 'Who the job runs for' })
  owner(_input: unknown, ctx: AgentFunctionContext) {
    return { user: ctx.user, request: ctx.request ?? null, scopes: ctx.scopes ?? null };
  }

  @AgentFunction({ description: 'The whole context' })
  context(_input: unknown, ctx: AgentFunctionContext) {
    const { signal, reportProgress, messages, ...rest } = ctx;
    return { ...rest, signal: signal instanceof AbortSignal, reportProgress: typeof reportProgress, messages: messages.length };
  }

  @AgentFunction({ description: 'Takes its time, and stops when its signal aborts' })
  async slow(_input: unknown, ctx: AgentFunctionContext) {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, 150);
      ctx.signal.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(ctx.signal.reason);
      });
    });
    return 'finally';
  }

  @AgentFunction({ description: 'Waits for the signal' })
  async wait(_input: unknown, ctx: AgentFunctionContext) {
    await new Promise((resolve) => ctx.signal.addEventListener('abort', resolve, { once: true }));
    return { aborted: ctx.signal.aborted };
  }
}

/** What a guard of an agent function reads of its call. */
@Injectable()
class SeenGuard implements CanActivate {
  static seen: unknown[] = [];

  canActivate(context: ExecutionContext): boolean {
    const call = AiExecutionContext.create(context);
    SeenGuard.seen.push({
      type: context.getType<AiContextType>(),
      ownType: call.getType(),
      input: call.getInput(),
      rpcData: call.switchToRpc().getData(),
      name: call.getName(),
      request: call.getRequest() ?? null,
      toolCallId: call.getContext().toolCallId,
      handler: call.getHandler()?.name,
      class: call.getClass()?.name,
    });
    return true;
  }
}

@Injectable()
class GuardedTools {
  @AgentFunction({ name: 'look-up', inputSchema: z.object({ id: z.number() }) })
  @UseGuards(SeenGuard)
  lookUp({ id }: { id: number }) {
    return { id };
  }
}

/** The context type a custom parameter decorator sees. */
const SeenType = createParamDecorator((_data: unknown, context: ExecutionContext) => context.getType());

@Injectable()
class StackedTools {
  @Tool({ name: 'seen-type' })
  @AgentFunction({ name: 'seen-type' })
  seen(_input: unknown, _ctx: unknown, @SeenType() type: string) {
    return type;
  }
}

describe('the registry of agent functions', () => {
  it('finds the @AgentFunction() methods, and selects them by name or class', async () => {
    const { moduleRef } = await createAi({ models: { default: new FakeModel([]) } }, { providers: [CatalogTools, JobTools] });
    const tools = moduleRef.get(AiTools);
    expect([...tools.tools.keys()].sort()).toEqual(
      ['catalog-search', 'context', 'failed', 'owner', 'product', 'slow', 'stock', 'structured', 'texts', 'wait'].sort(),
    );
    expect(tools.tools.get('product')).toMatchObject({ where: 'CatalogTools.product', description: 'Look a product up' });
    expect(Object.keys(tools.select(['owner', 'product', 'owner']))).toEqual(['owner', 'product']);
    expect(Object.keys(tools.select([JobTools]))).toEqual(['owner', 'context', 'slow', 'wait']);
    expect(() => tools.select(['nope'])).toThrow('There is no agent function named "nope". Agent functions:');
    expect(() => tools.select([class Unrelated {}])).toThrow('Unrelated has no agent functions');
    // A method with `@Tool()` alone is no agent function: there is no fallback from one decorator to the other.
    expect(() => tools.select(['mcp-only'])).toThrow('There is no agent function named "mcp-only"');
    expect(() => tools.select(['catalog.search'])).toThrow('There is no agent function named "catalog.search"');
    await moduleRef.close();
  });

  it('keeps the options of @Tool() and @AgentFunction() apart on a method that has both', async () => {
    const { moduleRef } = await createAi(
      { models: { default: new FakeModel([]) } },
      { providers: [CatalogTools], imports: [McpModule.forRoot({ name: 'catalog', version: '1.0.0' })] },
    );
    const mcp = moduleRef.get(McpRegistry, { strict: false });
    expect(mcp.tools.map((tool) => [tool.name, tool.options.description])).toEqual([
      ['catalog.search', 'Search the catalog (MCP)'],
      ['mcp-only', 'For MCP clients'],
    ]);
    const search = moduleRef.get(AiTools).tools.get('catalog-search')!;
    expect(search).toMatchObject({ where: 'CatalogTools.search', description: 'Search the catalog' });
    expect(await moduleRef.get(AiTools).call('catalog-search', { query: 'kibble' })).toBe('found kibble');
    await moduleRef.close();
  });

  it('turns agent functions into `ai` tools for code that calls `ai` itself', async () => {
    const { moduleRef } = await createAi({ models: { default: new FakeModel([]) } }, { providers: [CatalogTools] });
    const model = new FakeModel([{ toolCalls: [{ name: 'product', input: { sku: 'SKU-1' } }] }]);

    const result = await generateText({ model, prompt: 'Is SKU-1 back?', tools: moduleRef.get(AiTools).select(['product']) });
    expect(result.toolResults).toEqual([
      expect.objectContaining({
        toolName: 'product',
        output: { sku: 'SKU-1', name: 'Salmon kibble', restocked: '2026-11-01T00:00:00.000Z' },
      }),
    ]);
    // The model was sent the JSON Schema of the arguments.
    expect(model.lastCall!.options.tools![0]).toMatchObject({
      name: 'product',
      description: 'Look a product up',
      inputSchema: { type: 'object', properties: { sku: { type: 'string' } }, required: ['sku'] },
    });
    await moduleRef.close();
  });

  it('reads MCP results: structured content, text blocks, isError; and undefined as null', async () => {
    const { moduleRef } = await createAi({ models: { default: new FakeModel([]) } }, { providers: [CatalogTools] });
    const tools = moduleRef.get(AiTools);
    expect(await tools.call('structured', {})).toEqual({ count: 2 });
    expect(await tools.call('texts', {})).toBe('one\ntwo');
    await expect(tools.call('failed', {})).rejects.toMatchObject({ name: 'AiToolError', message: 'Out of stock' });
    expect(await tools.call('stock', {})).toBeNull();
    await expect(tools.call('product', { sku: 1 })).rejects.toThrow('Invalid input: sku: Invalid input: expected string');
    await moduleRef.close();
  });

  it('unwraps Observables returned by exception filters', async () => {
    const recovered = { refunded: false, reason: 'Not your order' };
    const model = new FakeModel([
      { toolCalls: [{ name: 'observable-result' }] },
      (call) => `The model read: ${JSON.stringify(call.toolResults[0]!.output)}`,
      { toolCalls: [{ name: 'observable-error' }] },
      'The model saw the tool error.',
    ]);
    const { moduleRef, ai } = await createAi(
      { models: { default: model } },
      { providers: [ObservableFilterTools] },
    );

    const result = await ai.generateText({
      prompt: 'Refund order 42',
      tools: ['observable-result'],
    });
    expect(result.toolResults[0]!.output).toEqual(recovered);
    expect(model.lastCall!.toolResults[0]!.output).toEqual(recovered);
    expect(result.text).toBe(`The model read: ${JSON.stringify(recovered)}`);

    const failed = await ai.generateText({
      prompt: 'Refund order 42',
      tools: ['observable-error'],
    });
    expect(failed.toolResults[0]!.error).toBe('Not your order');
    await expect(callTool(moduleRef, 'observable-error')).rejects.toMatchObject({
      name: 'AiToolError',
      message: 'Not your order',
    });
    await expect(callTool(moduleRef, 'observable-empty')).rejects.toMatchObject({
      name: 'AiToolError',
      message: 'Internal server error',
    });
    await moduleRef.close();
  });

  it('gives an agent function the user and scopes of a caller without a request, and aborts it with the call', async () => {
    const model = new FakeModel([
      { toolCalls: [{ name: 'owner' }] },
      'done',
      { toolCalls: [{ name: 'wait' }] },
      'never',
    ]);
    const { moduleRef, ai } = await createAi({ models: { default: model } }, { providers: [JobTools] });

    const result = await ai.generateText({ prompt: 'Whose job?', tools: ['owner'], caller: { user: { id: 7 }, scopes: ['jobs:run'] } });
    expect(result.toolResults[0]!.output).toEqual({ user: { id: 7 }, request: null, scopes: ['jobs:run'] });

    const controller = new AbortController();
    const call = ai.generateText({ prompt: 'Wait', tools: ['wait'], abortSignal: controller.signal });
    await vi.waitFor(() => expect(model.calls).toHaveLength(3));
    controller.abort(new Error('job cancelled'));
    await expect(call).rejects.toThrow('job cancelled');
    await moduleRef.close();
  });

  it("gives an agent function its context: the caller's request, user and scopes, the tool call, nothing made up", async () => {
    const model = new FakeModel(() => ({ toolCalls: [{ name: 'context', id: 'call-x' }] }));
    const { moduleRef, ai } = await createAi({ models: { default: model } }, { providers: [JobTools] });
    const request = { headers: {}, user: { id: 1 } };

    const run = (caller: object) => ai.generateText({ prompt: 'Go', tools: ['context'], caller, maxSteps: 1 }).then((result) => result.toolResults[0]!.output);
    // Nothing but the caller's request, user and scopes reaches the context, whatever else the caller holds.
    expect(await run({ request, scopes: ['orders:read'], authInfo: { token: 't' } })).toEqual({
      name: 'context',
      request,
      scopes: ['orders:read'],
      toolCallId: 'call-x',
      signal: true,
      reportProgress: 'function',
      messages: 1,
    });
    // In a job: the user is the caller's; nothing the caller lacks is made up.
    expect(await run({ user: { id: 2 } })).toEqual({
      name: 'context',
      user: { id: 2 },
      toolCallId: 'call-x',
      signal: true,
      reportProgress: 'function',
      messages: 1,
    });
    await moduleRef.close();
  });

  it("runs an agent function's enhancers with the context type 'ai', which AiExecutionContext reads", async () => {
    SeenGuard.seen = [];
    const model = new FakeModel([{ toolCalls: [{ name: 'look-up', input: { id: 7 }, id: 'call-7' }] }, 'done']);
    const { moduleRef, ai } = await createAi({ models: { default: model } }, { providers: [GuardedTools, SeenGuard] });
    const request = { headers: {}, user: { id: 1 } };

    await ai.generateText({ prompt: 'Look 7 up', tools: ['look-up'], caller: { request } });
    expect(SeenGuard.seen).toEqual([
      {
        type: 'ai',
        ownType: 'ai',
        input: { id: 7 },
        rpcData: { id: 7 },
        name: 'look-up',
        request,
        toolCallId: 'call-7',
        handler: 'lookUp',
        class: 'GuardedTools',
      },
    ]);
    await moduleRef.close();
  });

  it("does not time an agent function out with the model call's timeout, nor run it twice", async () => {
    const model = new FakeModel([{ toolCalls: [{ name: 'slow' }] }, (call) => String(call.toolResults[0]!.output)]);
    const { moduleRef, ai } = await createAi({ models: { default: model } }, { providers: [JobTools] });

    const result = await ai.generateText({ prompt: 'Go', tools: ['slow'], timeout: 50 });
    expect(result.text).toBe('finally');
    expect(result.steps.map((step) => step.attempts)).toEqual([1, 1]);
    expect(model.calls).toHaveLength(2);
    await moduleRef.close();
  });

  // Both packages wrap the method with Nest's ExternalContextCreator, whose own cache would give its parameter
  // decorators the context type of whichever wrapped it first. Both import orders, so that each package goes first.
  it.each([
    ['AiModule', 'McpModule'],
    ['McpModule', 'AiModule'],
  ])("gives a method that is an MCP tool too its parameter decorators' own context type, with %s first", async (first) => {
    const ai = AiModule.forRoot({ models: { default: new FakeModel([]) } });
    const mcp = McpModule.forRoot({ name: 'stacked', version: '1.0.0' });
    const imports: DynamicModule[] = first === 'AiModule' ? [ai, mcp] : [mcp, ai];
    const moduleRef = await Test.createTestingModule({ imports, providers: [StackedTools] }).compile();
    moduleRef.useLogger(false);
    const client = await connectMcpClient(moduleRef);
    try {
      const result = await client.callTool({ name: 'seen-type', arguments: {} });
      expect((result.content as Array<{ text: string }>)[0]!.text).toBe('mcp');
      expect(await moduleRef.get(AiTools).call('seen-type', {})).toBe('ai');
    } finally {
      await client.close();
      await moduleRef.close();
    }
  });

  it('searches only the `include` modules', async () => {
    @Module({ providers: [JobTools] })
    class JobsModule {}
    @Module({ providers: [CatalogTools] })
    class CatalogModule {}

    const { moduleRef } = await createAi(
      { models: { default: new FakeModel([]) }, include: [JobsModule] },
      { imports: [JobsModule, CatalogModule] },
    );
    expect([...moduleRef.get(AiTools).tools.keys()]).toEqual(['owner', 'context', 'slow', 'wait']);
    await moduleRef.close();
  });

  it('refuses @AgentFunction() on what is not a method, or twice on one', () => {
    class Target {
      get price() {
        return 1;
      }
      twice() {}
    }
    const getter = Object.getOwnPropertyDescriptor(Target.prototype, 'price')!;
    expect(() => AgentFunction()(Target.prototype, 'price', getter)).toThrow('@AgentFunction() goes on a method, and Target.price is not one.');
    const method = Object.getOwnPropertyDescriptor(Target.prototype, 'twice')!;
    AgentFunction({ name: 'once' })(Target.prototype, 'twice', method);
    expect(() => AgentFunction()(Target.prototype, 'twice', method)).toThrow('Target.twice has @AgentFunction() twice');
  });

  describe('refuses at startup', () => {
    const compile = (...providers: Function[]) =>
      Test.createTestingModule({ imports: [AiModule.forRoot({ models: { default: new FakeModel([]) } })], providers: providers as never[] })
        .compile()
        .then((moduleRef) => moduleRef.init());

    it('two agent functions with one name', async () => {
      @Injectable()
      class Twice {
        @AgentFunction({ name: 'same' })
        a() {}
        @AgentFunction({ name: 'same' })
        b() {}
      }
      await expect(compile(Twice)).rejects.toThrow('there is already an agent function named "same" (Twice.a)');
    });

    it('a name model providers refuse as a tool name', async () => {
      @Injectable()
      class Dotted {
        @AgentFunction({ name: 'catalog.search' })
        search() {}
      }
      await expect(compile(Dotted)).rejects.toThrow(
        'Dotted.search: the name "catalog.search" is not one model providers accept as a tool name',
      );

      @Injectable()
      class TooLong {
        @AgentFunction({ name: 'x'.repeat(65) })
        search() {}
      }
      await expect(compile(TooLong)).rejects.toThrow(AiConfigurationError);
    });

    it('an agent function on a request-scoped provider', async () => {
      @Injectable({ scope: Scope.REQUEST })
      class PerRequest {
        @AgentFunction()
        lookup() {}
      }
      await expect(compile(PerRequest)).rejects.toThrow(AiConfigurationError);
    });

    it('a schema that is not a Standard Schema with JSON Schema, or not an object', async () => {
      @Injectable()
      class NotASchema {
        @AgentFunction({ inputSchema: { type: 'object' } as never })
        find() {}
      }
      await expect(compile(NotASchema)).rejects.toThrow('not a Standard Schema');

      @Injectable()
      class NotAnObject {
        @AgentFunction({ inputSchema: z.string() })
        find() {}
      }
      await expect(compile(NotAnObject)).rejects.toThrow('arguments are an object');
    });
  });
});
