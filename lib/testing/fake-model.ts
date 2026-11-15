import { APICallError } from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import type { AiUsage } from '../interfaces/ai-usage.interface.js';

type CallOptions = Parameters<MockLanguageModelV4['doGenerate']>[0];
type GenerateResult = Awaited<ReturnType<MockLanguageModelV4['doGenerate']>>;
type StreamResult = Awaited<ReturnType<MockLanguageModelV4['doStream']>>;
type StreamPart = StreamResult['stream'] extends ReadableStream<infer P> ? P : never;
type FinishReason = GenerateResult['finishReason']['unified'];

/** A tool call the fake model makes. */
export interface FakeToolCall {
  /** The tool's name. */
  name: string;
  /** Its arguments. Default `{}`. */
  input?: unknown;
  /** Default `call-<n>`. */
  id?: string;
}

/** One answer of the fake model. */
export interface FakeTurn {
  text?: string;
  /** Answered as JSON text: what `generateObject()` reads. */
  object?: unknown;
  toolCalls?: FakeToolCall[];
  /** Default: 10 input and 10 output tokens. */
  usage?: Partial<Pick<AiUsage, 'inputTokens' | 'outputTokens' | 'cachedInputTokens' | 'reasoningTokens'>>;
  /** Default `tool-calls` with tool calls, else `stop`. */
  finishReason?: FinishReason;
  /** Milliseconds before the model answers (a stream: before it opens). The call's abort cuts it short. */
  delay?: number;
  /** Milliseconds between a stream's chunks. */
  chunkDelay?: number;
  /** A stream fails with this error after its text, as a connection that breaks does. */
  streamError?: unknown;
}

/**
 * What the fake model answers a call with: text, a turn, an error (thrown, as a provider's failure: see
 * `providerError()`), or a function of the call that returns one of these.
 */
export type FakeAnswer = string | FakeTurn | Error | ((call: FakeModelCall) => FakeAnswer | Promise<FakeAnswer>);

/** A call the fake model received. */
export interface FakeModelCall {
  stream: boolean;
  /** The system prompt. */
  instructions?: string;
  /** The text of the last user message. */
  lastUserMessage?: string;
  /** The results of the tools called in the previous step, as the model reads them. */
  toolResults: Array<{ toolName: string; toolCallId: string; output: unknown }>;
  /** The names of the tools offered. */
  tools: string[];
  /** Everything the model was sent (`ai`'s provider-level call options). */
  options: CallOptions;
}

/**
 * A language model for tests, with no network: it answers each call with the next of its `answers` (text, tool
 * calls, objects, errors, delays), and records the calls. Pass a function instead of a list to answer every call
 * with it.
 *
 * ```ts
 * const model = new FakeModel([
 *   { toolCalls: [{ name: 'find-order', input: { orderId: 'ord_1' } }] },
 *   'Your order ships tomorrow.',
 * ]);
 * ```
 */
export class FakeModel extends MockLanguageModelV4 {
  readonly calls: FakeModelCall[] = [];
  private readonly answers: FakeAnswer[];
  private readonly answerAll: ((call: FakeModelCall) => FakeAnswer | Promise<FakeAnswer>) | undefined;
  private toolCallCount = 0;

  constructor(answers: FakeAnswer[] | ((call: FakeModelCall) => FakeAnswer | Promise<FakeAnswer>) = [], { provider = 'fake', modelId = 'fake-model' } = {}) {
    super({ provider, modelId });
    this.answers = Array.isArray(answers) ? [...answers] : [];
    this.answerAll = typeof answers === 'function' ? answers : undefined;
    this.doGenerate = async (options) => this.generate(await this.answer(options, false), options);
    this.doStream = async (options) => this.stream(await this.answer(options, true), options);
  }

  /** Adds answers, after the ones not used yet. */
  reply(...answers: FakeAnswer[]): this {
    this.answers.push(...answers);
    return this;
  }

  /** The last call, or `undefined`. */
  get lastCall(): FakeModelCall | undefined {
    return this.calls[this.calls.length - 1];
  }

  private async answer(options: CallOptions, stream: boolean): Promise<FakeTurn> {
    const call = describeCall(options, stream);
    this.calls.push(call);
    let answer: FakeAnswer | undefined = this.answerAll ?? this.answers.shift();
    if (answer === undefined) {
      throw new Error(`FakeModel: no answer left for call ${this.calls.length}. Give it more answers.`);
    }
    while (typeof answer === 'function') {
      answer = await answer(call);
    }
    if (answer instanceof Error) {
      throw answer;
    }
    const turn: FakeTurn = typeof answer === 'string' ? { text: answer } : answer;
    if (turn.delay) {
      await sleep(turn.delay, options.abortSignal);
    }
    return turn;
  }

  private generate(turn: FakeTurn, options: CallOptions): GenerateResult {
    options.abortSignal?.throwIfAborted();
    const text = textOf(turn);
    const toolCalls = this.toolCalls(turn);
    return {
      content: [...(text ? [{ type: 'text' as const, text }] : []), ...toolCalls],
      finishReason: finishOf(turn, toolCalls.length),
      usage: usageOf(turn),
      warnings: [],
      response: { id: `response-${this.calls.length}`, modelId: this.modelId, timestamp: new Date(0) },
    };
  }

  private stream(turn: FakeTurn, options: CallOptions): StreamResult {
    const text = textOf(turn);
    const toolCalls = this.toolCalls(turn);
    const parts: StreamPart[] = [
      { type: 'stream-start', warnings: [] },
      { type: 'response-metadata', id: `response-${this.calls.length}`, modelId: this.modelId, timestamp: new Date(0) },
    ];
    if (text) {
      parts.push({ type: 'text-start', id: 'text-0' });
      for (const delta of text.match(/\S+\s*|\s+/g) ?? []) {
        parts.push({ type: 'text-delta', id: 'text-0', delta });
      }
      parts.push({ type: 'text-end', id: 'text-0' });
    }
    parts.push(...toolCalls);
    if (turn.streamError !== undefined) {
      parts.push({ type: 'error', error: turn.streamError });
    } else {
      parts.push({ type: 'finish', finishReason: finishOf(turn, toolCalls.length), usage: usageOf(turn) });
    }

    const signal = options.abortSignal;
    let index = 0;
    return {
      stream: new ReadableStream<StreamPart>({
        pull: async (controller) => {
          if (index > 0 && turn.chunkDelay) {
            try {
              await sleep(turn.chunkDelay, signal);
            } catch (error) {
              controller.error(error);
              return;
            }
          }
          if (signal?.aborted) {
            controller.error(signal.reason);
            return;
          }
          const part = parts[index++];
          if (part === undefined) {
            controller.close();
          } else {
            controller.enqueue(part);
            if (part.type === 'error') {
              controller.close();
            }
          }
        },
      }),
    };
  }

  private toolCalls(turn: FakeTurn) {
    return (turn.toolCalls ?? []).map((call) => ({
      type: 'tool-call' as const,
      toolCallId: call.id ?? `call-${++this.toolCallCount}`,
      toolName: call.name,
      input: JSON.stringify(call.input ?? {}),
    }));
  }
}

/** `new FakeModel(answers)`. */
export function fakeModel(...answers: FakeAnswer[]): FakeModel {
  return new FakeModel(answers);
}

/**
 * A provider's HTTP error, as `@ai-sdk/*` providers throw it: retryable for 408, 409, 429 and 5xx.
 * `retryAfter` (in seconds) is sent as `retry-after`; explicit `responseHeaders` are layered over it, taking
 * precedence for a header with the same name.
 */
export function providerError(
  status: number,
  {
    message,
    retryAfter,
    responseHeaders,
  }: {
    message?: string;
    retryAfter?: number;
    responseHeaders?: Record<string, string>;
  } = {},
): APICallError {
  const headers = {
    ...(retryAfter !== undefined && { 'retry-after': String(retryAfter) }),
    ...responseHeaders,
  };
  return new APICallError({
    message: message ?? `The provider answered ${status}`,
    url: 'https://fake-provider.test/v1/messages',
    requestBodyValues: {},
    statusCode: status,
    ...(Object.keys(headers).length > 0 && { responseHeaders: headers }),
  });
}

function textOf(turn: FakeTurn): string {
  return turn.object !== undefined ? JSON.stringify(turn.object) : (turn.text ?? '');
}

function finishOf(turn: FakeTurn, toolCalls: number) {
  const unified: FinishReason = turn.finishReason ?? (toolCalls > 0 ? 'tool-calls' : 'stop');
  return { unified, raw: unified };
}

function usageOf(turn: FakeTurn): GenerateResult['usage'] {
  const input = turn.usage?.inputTokens ?? 10;
  const output = turn.usage?.outputTokens ?? 10;
  const cached = turn.usage?.cachedInputTokens ?? 0;
  const reasoning = turn.usage?.reasoningTokens ?? 0;
  return {
    inputTokens: { total: input, noCache: input - cached, cacheRead: cached, cacheWrite: undefined },
    outputTokens: { total: output, text: output - reasoning, reasoning },
  };
}

type PromptMessage = CallOptions['prompt'][number];

function describeCall(options: CallOptions, stream: boolean): FakeModelCall {
  const prompt = options.prompt;
  const system = prompt.filter((message) => message.role === 'system').map((message) => message.content as string);
  const lastUser = [...prompt].reverse().find((message) => message.role === 'user');
  const trailingTools: PromptMessage[] = [];
  for (let index = prompt.length - 1; index >= 0 && prompt[index]!.role === 'tool'; index--) {
    trailingTools.unshift(prompt[index]!);
  }
  return {
    stream,
    ...(system.length > 0 && { instructions: system.join('\n') }),
    ...(lastUser && {
      lastUserMessage: (lastUser.content as Array<{ type: string; text?: string }>)
        .filter((part) => part.type === 'text')
        .map((part) => part.text)
        .join(''),
    }),
    toolResults: trailingTools.flatMap((message) =>
      (message.content as Array<{ type: string; toolCallId: string; toolName: string; output: { type: string; value?: unknown } }>)
        .filter((part) => part.type === 'tool-result')
        .map((part) => ({ toolName: part.toolName, toolCallId: part.toolCallId, output: part.output?.value })),
    ),
    tools: (options.tools ?? []).map((tool) => tool.name),
    options,
  };
}

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal!.reason);
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
