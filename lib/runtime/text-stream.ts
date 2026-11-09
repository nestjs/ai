import { createTextStreamResponse, createUIMessageStreamResponse, type FinishReason, type ModelMessage, type UIMessageChunk } from 'ai';
import { withKeepAlive } from '../http/keep-alive.util.js';
import { toNodeResponse, writeWebResponse } from '../http/node-response.util.js';
import type { Duration } from '@nestjs/resilience';
import type { AiStep, AiTextResult } from '../interfaces/ai-result.interface.js';
import type { AiUsage } from '../interfaces/ai-usage.interface.js';
import { toMilliseconds } from '../utils/duration.util.js';

/** What the browser is told about a failure. Default: `An error occurred.`, so that no server detail leaks. */
export interface AiUIMessageStreamOptions {
  errorText?: (error: unknown) => string;
}

export interface AiStreamResponseOptions extends AiUIMessageStreamOptions {
  /** Default 200. */
  status?: number;
  headers?: Record<string, string>;
  /**
   * A comment line (`: keep-alive`) sent after this long without a chunk, while a tool runs or the model thinks, so
   * that proxies and load balancers do not close the connection as idle. `pipeUIMessageStreamToResponse()` sends
   * nothing before the first chunk, keep-alive included, so that a failure before it can still be retried, fall back
   * to another model or reach the route's exception filters. Default `'15s'`; `false` for none.
   */
  keepAlive?: Duration | false;
}

const DEFAULT_KEEP_ALIVE_MS = 15_000;

type Entry = { chunk: UIMessageChunk } | { error: unknown };

const DEFAULT_ERROR_TEXT = 'An error occurred.';

/**
 * A streamed answer: `streamText()`'s result. The call runs once, from the moment it is made, whatever reads it: the
 * UI message stream (`toUIMessageStream()`, `pipeUIMessageStreamToResponse()`), the text (`textStream`,
 * `pipeTextStreamToResponse()`), or the promises, which settle when the call ends. Each reader gets the whole
 * stream from its start. A reader that cancels (a browser that goes away) aborts the call.
 */
export class AiTextStream {
  /** Everything about the call, when it ends; rejects with its error. */
  readonly result: Promise<AiTextResult>;
  private readonly entries: Entry[] = [];
  private readonly waiting = new Set<() => void>();
  private ended = false;
  /** Settles with the first chunk, or with the call's error when it fails before one. */
  private readonly opened: Promise<void>;

  /** @internal */
  constructor(
    private readonly controller: AbortController,
    produce: (emit: (chunk: UIMessageChunk) => void) => Promise<AiTextResult>,
  ) {
    let open!: () => void;
    let fail!: (error: unknown) => void;
    this.opened = new Promise<void>((resolve, reject) => {
      open = resolve;
      fail = reject;
    });
    const push = (entry: Entry) => {
      this.entries.push(entry);
      for (const wake of this.waiting) {
        wake();
      }
      this.waiting.clear();
    };
    const emit = (chunk: UIMessageChunk) => {
      if (!this.ended) {
        push({ chunk });
        open();
      }
    };
    this.result = Promise.resolve()
      .then(() => produce(emit))
      .then(
        (result) => {
          this.ended = true;
          open();
          this.wakeAll();
          return result;
        },
        (error: unknown) => {
          push({ error });
          this.ended = true;
          fail(error);
          this.wakeAll();
          throw error;
        },
      );
    // Rejections are the readers' to handle: an unread stream must not crash the process.
    this.result.catch(() => {});
    this.opened.catch(() => {});
  }

  get text(): Promise<string> {
    return this.result.then((result) => result.text);
  }

  get usage(): Promise<AiUsage> {
    return this.result.then((result) => result.usage);
  }

  get steps(): Promise<AiStep[]> {
    return this.result.then((result) => result.steps);
  }

  get finishReason(): Promise<FinishReason> {
    return this.result.then((result) => result.finishReason);
  }

  get responseMessages(): Promise<ModelMessage[]> {
    return this.result.then((result) => result.responseMessages);
  }

  /** The text as it is generated, over every step. Throws the call's error. */
  get textStream(): AsyncIterable<string> {
    const entries = this.read();
    return {
      [Symbol.asyncIterator]: async function* () {
        for await (const entry of entries) {
          if ('error' in entry) {
            throw entry.error;
          }
          if (entry.chunk.type === 'text-delta') {
            yield entry.chunk.delta;
          }
        }
      },
    };
  }

  /**
   * The AI SDK's UI message stream: what `useChat()` reads. One message: `start`, each step's chunks (text, tool
   * calls and their results), `finish`; or an `error` chunk when the call fails.
   */
  toUIMessageStream({ errorText = () => DEFAULT_ERROR_TEXT }: AiUIMessageStreamOptions = {}): ReadableStream<UIMessageChunk> {
    const entries = this.read()[Symbol.asyncIterator]();
    return new ReadableStream<UIMessageChunk>({
      pull: async (controller) => {
        const next = await entries.next();
        if (next.done) {
          controller.close();
        } else if ('error' in next.value) {
          controller.enqueue({ type: 'error', errorText: safely(errorText, next.value.error) });
        } else {
          controller.enqueue(next.value.chunk);
        }
      },
      cancel: () => this.abort(),
    });
  }

  /** The UI message stream as a web-standard `Response` (server-sent events). */
  toUIMessageStreamResponse({ status, headers, keepAlive, ...options }: AiStreamResponseOptions = {}): Response {
    const response = createUIMessageStreamResponse({
      stream: this.toUIMessageStream(options),
      ...(status !== undefined && { status }),
      ...(headers && { headers }),
    });
    const interval = keepAlive === false ? undefined : toMilliseconds(keepAlive ?? DEFAULT_KEEP_ALIVE_MS, 'keepAlive');
    if (!interval || !response.body) {
      return response;
    }
    return new Response(withKeepAlive(response.body, interval), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  /**
   * Sends the UI message stream to the browser, as server-sent events, on the response `@Res()` injects (Express,
   * Fastify or the Node adapter). Resolves when the stream has been sent. A call that fails before its first chunk
   * rejects instead, with nothing sent, so the route's exception filters answer it.
   */
  async pipeUIMessageStreamToResponse(response: unknown, options: AiStreamResponseOptions = {}): Promise<void> {
    await this.send(response, () => this.toUIMessageStreamResponse(options));
  }

  /** Sends the text alone, as `text/plain` chunks; a failure after the first chunk ends the response early. */
  async pipeTextStreamToResponse(
    response: unknown,
    { status, headers }: Omit<AiStreamResponseOptions, 'errorText' | 'keepAlive'> = {},
  ): Promise<void> {
    const text = this.textStream[Symbol.asyncIterator]();
    await this.send(response, () =>
      createTextStreamResponse({
        stream: new ReadableStream<string>({
          pull: async (controller) => {
            const next = await text.next();
            if (next.done) {
              controller.close();
            } else {
              controller.enqueue(next.value);
            }
          },
          cancel: () => this.abort(),
        }),
        ...(status !== undefined && { status }),
        ...(headers && { headers }),
      }),
    );
  }

  /** Aborts the call: the model call in flight and the tools running. */
  abort(reason?: unknown): void {
    this.controller.abort(reason ?? new DOMException('The stream was cancelled', 'AbortError'));
  }

  private async send(platformResponse: unknown, create: () => Response): Promise<void> {
    const node = toNodeResponse(platformResponse);
    const gone = () => {
      if (!node.res.writableFinished) {
        this.abort();
      }
    };
    node.res.once('close', gone);
    try {
      await this.opened;
      node.open();
      await writeWebResponse(create(), node.res);
    } catch (error) {
      this.abort(error);
      throw error;
    } finally {
      node.res.off('close', gone);
    }
  }

  /** Every entry from the first, then the new ones as they come, until the call ends. */
  private read(): AsyncIterable<Entry> {
    return {
      [Symbol.asyncIterator]: async function* (this: AiTextStream) {
        let position = 0;
        for (;;) {
          if (position < this.entries.length) {
            yield this.entries[position++]!;
            continue;
          }
          if (this.ended) {
            return;
          }
          await new Promise<void>((resolve) => {
            this.waiting.add(resolve);
          });
        }
      }.bind(this),
    };
  }

  private wakeAll() {
    for (const wake of this.waiting) {
      wake();
    }
    this.waiting.clear();
  }
}

function safely(errorText: (error: unknown) => string, error: unknown): string {
  try {
    return errorText(error);
  } catch {
    return DEFAULT_ERROR_TEXT;
  }
}
