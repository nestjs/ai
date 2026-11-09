/**
 * Model calls on `@nestjs/resilience`: a policy of the application's own (a circuit breaker) and a named preset in
 * place of `retry` and `timeout`, the events the policies publish, and the attempt of a resilient caller.
 */
import { subscribe, unsubscribe } from 'node:diagnostics_channel';
import type { TestingModule } from '@nestjs/testing';
import {
  CircuitBreakerPolicy,
  CircuitOpenError,
  ResilienceContext,
  ResilienceModule,
  ResiliencePolicy,
  RetryPolicy,
  type ResilienceEvent,
} from '@nestjs/resilience';
import { AiConfigurationError, AiModelError } from '../lib/index.js';
import { FakeEmbeddingModel, FakeModel, providerError } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

describe('model calls on @nestjs/resilience', () => {
  let moduleRef: TestingModule | undefined;

  afterEach(async () => {
    await moduleRef?.close();
    moduleRef = undefined;
  });

  it("runs a model's calls under a policy of the application's own, and falls back when its breaker opens", async () => {
    const breaker = new CircuitBreakerPolicy({ name: 'llm', minimumCalls: 2, slidingWindow: { type: 'count', size: 2 }, openDuration: '1m' });
    const primary = new FakeModel(() => providerError(500));
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: { model: primary, resilience: ResiliencePolicy.wrap(breaker, new RetryPolicy(false)) },
        backup: new FakeModel(() => 'from the backup'),
      },
    });
    moduleRef = ref;

    // Two failures open the breaker; each was the only attempt (the policy replaces `retry`).
    await expect(ai.generateText({ prompt: 'a', fallback: false })).rejects.toBeInstanceOf(AiModelError);
    await expect(ai.generateText({ prompt: 'b', fallback: false })).rejects.toBeInstanceOf(AiModelError);
    expect(primary.calls).toHaveLength(2);
    expect(breaker.state).toBe('open');

    // Open: rejected without a model call, as the breaker's own error, and the fallback takes it.
    await expect(ai.generateText({ prompt: 'c', fallback: false })).rejects.toBeInstanceOf(CircuitOpenError);
    expect((await ai.generateText({ prompt: 'd', fallback: 'backup' })).text).toBe('from the backup');
    expect(primary.calls).toHaveLength(2);
  });

  it('takes a preset of ResilienceService by name, and refuses one without ResilienceModule', async () => {
    const model = new FakeModel([providerError(503), 'second attempt']);
    const { moduleRef: ref, ai } = await createAi(
      { models: { default: { model, resilience: 'llm' } } },
      { imports: [ResilienceModule.forRoot({ presets: { llm: { retry: { attempts: 2, backoff: { delay: 0 } }, timeout: '5s' } } })] },
    );
    moduleRef = ref;
    expect((await ai.generateText({ prompt: 'Hi' })).text).toBe('second attempt');
    expect(model.calls).toHaveLength(2);

    const { moduleRef: without, ai: alone } = await createAi({ models: { default: { model: new FakeModel([]), resilience: 'llm' } } });
    await expect(alone.generateText({ prompt: 'Hi' })).rejects.toThrow(AiConfigurationError);
    await without.close();
  });

  it('validates fallback policies before running the primary model', async () => {
    const primary = new FakeModel(() => providerError(503));
    const fallback = new FakeModel(() => 'must not run');
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: { model: primary, fallback: 'backup' },
        backup: {
          model: fallback,
          resilience: ResiliencePolicy.wrap(new RetryPolicy(false)),
          retry: false,
        },
      },
    });
    moduleRef = ref;

    await expect(ai.generateText({ prompt: 'Hi' })).rejects.toThrow(AiConfigurationError);
    expect(primary.calls).toHaveLength(0);
    expect(fallback.calls).toHaveLength(0);
  });

  it('resolves retry and timeout independently through call, model and module settings', async () => {
    const generated = new FakeModel([
      providerError(503),
      { text: 'retried', delay: 150 },
      { text: 'model timeout', delay: 50 },
    ]);
    const embedded = new FakeEmbeddingModel([
      providerError(503),
      { embeddings: [[1]], usage: { tokens: 1 }, delay: 150 },
      { embeddings: [[2]], usage: { tokens: 1 }, delay: 50 },
    ]);
    const { moduleRef: ref, ai } = await createAi({
      models: { default: { model: generated, timeout: '100ms' } },
      defaults: {
        retry: { attempts: 2, backoff: { delay: 0 } },
        timeout: '5ms',
      },
      embeddingModels: {
        default: {
          model: embedded,
          retry: { attempts: 2, backoff: { delay: 0 } },
          timeout: '100ms',
        },
      },
      embeddingDefaults: {
        retry: false,
        timeout: '5ms',
      },
    });
    moduleRef = ref;

    expect((await ai.generateText({ prompt: 'Hi', timeout: '250ms' })).text).toBe('retried');
    expect(generated.calls).toHaveLength(2);
    expect((await ai.generateText({ prompt: 'Hi' })).text).toBe('model timeout');
    expect(generated.calls).toHaveLength(3);
    expect((await ai.embed({ value: 'hello', timeout: '250ms' })).embedding).toEqual([1]);
    expect(embedded.calls).toHaveLength(2);
    expect((await ai.embed({ value: 'hello' })).embedding).toEqual([2]);
    expect(embedded.calls).toHaveLength(3);
  });

  it('lets model resilience replace lower-priority retry and timeout defaults', async () => {
    const generated = new FakeModel([
      providerError(503),
      { text: 'model policy', delay: 20 },
    ]);
    const embedded = new FakeEmbeddingModel([
      providerError(503),
      { embeddings: [[4]], usage: { tokens: 1 }, delay: 20 },
    ]);
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: {
          model: generated,
          resilience: ResiliencePolicy.wrap(
            new RetryPolicy({ attempts: 2, backoff: { delay: 0 } }),
          ),
        },
      },
      defaults: {
        retry: { attempts: 4, backoff: { delay: 0 } },
        timeout: '5ms',
      },
      embeddingModels: {
        default: {
          model: embedded,
          resilience: ResiliencePolicy.wrap(
            new RetryPolicy({ attempts: 2, backoff: { delay: 0 } }),
          ),
        },
      },
      embeddingDefaults: {
        retry: { attempts: 4, backoff: { delay: 0 } },
        timeout: '5ms',
      },
    });
    moduleRef = ref;

    expect((await ai.generateText({ prompt: 'Hi' })).text).toBe('model policy');
    expect(generated.calls).toHaveLength(2);
    expect((await ai.embed({ value: 'hello' })).embedding).toEqual([4]);
    expect(embedded.calls).toHaveLength(2);
  });

  it('preserves explicit false retry and timeout settings on calls', async () => {
    const generated = new FakeModel(async () => {
      await new Promise<void>((resolve) => setTimeout(resolve, 40));
      return providerError(503);
    });
    const embedded = new FakeEmbeddingModel(async () => {
      await new Promise<void>((resolve) => setTimeout(resolve, 40));
      return providerError(503);
    });
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: {
          model: generated,
          retry: { attempts: 2, backoff: { delay: 0 } },
          timeout: '15ms',
        },
      },
      defaults: {
        retry: { attempts: 3, backoff: { delay: 0 } },
        timeout: '5ms',
      },
      embeddingModels: {
        default: {
          model: embedded,
          retry: { attempts: 2, backoff: { delay: 0 } },
          timeout: '15ms',
        },
      },
      embeddingDefaults: {
        retry: { attempts: 3, backoff: { delay: 0 } },
        timeout: '5ms',
      },
    });
    moduleRef = ref;

    const generationStarted = performance.now();
    const generationError = await ai
      .generateText({ prompt: 'Hi', retry: false, timeout: false })
      .catch((error: unknown) => error);
    expect(generationError).toBeInstanceOf(AiModelError);
    expect(generated.calls).toHaveLength(1);
    expect(performance.now() - generationStarted).toBeGreaterThanOrEqual(30);

    const embeddingStarted = performance.now();
    const embeddingError = await ai
      .embed({ value: 'hello', retry: false, timeout: false })
      .catch((error: unknown) => error);
    expect(embeddingError).toBeInstanceOf(AiModelError);
    expect(embedded.calls).toHaveLength(1);
    expect(performance.now() - embeddingStarted).toBeGreaterThanOrEqual(30);
  });

  it('treats null resilience settings as unspecified', async () => {
    const generated = new FakeModel([
      providerError(503),
      { text: 'model policy', delay: 20 },
    ]);
    const embedded = new FakeEmbeddingModel([
      providerError(503),
      { embeddings: [[3]], usage: { tokens: 1 }, delay: 20 },
    ]);
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: {
          model: generated,
          resilience: ResiliencePolicy.wrap(
            new RetryPolicy({ attempts: 2, backoff: { delay: 0 } }),
          ),
        },
      },
      defaults: {
        retry: { attempts: 4, backoff: { delay: 0 } },
        timeout: '5ms',
      },
      embeddingModels: {
        default: {
          model: embedded,
          resilience: ResiliencePolicy.wrap(
            new RetryPolicy({ attempts: 2, backoff: { delay: 0 } }),
          ),
        },
      },
      embeddingDefaults: {
        retry: { attempts: 4, backoff: { delay: 0 } },
        timeout: '5ms',
      },
    });
    moduleRef = ref;

    expect(
      (
        await ai.generateText({
          prompt: 'Hi',
          resilience: null as never,
          retry: null as never,
          timeout: null as never,
        })
      ).text,
    ).toBe('model policy');
    expect(generated.calls).toHaveLength(2);
    expect(
      (
        await ai.embed({
          value: 'hello',
          resilience: null as never,
          retry: null as never,
          timeout: null as never,
        })
      ).embedding,
    ).toEqual([3]);
    expect(embedded.calls).toHaveLength(2);
  });

  it('rejects conflicts at the selected resilience level or above before provider execution', async () => {
    const instance = ResiliencePolicy.wrap(new RetryPolicy(false));
    const callInstance = ResiliencePolicy.wrap(new RetryPolicy(false));
    const generated = new FakeModel(() => 'must not run');
    const embedded = new FakeEmbeddingModel(() => ({ embeddings: [[1]] }));
    const { moduleRef: ref, ai } = await createAi(
      {
        models: {
          default: {
            model: generated,
            resilience: instance,
            retry: false,
            timeout: false,
          },
        },
        defaults: { resilience: instance, timeout: false },
        embeddingModels: {
          default: { model: embedded, retry: false, timeout: false },
        },
        embeddingDefaults: { resilience: 'embedding-shared', timeout: false },
      },
      {
        imports: [
          ResilienceModule.forRoot({
            presets: { 'embedding-shared': { retry: false } },
          }),
        ],
      },
    );
    moduleRef = ref;

    await expect(
      ai.generateText({ prompt: 'model policy conflict', timeout: false }),
    ).rejects.toMatchObject({
      name: 'AiConfigurationError',
      message: expect.stringContaining('`timeout` at call'),
    });
    await expect(
      ai.generateText({
        prompt: 'call policy conflict',
        resilience: callInstance,
        timeout: false,
      }),
    ).rejects.toMatchObject({
      name: 'AiConfigurationError',
      message: expect.stringContaining(
        '`timeout` at call with `resilience` at call. Remove the listed `retry`/`timeout` settings',
      ),
    });
    await expect(ai.embed({ value: 'default policy conflict', retry: false })).rejects.toThrow(
      AiConfigurationError,
    );
    expect(generated.calls).toHaveLength(0);
    expect(embedded.calls).toHaveLength(0);
  });

  it('lets higher-priority resilience replace lower-priority retry, timeout and resilience settings', async () => {
    const generated = new FakeModel([
      providerError(503),
      { text: 'retried by call policy', delay: 20 },
    ]);
    const embedded = new FakeEmbeddingModel([
      providerError(503),
      { embeddings: [[2]], usage: { tokens: 1 }, delay: 20 },
    ]);
    const lowerPolicy = ResiliencePolicy.wrap(new RetryPolicy(false));
    const callPolicy = ResiliencePolicy.wrap(
      new RetryPolicy({ attempts: 2, backoff: { delay: 0 } }),
    );
    const { moduleRef: ref, ai } = await createAi(
      {
        models: {
          default: {
            model: generated,
            resilience: lowerPolicy,
            retry: { attempts: 3, backoff: { delay: 0 } },
            timeout: '5ms',
          },
        },
        defaults: {
          resilience: lowerPolicy,
          retry: { attempts: 4, backoff: { delay: 0 } },
          timeout: '5ms',
        },
        embeddingModels: {
          default: {
            model: embedded,
            resilience: lowerPolicy,
            retry: { attempts: 3, backoff: { delay: 0 } },
            timeout: '5ms',
          },
        },
        embeddingDefaults: {
          resilience: lowerPolicy,
          retry: { attempts: 4, backoff: { delay: 0 } },
          timeout: '5ms',
        },
      },
      {
        imports: [
          ResilienceModule.forRoot({
            presets: {
              'call-embedding': {
                retry: { attempts: 2, backoff: { delay: 0 } },
                timeout: '1s',
              },
            },
          }),
        ],
      },
    );
    moduleRef = ref;

    expect(
      (await ai.generateText({ prompt: 'Hi', resilience: callPolicy })).text,
    ).toBe('retried by call policy');
    expect(generated.calls).toHaveLength(2);
    expect(
      (
        await ai.embed({
          value: 'hello',
          resilience: 'call-embedding',
        })
      ).embedding,
    ).toEqual([2]);
    expect(embedded.calls).toHaveLength(2);
  });

  it("publishes the policies' events, named after the model and labeled with the call", async () => {
    const events: ResilienceEvent[] = [];
    const listener = (event: unknown) => void events.push(event as ResilienceEvent);
    subscribe('nestjs:resilience:retry', listener);
    subscribe('nestjs:resilience:fallback', listener);
    try {
      const { moduleRef: ref, ai } = await createAi({
        models: { default: { model: new FakeModel(() => providerError(429)), fallback: 'backup' }, backup: new FakeModel(() => 'ok') },
        defaults: { retry: { attempts: 2, backoff: { delay: 0 } } },
      });
      moduleRef = ref;
      await ai.generateText({ prompt: 'Hi', label: 'triage' });
    } finally {
      unsubscribe('nestjs:resilience:retry', listener);
      unsubscribe('nestjs:resilience:fallback', listener);
    }
    expect(events).toEqual([
      expect.objectContaining({ type: 'retry', policy: 'ai:default retry', source: 'triage', attempt: 1 }),
      expect.objectContaining({ type: 'fallback', policy: 'ai:default fallback', source: 'triage' }),
    ]);
  });

  it('stops when the resilient attempt it runs in is aborted', async () => {
    const model = new FakeModel(() => ({ text: 'slow', delay: 5_000 }));
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
    moduleRef = ref;

    const controller = new AbortController();
    const call = new ResilienceContext().run({ signal: controller.signal }, () => ai.generateText({ prompt: 'Hi' }));
    await vi.waitFor(() => expect(model.calls).toHaveLength(1));
    controller.abort(new Error('the route timed out'));
    await expect(call).rejects.toThrow('the route timed out');
    expect(model.calls).toHaveLength(1);
  });
});
