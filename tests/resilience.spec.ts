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

const shared = () => ResiliencePolicy.wrap(new RetryPolicy(false));

describe('model calls on @nestjs/resilience', () => {
  let moduleRef: TestingModule | undefined;

  afterEach(async () => {
    vi.useRealTimers();
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

  it("does not hold a call's settings against a fallback it never reaches", async () => {
    const primary = new FakeModel(() => 'from the primary');
    const fallback = new FakeModel(() => 'must not run');
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: { model: primary, fallback: 'backup' },
        backup: { model: fallback, resilience: ResiliencePolicy.wrap(new RetryPolicy(false)) },
      },
    });
    moduleRef = ref;

    expect((await ai.generateText({ prompt: 'Hi', timeout: '2s' })).text).toBe('from the primary');
    expect(primary.calls).toHaveLength(1);
    expect(fallback.calls).toHaveLength(0);
  });

  it.each([{ timeout: '2s' as const }, { retry: false as const }])('runs a fallback under its own policy when a call sets %j', async (settings) => {
    const primary = new FakeModel(() => providerError(503));
    const fallback = new FakeModel([providerError(503), 'from the backup']);
    const later = new FakeModel(() => 'must not run');
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: { model: primary, retry: false, fallback: ['backup', 'later'] },
        backup: {
          model: fallback,
          resilience: ResiliencePolicy.wrap(new RetryPolicy({ attempts: 2, backoff: { delay: 0 } })),
        },
        later,
      },
    });
    moduleRef = ref;

    expect((await ai.generateText({ prompt: 'Hi', ...settings })).text).toBe('from the backup');
    expect(primary.calls).toHaveLength(1);
    expect(fallback.calls).toHaveLength(2);
    expect(later.calls).toHaveLength(0);
  });

  it('goes on to the next fallback when the policy of a fallback gives up', async () => {
    const primary = new FakeModel(() => providerError(503));
    const fallback = new FakeModel(() => providerError(503));
    const later = new FakeModel(() => 'from the later fallback');
    const { moduleRef: ref, ai } = await createAi({
      models: {
        default: { model: primary, fallback: ['backup', 'later'] },
        backup: { model: fallback, resilience: ResiliencePolicy.wrap(new RetryPolicy(false)) },
        later,
      },
    });
    moduleRef = ref;

    expect((await ai.generateText({ prompt: 'Hi', retry: false })).text).toBe('from the later fallback');
    expect(fallback.calls).toHaveLength(1);
    expect(later.calls).toHaveLength(1);
  });

  it.each([
    {
      where: 'beside the policy of a model',
      options: () => ({ models: { default: { model: new FakeModel(() => 'x'), resilience: shared(), retry: false as const } } }),
      message: 'AiModule: the model "default" cannot use `retry` at model with `resilience` at model',
    },
    {
      where: 'on a model that takes the policy of the defaults',
      options: () => ({ models: { default: { model: new FakeModel(() => 'x'), timeout: '10s' as const } }, defaults: { resilience: shared() } }),
      message: 'AiModule: the model "default" cannot use `timeout` at model with `resilience` at module defaults',
    },
    {
      where: 'beside the policy of the defaults',
      options: () => ({ models: { default: new FakeModel(() => 'x') }, defaults: { resilience: shared(), timeout: false as const } }),
      message: 'AiModule: `defaults` cannot use `timeout` at module defaults with `resilience` at module defaults',
    },
    {
      where: 'on a model that only serves as a fallback',
      options: () => ({
        models: {
          default: { model: new FakeModel(() => 'x'), fallback: 'backup' },
          backup: { model: new FakeModel(() => 'y'), resilience: shared(), retry: false as const },
        },
      }),
      message: 'AiModule: the model "backup" cannot use `retry` at model',
    },
    {
      where: 'on an embedding model that takes the policy of the defaults',
      options: () => ({
        models: { default: new FakeModel(() => 'x') },
        embeddingModels: { default: { model: new FakeEmbeddingModel(() => ({ embeddings: [[1]] })), retry: false as const } },
        embeddingDefaults: { resilience: shared() },
      }),
      message: 'AiModule: the embedding model "default" cannot use `retry` at model with `resilience` at embedding defaults',
    },
    {
      where: 'beside the policy of the embedding defaults',
      options: () => ({
        models: { default: new FakeModel(() => 'x') },
        embeddingDefaults: { resilience: shared(), timeout: '1s' as const },
      }),
      message: 'AiModule: `embeddingDefaults` cannot use `timeout` at embedding defaults',
    },
  ])('fails at startup with a retry or timeout $where', async ({ options, message }) => {
    await expect(createAi(options())).rejects.toMatchObject({
      name: 'AiConfigurationError',
      message: expect.stringContaining(message),
    });
  });

  it('resolves retry and timeout independently through call, model and module settings', async () => {
    vi.useFakeTimers();
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

    const callTimedGeneration = ai.generateText({
      prompt: 'Hi',
      timeout: '250ms',
    });
    await vi.advanceTimersByTimeAsync(150);
    expect((await callTimedGeneration).text).toBe('retried');
    expect(generated.calls).toHaveLength(2);

    const modelTimedGeneration = ai.generateText({ prompt: 'Hi' });
    await vi.advanceTimersByTimeAsync(50);
    expect((await modelTimedGeneration).text).toBe('model timeout');
    expect(generated.calls).toHaveLength(3);

    const callTimedEmbedding = ai.embed({ value: 'hello', timeout: '250ms' });
    await vi.advanceTimersByTimeAsync(150);
    expect((await callTimedEmbedding).embedding).toEqual([1]);
    expect(embedded.calls).toHaveLength(2);

    const modelTimedEmbedding = ai.embed({ value: 'hello' });
    await vi.advanceTimersByTimeAsync(50);
    expect((await modelTimedEmbedding).embedding).toEqual([2]);
    expect(embedded.calls).toHaveLength(3);
  });

  it('lets model resilience replace lower-priority retry and timeout defaults', async () => {
    vi.useFakeTimers();
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

    const generation = ai.generateText({ prompt: 'Hi' });
    await vi.advanceTimersByTimeAsync(20);
    expect((await generation).text).toBe('model policy');
    expect(generated.calls).toHaveLength(2);

    const embedding = ai.embed({ value: 'hello' });
    await vi.advanceTimersByTimeAsync(20);
    expect((await embedding).embedding).toEqual([4]);
    expect(embedded.calls).toHaveLength(2);
  });

  it('preserves explicit false retry and timeout settings on calls', async () => {
    vi.useFakeTimers();
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

    const generation = ai
      .generateText({ prompt: 'Hi', retry: false, timeout: false })
      .catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(40);
    const generationError = await generation;
    expect(generationError).toBeInstanceOf(AiModelError);
    expect((generationError as AiModelError).cause).toMatchObject({
      statusCode: 503,
    });
    expect((generationError as AiModelError).attempts).toHaveLength(1);
    expect(generated.calls).toHaveLength(1);

    const embedding = ai
      .embed({ value: 'hello', retry: false, timeout: false })
      .catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(40);
    const embeddingError = await embedding;
    expect(embeddingError).toBeInstanceOf(AiModelError);
    expect((embeddingError as AiModelError).cause).toMatchObject({
      statusCode: 503,
    });
    expect((embeddingError as AiModelError).attempts).toHaveLength(1);
    expect(embedded.calls).toHaveLength(1);
  });

  it('treats null resilience settings as unspecified', async () => {
    vi.useFakeTimers();
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

    const generation = ai.generateText({
      prompt: 'Hi',
      resilience: null as never,
      retry: null as never,
      timeout: null as never,
    });
    await vi.advanceTimersByTimeAsync(20);
    expect((await generation).text).toBe('model policy');
    expect(generated.calls).toHaveLength(2);

    const embedding = ai.embed({
      value: 'hello',
      resilience: null as never,
      retry: null as never,
      timeout: null as never,
    });
    await vi.advanceTimersByTimeAsync(20);
    expect((await embedding).embedding).toEqual([3]);
    expect(embedded.calls).toHaveLength(2);
  });

  it("rejects a call's settings beside the policy that applies before provider execution", async () => {
    const instance = ResiliencePolicy.wrap(new RetryPolicy(false));
    const callInstance = ResiliencePolicy.wrap(new RetryPolicy(false));
    const generated = new FakeModel(() => 'must not run');
    const embedded = new FakeEmbeddingModel(() => ({ embeddings: [[1]] }));
    const { moduleRef: ref, ai } = await createAi(
      {
        models: { default: { model: generated, resilience: instance } },
        embeddingModels: { default: { model: embedded } },
        embeddingDefaults: { resilience: 'embedding-shared' },
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
        '`timeout` at call with `resilience` at call',
      ),
    });
    await expect(ai.embed({ value: 'default policy conflict', retry: false })).rejects.toThrow(
      AiConfigurationError,
    );
    expect(generated.calls).toHaveLength(0);
    expect(embedded.calls).toHaveLength(0);
  });

  it('identifies an unnamed model by provider and model id in configuration errors', async () => {
    const unnamed = new FakeModel(() => 'must not run');
    const { moduleRef: ref, ai } = await createAi({
      models: { default: { model: new FakeModel(() => 'default') } },
    });
    moduleRef = ref;

    const error = await ai
      .generateText({
        prompt: 'Hi',
        model: unnamed,
        resilience: ResiliencePolicy.wrap(new RetryPolicy(false)),
        retry: false,
      })
      .catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(AiConfigurationError);
    expect(error).toMatchObject({
      message: expect.stringContaining('Model "fake:fake-model"'),
    });
    expect(unnamed.calls).toHaveLength(0);
  });

  it('lets higher-priority resilience replace lower-priority retry, timeout and resilience settings', async () => {
    vi.useFakeTimers();
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
        models: { default: { model: generated, resilience: lowerPolicy } },
        defaults: {
          retry: { attempts: 4, backoff: { delay: 0 } },
          timeout: '5ms',
        },
        embeddingModels: { default: { model: embedded, resilience: lowerPolicy } },
        embeddingDefaults: {
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

    const generation = ai.generateText({
      prompt: 'Hi',
      resilience: callPolicy,
    });
    await vi.advanceTimersByTimeAsync(20);
    expect((await generation).text).toBe('retried by call policy');
    expect(generated.calls).toHaveLength(2);

    const embedding = ai.embed({
      value: 'hello',
      resilience: 'call-embedding',
    });
    await vi.advanceTimersByTimeAsync(20);
    expect((await embedding).embedding).toEqual([2]);
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
