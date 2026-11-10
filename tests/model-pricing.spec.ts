/**
 * Language-model pricing: startup validation and costs of generated and streamed calls.
 */
import { Injectable } from '@nestjs/common';
import { Test, type TestingModule } from '@nestjs/testing';
import {
  AgentFunction,
  AiBudgetExceededError,
  AiConfigurationError,
  AiModule,
  type AiModelPricing,
} from '../lib/index.js';
import { FakeModel } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

@Injectable()
class PricingTools {
  @AgentFunction()
  tick() {
    return 'done';
  }
}

describe('language model pricing', () => {
  const modules: TestingModule[] = [];

  afterEach(async () => {
    await Promise.all(modules.splice(0).map((ref) => ref.close()));
  });

  const compile = async (pricing: unknown, asyncOptions = false) => {
    const model = new FakeModel([]);
    const options = { models: { priced: { model, pricing: pricing as AiModelPricing } } };
    const ref = await Test.createTestingModule({
      imports: [
        asyncOptions
          ? AiModule.forRootAsync({ useFactory: () => options })
          : AiModule.forRoot(options),
      ],
    }).compile();
    modules.push(ref);
    return ref;
  };

  describe.each(['input', 'output', 'cachedInput'] as const)('pricing.%s', (field) => {
    it.each([-1, NaN, Infinity, -Infinity, '1', null])('rejects %s at startup', async (value) => {
      await expect(compile({ input: 1, output: 2, [field]: value })).rejects.toMatchObject({
        name: 'AiConfigurationError',
        message:
          `AiModule: the model "priced" pricing.${field} must be ` + 'a finite nonnegative number.',
      });
    });
  });

  it.each(['input', 'output'] as const)('requires pricing.%s', async (field) => {
    await expect(compile({ input: 1, output: 2, [field]: undefined })).rejects.toBeInstanceOf(
      AiConfigurationError,
    );
  });

  it.each([{ pricing: null }, { pricing: [] }, { pricing: '1' }, { pricing: 1 }])(
    'rejects malformed pricing $pricing',
    async ({ pricing }) => {
      await expect(compile(pricing)).rejects.toBeInstanceOf(AiConfigurationError);
    },
  );

  it('validates asynchronous configuration before a provider is called', async () => {
    await expect(compile({ input: NaN, output: 2 }, true)).rejects.toThrow(
      'the model "priced" pricing.input',
    );
  });

  it.each([
    undefined,
    { input: 0, output: 0 },
    { input: 0.1, output: 0.5, cachedInput: 0 },
    { input: 0.1, output: 0.5, cachedInput: 0.02 },
  ])('accepts valid pricing %j without contacting a provider', async (pricing) => {
    await expect(compile(pricing)).resolves.toBeDefined();
  });

  it.each(['generateText', 'streamText'] as const)(
    '%s preserves cached-input costs and stops at the cost limit',
    async (method) => {
      const model = new FakeModel(() => ({
        toolCalls: [{ name: 'tick' }],
        usage: { inputTokens: 10, outputTokens: 10, cachedInputTokens: 4 },
      }));
      const events: number[] = [];
      const { moduleRef, ai } = await createAi(
        {
          models: {
            default: {
              model,
              pricing: { input: 1_000_000, output: 2_000_000, cachedInput: 500_000 },
            },
          },
          onUsage: (event) => {
            events.push(event.cost!);
          },
        },
        { providers: [PricingTools] },
      );
      modules.push(moduleRef);

      const options = { prompt: 'Go', tools: ['tick'], budget: { maxCost: 28 } };
      const result =
        method === 'streamText' ? ai.streamText(options).result : ai.generateText(options);
      await expect(result).rejects.toBeInstanceOf(AiBudgetExceededError);
      expect(events).toEqual([28]); // 6 uncached + 4 * 0.5 cached + 10 * 2 output.
      expect(model.calls).toHaveLength(1);
    },
  );
});
