import type { ResiliencePolicy } from '@nestjs/resilience';
import {
  embed,
  embedMany,
  gateway,
  wrapEmbeddingModel,
  type EmbeddingModel,
  type EmbedManyResult,
  type EmbedResult,
} from 'ai';
import { AiConfigurationError } from '../errors/ai.errors.js';
import type {
  AiEmbeddingCallOptions,
  AiEmbeddingSettings,
  AiEmbeddingUsageEvent,
} from '../interfaces/ai-embedding.interface.js';
import type { AiModuleOptions } from '../interfaces/ai-module-options.interface.js';
import type {
  AiEmbeddingModels,
  ResolvedEmbeddingModel,
} from '../models/ai-embedding-models.service.js';
import { resolveModelPolicy, runStep, type PolicyPlan } from './model-policy.js';

const NEVER_ABORTED = new AbortController().signal;
const DEFAULT_MAX_PARALLEL_CALLS = 4;

/** One operation's state. Clients and explicitly supplied resilience policies remain shared. */
export interface EmbeddingPlan extends PolicyPlan<ResolvedEmbeddingModel> {
  model: ResolvedEmbeddingModel;
  label?: string;
  settings: Pick<AiEmbeddingSettings, 'headers' | 'providerOptions'> & { maxParallelCalls: number };
  listeners: Array<(event: AiEmbeddingUsageEvent) => void | Promise<void>>;
}

/**
 * Resolves only embedding settings: language generation defaults never reach an embedding provider.
 */
export function createEmbeddingPlan(
  options: AiEmbeddingCallOptions,
  models: AiEmbeddingModels,
  moduleOptions: AiModuleOptions,
  preset: (name: string) => ResiliencePolicy,
): EmbeddingPlan {
  if (typeof options !== 'object' || options === null) {
    throw new AiConfigurationError(
      'Pass the embedding call its options: { value } or { values }, and the rest.',
    );
  }
  const model = models.resolve(options.model);
  const defaults = moduleOptions.embeddingDefaults ?? {};
  const pick = <K extends keyof AiEmbeddingSettings>(key: K): AiEmbeddingSettings[K] =>
    options[key] ?? model.settings[key] ?? defaults[key];
  const maxParallelCalls = pick('maxParallelCalls') ?? DEFAULT_MAX_PARALLEL_CALLS;
  if (!Number.isSafeInteger(maxParallelCalls) || maxParallelCalls < 1) {
    throw new AiConfigurationError('`maxParallelCalls` must be a finite positive whole number.');
  }
  const policy = resolveModelPolicy(
    model,
    [
      { name: 'call', settings: options },
      { name: 'model', settings: model.settings },
      { name: 'embedding defaults', settings: defaults },
    ],
    preset,
    'ai:embedding',
  );
  return {
    model,
    chain: [model],
    signal: options.abortSignal ?? NEVER_ABORTED,
    source: options.label ?? 'AiService',
    fallbackIf: () => false,
    policyFor: () => policy,
    ...(options.label !== undefined && { label: options.label }),
    settings: {
      maxParallelCalls,
      ...(pick('headers') !== undefined && { headers: pick('headers') }),
      ...(pick('providerOptions') !== undefined && { providerOptions: pick('providerOptions') }),
    },
    listeners: [moduleOptions.onEmbeddingUsage, options.onEmbeddingUsage].filter(
      (listener): listener is (event: AiEmbeddingUsageEvent) => void | Promise<void> =>
        typeof listener === 'function',
    ),
  };
}

interface EmbeddingExecution<R> {
  result: R;
  event: AiEmbeddingUsageEvent;
  attempts: number;
  cost?: number;
}

/**
 * The SDK splits `values` into provider requests. The policy runs around each of them, through the
 * SDK's middleware, so a timeout or a retry covers one request and never repeats the others. When a
 * request fails for good, its siblings are aborted, so that a failed operation does not keep
 * consuming provider capacity.
 */
export function runEmbedding(
  plan: EmbeddingPlan,
  input: { value: string },
): Promise<EmbeddingExecution<EmbedResult>>;
export function runEmbedding(
  plan: EmbeddingPlan,
  input: { values: string[] },
): Promise<EmbeddingExecution<EmbedManyResult>>;
export async function runEmbedding(
  plan: EmbeddingPlan,
  input: { value: string } | { values: string[] },
): Promise<EmbeddingExecution<EmbedResult | EmbedManyResult>> {
  const started = performance.now();
  const many = 'values' in input;
  if (
    many
      ? !Array.isArray(input.values) || [...input.values].some((value) => typeof value !== 'string')
      : typeof input.value !== 'string'
  ) {
    throw new AiConfigurationError(
      many ? '`values` must be an array of strings.' : '`value` must be a string.',
    );
  }
  const values = many ? [...input.values] : undefined;
  const controller = new AbortController();
  const signal = AbortSignal.any([plan.signal, controller.signal]);
  let attempts = 0;
  const wrapped = wrapEmbeddingModel({
    model: modelObject(plan.model.model),
    middleware: {
      wrapEmbed: async ({ params, model }) => {
        const { value, attempts: used } = await runStep({ ...plan, signal }, async (attempt) => {
          attempt.signal.throwIfAborted();
          return model.doEmbed({ ...params, abortSignal: attempt.signal });
        });
        attempts += used;
        return value;
      },
    },
  });
  let result: EmbedResult | EmbedManyResult;
  try {
    signal.throwIfAborted();
    const settings = {
      model: wrapped,
      headers: plan.settings.headers,
      providerOptions: plan.settings.providerOptions,
      abortSignal: signal,
      maxRetries: 0,
    };
    result =
      values !== undefined
        ? await embedMany({
            ...settings,
            values,
            maxParallelCalls: plan.settings.maxParallelCalls,
          })
        : await embed({ ...settings, value: (input as { value: string }).value });
    signal.throwIfAborted();
  } finally {
    controller.abort();
  }
  const pricing = plan.model.pricing;
  const cost =
    pricing && Number.isFinite(result.usage.tokens)
      ? (result.usage.tokens * pricing.input) / 1_000_000
      : undefined;
  return {
    result,
    attempts,
    ...(cost !== undefined && { cost }),
    event: {
      operation: many ? 'embedMany' : 'embed',
      ...(plan.label !== undefined && { label: plan.label }),
      model: plan.model.info,
      inputCount: values?.length ?? 1,
      usage: { tokens: result.usage.tokens },
      attempts,
      durationMs: Math.round(performance.now() - started),
      ...(cost !== undefined && { cost }),
    },
  };
}

type ModelObject = Parameters<typeof wrapEmbeddingModel>[0]['model'];

/**
 * `wrapEmbeddingModel()` takes model objects. An id resolves as the SDK resolves it: with its
 * default provider, the gateway unless the application set another.
 */
function modelObject(model: EmbeddingModel): ModelObject {
  if (typeof model !== 'string') {
    return model as ModelObject;
  }
  const provider = (globalThis.AI_SDK_DEFAULT_PROVIDER ?? gateway) as unknown as {
    embeddingModel(modelId: string): ModelObject;
  };
  return provider.embeddingModel(model);
}
