import type { Duration, ResiliencePolicy, RetryOptions } from '@nestjs/resilience';
import type { EmbeddingModel, LanguageModel } from 'ai';
import type { AiUsageEvent } from './ai-usage.interface.js';
import type {
  AiEmbeddingModelDefinition,
  AiEmbeddingSettings,
  AiEmbeddingUsageEvent,
} from './ai-embedding.interface.js';

/** Prices per million tokens, in a unit of your choice: what `cost` is reported in. */
export interface AiModelPricing {
  input: number;
  output: number;
  /** Input tokens read from the prompt cache. Default: `input`. */
  cachedInput?: number;
}

/** Limits on one call, over every step it makes. */
export interface AiBudget {
  /** Total tokens, input and output. A finite whole number of at least 0. */
  maxTokens?: number;
  /** In the unit of the models' `pricing`. A model without pricing counts as free. Must be finite and at least 0. */
  maxCost?: number;
}

/** Resilience settings shared by language-model and embedding calls. */
export interface AiResilienceSettings {
  /**
   * The longest a model call may take to answer: its whole response for `generateText()` and `generateObject()`,
   * its first chunk for `streamText()`, until its tools start for a step that calls them, one provider request for
   * an embedding. Applied to each attempt, as a `TimeoutPolicy`. Default `'60s'`; `false` for none.
   */
  timeout?: Duration | false;
  /**
   * Retries of a failed model call, as `@nestjs/resilience`'s `RetryPolicy` takes them: `attempts` (default 3, the
   * first included), `backoff` (default from `'1s'`, doubling, up to `'30s'`, full jitter), `retryIf`. Retried by
   * default: what the provider marks retryable (408, 409, 429, 5xx and network failures) and timeouts; a
   * `retry-after` the provider sends is honored, and one longer than `backoff.maxDelay` moves on to the fallback.
   * An embedding call has no fallback, and retries only the provider request that failed. `false` for none.
   */
  retry?: RetryOptions | false;
  /**
   * A policy of your own (`ResiliencePolicy.wrap(breaker, retry, timeout)`), or the name of a preset of
   * `ResilienceService` (`ResilienceModule.forRoot({ presets })`), in place of `retry` and `timeout`: a circuit
   * breaker or a bulkhead shared with the rest of the application. Lower-priority `retry` and `timeout` settings are
   * replaced; explicitly setting either at the same or a higher-priority level is an error. Fallback models are
   * applied around it. An embedding call runs it around each provider request, so a bulkhead or a rate limit counts
   * requests.
   */
  resilience?: ResiliencePolicy | string;
}

/** Settings a language model, the module's `defaults` and each call can set. Each field resolves call, model, defaults. */
export interface AiCallSettings extends AiResilienceSettings {
  /** Generation settings passed to the model. */
  maxOutputTokens?: number;
  temperature?: number;
  /** Provider-specific options, by provider name (`{ anthropic: { ... } }`). */
  providerOptions?: Record<string, Record<string, any>>;
  /** HTTP headers sent to the provider. */
  headers?: Record<string, string | undefined>;
}

export interface AiModelDefinition extends AiCallSettings {
  model: LanguageModel;
  pricing?: AiModelPricing;
  /** The models tried, in order, when this one fails after its retries: names of other models. */
  fallback?: string | string[];
}

/** Defaults for language-model calls. `maxSteps` and `budget` are also those of durable runs (`AiAgents.run()`). */
export interface AiCallDefaults extends AiCallSettings {
  /** Model calls one call may make while the model asks for tools. Default 10 (25 for a durable run). */
  maxSteps?: number;
  budget?: AiBudget;
}

/** `forRoot()`'s options, and what the `forRootAsync()` factory returns. */
export interface AiModuleOptions {
  /**
   * The models, by name: a `LanguageModel` (`anthropic('claude-opus-5-5')`, a gateway id such as
   * `'openai/gpt-5'`), or a definition with its pricing, fallback and settings. Calls that name no model use
   * `default`.
   */
  models?: Record<string, LanguageModel | AiModelDefinition>;
  defaults?: AiCallDefaults;
  /**
   * Named embedding models, independent of `models`. At least one of the two registries must be
   * nonempty.
   */
  embeddingModels?: Record<string, EmbeddingModel | AiEmbeddingModelDefinition>;
  /**
   * Embedding-only defaults. They do not inherit generation settings, headers or options from
   * `defaults`.
   */
  embeddingDefaults?: AiEmbeddingSettings;
  /** Called after every model call that answered, with its tokens and cost. Errors it throws are logged. */
  onUsage?: (event: AiUsageEvent) => void | Promise<void>;
  /** Called once after a successful embedding operation. Errors it throws are logged. */
  onEmbeddingUsage?: (event: AiEmbeddingUsageEvent) => void | Promise<void>;
  /**
   * The modules whose providers are searched for agent functions (`@AgentFunction()` methods). Default: every module
   * of the application.
   */
  include?: Function[];
}

/** What only the top level of `forRoot()` and `forRootAsync()` takes. */
export interface AiModuleStructure {
  /**
   * Model names `@InjectModel()` may inject, beyond those it was seen asking for when the module was defined (the
   * classes the module's file imports): for classes loaded later, such as lazily loaded modules.
   */
  modelNames?: string[];
  /**
   * Embedding-model counterpart of `modelNames`, for injection sites loaded after module
   * definition.
   */
  embeddingModelNames?: string[];
}

/** What a class passed to `forRootAsync({ useClass })` implements. */
export interface AiOptionsFactory {
  createAiOptions(): AiModuleOptions | Promise<AiModuleOptions>;
}
