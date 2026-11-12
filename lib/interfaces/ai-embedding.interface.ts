import type { EmbeddingModel, EmbeddingModelUsage, EmbedManyResult, EmbedResult } from 'ai';
import type { AiResilienceSettings } from './ai-module-options.interface.js';
import type { AiModelInfo } from './ai-result.interface.js';

/** Embedding settings, independent of language-model generation defaults. */
export interface AiEmbeddingSettings extends AiResilienceSettings {
  /** Provider-specific options, by provider name. */
  providerOptions?: Record<string, Record<string, any>>;
  /** HTTP headers sent to the embedding provider. */
  headers?: Record<string, string>;
  /** Maximum concurrent SDK batch requests within one `embedMany()` operation. Default 4. */
  maxParallelCalls?: number;
}

/**
 * Price per million input tokens, in the unit used for reported `cost`.
 * The price must be finite and nonnegative; invalid pricing fails at startup.
 */
export interface AiEmbeddingModelPricing {
  input: number;
}

/** A registered SDK embedding model with its own embedding settings. */
export interface AiEmbeddingModelDefinition extends AiEmbeddingSettings {
  model: EmbeddingModel;
  pricing?: AiEmbeddingModelPricing;
}

/** A registered embedding name, or an SDK embedding-model object passed directly to a call. */
export type AiEmbeddingModelReference = string | Exclude<EmbeddingModel, string>;

/** Options shared by single and batch embedding calls. */
export interface AiEmbeddingCallOptions extends AiEmbeddingSettings {
  /** Name in `embeddingModels` (default `'default'`), or a direct SDK model object. */
  model?: AiEmbeddingModelReference;
  /**
   * Cancels in-flight provider requests and retry backoff. `timeout` covers one provider request,
   * so this is how to set a deadline for the whole operation (`AbortSignal.timeout(ms)`).
   */
  abortSignal?: AbortSignal;
  /** Names this operation in usage and resilience events. */
  label?: string;
  /** Called after a successful operation, after the module's embedding usage listener. */
  onEmbeddingUsage?: (event: AiEmbeddingUsageEvent) => void | Promise<void>;
}

export interface AiEmbedOptions extends AiEmbeddingCallOptions {
  value: string;
}

export interface AiEmbedManyOptions extends AiEmbeddingCallOptions {
  values: string[];
}

/** SDK single-embedding result with the resolved model and attempt count. */
export interface AiEmbedResult extends EmbedResult {
  model: AiModelInfo;
  /** Provider requests made, retries included. */
  attempts: number;
  /** Cost of the reported usage; absent without model pricing. */
  cost?: number;
}

/**
 * SDK batch result, in input order. A failed operation rejects rather than returning partial
 * results.
 */
export interface AiEmbedManyResult extends EmbedManyResult {
  model: AiModelInfo;
  /** Provider requests made, retries included. */
  attempts: number;
  /** Cost of the reported usage; absent without model pricing. */
  cost?: number;
}

/**
 * One successful embedding operation. Not a billing ledger: a provider may bill a request that
 * timed out, and an operation that fails reports nothing, including the requests it completed.
 * Input contents and output vectors are excluded.
 */
export interface AiEmbeddingUsageEvent {
  operation: 'embed' | 'embedMany';
  label?: string;
  model: AiModelInfo;
  inputCount: number;
  usage: EmbeddingModelUsage;
  /** Provider requests made, retries included. */
  attempts: number;
  /** Total duration of the operation, including retries and backoff. */
  durationMs: number;
  cost?: number;
}
