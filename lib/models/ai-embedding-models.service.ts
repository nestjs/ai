import { Inject, Injectable, Optional } from '@nestjs/common';
import type { EmbeddingModel } from 'ai';
import {
  AI_EMBEDDING_MODEL_OVERRIDES,
  AI_MODULE_OPTIONS,
  DEFAULT_MODEL_NAME,
} from '../ai.constants.js';
import { AiConfigurationError } from '../errors/ai.errors.js';
import type {
  AiEmbeddingModelDefinition,
  AiEmbeddingModelPricing,
  AiEmbeddingModelReference,
  AiEmbeddingSettings,
} from '../interfaces/ai-embedding.interface.js';
import type { AiModuleOptions } from '../interfaces/ai-module-options.interface.js';
import type { AiModelInfo } from '../interfaces/ai-result.interface.js';
import { assertPolicyLevels } from '../runtime/model-policy.js';
import { assertModelRegistries } from './model-registry.util.js';

/** One resolved embedding model. Internal; embedding calls never switch to fallback models. */
export interface ResolvedEmbeddingModel {
  info: AiModelInfo;
  model: EmbeddingModel;
  pricing?: AiEmbeddingModelPricing;
  settings: AiEmbeddingSettings;
}

/** The configured embedding models, independent of language-model names. */
@Injectable()
export class AiEmbeddingModels {
  private readonly models = new Map<string, ResolvedEmbeddingModel>();

  constructor(
    @Inject(AI_MODULE_OPTIONS) options: AiModuleOptions,
    @Optional()
    @Inject(AI_EMBEDDING_MODEL_OVERRIDES)
    overrides?: Record<string, EmbeddingModel> | null,
  ) {
    assertModelRegistries(options);
    assertSettings(options.embeddingDefaults, 'AiModule: `embeddingDefaults`');
    const defaults = options.embeddingDefaults ?? {};
    assertPolicyLevels('AiModule: `embeddingDefaults`', [{ name: 'embedding defaults', settings: defaults }]);
    for (const [name, entry] of Object.entries(options.embeddingModels ?? {})) {
      const { model, pricing, ...settings } = isDefinition(entry) ? entry : { model: entry };
      assertModel(model, `AiModule: the embedding model "${name}"`);
      assertSettings(settings, `AiModule: the embedding model "${name}"`);
      assertPolicyLevels(`AiModule: the embedding model "${name}"`, [
        { name: 'model', settings },
        { name: 'embedding defaults', settings: defaults },
      ]);
      if (
        pricing !== undefined &&
        (typeof pricing !== 'object' ||
          pricing === null ||
          !Number.isFinite(pricing.input) ||
          pricing.input < 0)
      ) {
        throw new AiConfigurationError(
          `AiModule: the embedding model "${name}" pricing.input must be ` +
            'a finite nonnegative number.',
        );
      }
      const used = overrides?.[name] ?? model;
      assertModel(used, `AiModule: the replacement embedding model "${name}"`);
      this.models.set(name, {
        info: { name, ...identify(used) },
        model: used,
        ...(pricing !== undefined && { pricing }),
        settings,
      });
    }
    for (const name of Object.keys(overrides ?? {})) {
      if (!this.models.has(name)) {
        throw new AiConfigurationError(
          `overrideEmbeddingModels(): there is no embedding model "${name}" to override. ` +
            this.known(),
        );
      }
    }
  }

  names(): string[] {
    return [...this.models.keys()];
  }

  /** The SDK model, for direct calls without `AiService`'s resilience or usage events. */
  get(name: string = DEFAULT_MODEL_NAME): EmbeddingModel {
    return this.resolve(name).model;
  }

  has(name: string): boolean {
    return this.models.has(name);
  }

  /**
   * @internal What `@InjectEmbeddingModel()` injects; unknown names fail during module
   * initialization.
   */
  injected(name: string, inquirer: unknown): EmbeddingModel {
    const resolved = this.models.get(name);
    if (!resolved) {
      const requester = (inquirer as { constructor?: { name?: string } } | undefined)?.constructor
        ?.name;
      throw new AiConfigurationError(
        `${requester ? `${requester}: ` : ''}@InjectEmbeddingModel('${name}') ` +
          'asks for an embedding model AiModule does not configure. ' +
          `${this.known()} Add "${name}" to \`embeddingModels\` ` +
          '(in forRootAsync(), to what the factory returns).',
      );
    }
    return resolved.model;
  }

  /**
   * @internal Resolves a name or an SDK object. Direct objects have no registered settings or
   * pricing.
   */
  resolve(reference: AiEmbeddingModelReference | undefined): ResolvedEmbeddingModel {
    if (reference === undefined || typeof reference === 'string') {
      const name = reference ?? DEFAULT_MODEL_NAME;
      const resolved = this.models.get(name);
      if (!resolved) {
        throw new AiConfigurationError(
          `There is no embedding model named "${name}". ${this.known()}`,
        );
      }
      return resolved;
    }
    assertModel(reference, 'The embedding `model` option');
    return { info: identify(reference), model: reference, settings: {} };
  }

  private known(): string {
    return `Configured embedding models: ${this.names().join(', ')}.`;
  }
}

function isDefinition(
  entry: EmbeddingModel | AiEmbeddingModelDefinition,
): entry is AiEmbeddingModelDefinition {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    'model' in entry &&
    !('specificationVersion' in entry)
  );
}

function assertModel(model: unknown, where: string): void {
  const valid =
    (typeof model === 'string' && model.length > 0) ||
    (typeof model === 'object' &&
      model !== null &&
      typeof (model as { doEmbed?: unknown }).doEmbed === 'function');
  if (!valid) {
    throw new AiConfigurationError(
      `${where} is not an embedding model. ` +
        'Pass an embedding model from an AI SDK provider package.',
    );
  }
}

function assertSettings(settings: AiEmbeddingSettings | undefined, where: string): void {
  if (settings === undefined) {
    return;
  }
  if (typeof settings !== 'object' || settings === null || Array.isArray(settings)) {
    throw new AiConfigurationError(`${where} must contain embedding settings.`);
  }
  if (
    settings.maxParallelCalls !== undefined &&
    (!Number.isSafeInteger(settings.maxParallelCalls) || settings.maxParallelCalls < 1)
  ) {
    throw new AiConfigurationError(
      `${where} maxParallelCalls must be a finite positive safe integer.`,
    );
  }
}

/** Registered string ids are resolved by the SDK's default gateway provider. */
function identify(model: EmbeddingModel): Omit<AiModelInfo, 'name'> {
  return typeof model === 'string'
    ? { provider: 'gateway', modelId: model }
    : { provider: model.provider, modelId: model.modelId };
}
