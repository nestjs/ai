/**
 * Embedding registration and injection share Nest's module conventions without sharing model
 * namespaces.
 */
import { Injectable, Module } from '@nestjs/common';
import { Test, type TestingModule } from '@nestjs/testing';
import type { EmbeddingModel, LanguageModel } from 'ai';
import {
  AiConfigurationError,
  AiEmbeddingModels,
  AiModels,
  AiModule,
  AiService,
  getEmbeddingModelToken,
  getModelToken,
  InjectEmbeddingModel,
  InjectModel,
} from '../lib/index.js';
import {
  FakeEmbeddingModel,
  FakeModel,
  overrideEmbeddingModels,
  overrideModels,
} from '../lib/testing/index.js';

@Injectable()
class UsesEmbeddings {
  constructor(
    @InjectEmbeddingModel() readonly main: EmbeddingModel,
    @InjectEmbeddingModel('documents') readonly documents: EmbeddingModel,
    readonly ai: AiService,
  ) {}
}

describe('embedding model registration', () => {
  const modules: TestingModule[] = [];
  const compile = async (builder: ReturnType<typeof Test.createTestingModule>) => {
    const ref = await builder.compile();
    modules.push(ref);
    return ref;
  };

  afterEach(async () => {
    await Promise.all(modules.splice(0).map((ref) => ref.close()));
  });

  it('starts embedding-only modules without provider calls and injects clients', async () => {
    const main = new FakeEmbeddingModel([{ embeddings: [[1, 2]] }]);
    const documents = new FakeEmbeddingModel([]);
    const ref = await compile(
      Test.createTestingModule({
        imports: [AiModule.forRoot({ embeddingModels: { default: main, documents } })],
        providers: [UsesEmbeddings],
      }),
    );
    await ref.init();
    expect(main.calls).toEqual([]);
    expect(documents.calls).toEqual([]);
    expect(ref.get(UsesEmbeddings).main).toBe(main);
    expect(ref.get(UsesEmbeddings).documents).toBe(documents);
    const models = ref.get(AiEmbeddingModels);
    expect(models.names()).toEqual(['default', 'documents']);
    expect(models.get()).toBe(main);
    expect(models.has('documents')).toBe(true);
    expect(models.has('missing')).toBe(false);
    expect(ref.get(AiModels).names()).toEqual([]);
    expect((await ref.get(AiService).embed({ value: 'hello' })).embedding).toEqual([1, 2]);
    await expect(ref.get(AiService).generateText({ prompt: 'hello' })).rejects.toThrow(
      AiConfigurationError,
    );
  });

  it('keeps same-named language and embedding models independent', async () => {
    @Injectable()
    class UsesBoth {
      constructor(
        @InjectModel() readonly text: LanguageModel,
        @InjectEmbeddingModel() readonly vectors: EmbeddingModel,
      ) {}
    }
    const text = new FakeModel(['hello']);
    const vectors = new FakeEmbeddingModel([{ embeddings: [[3]] }]);
    const ref = await compile(
      Test.createTestingModule({
        imports: [
          AiModule.forRoot({
            models: { default: text },
            embeddingModels: { default: vectors },
          }),
        ],
        providers: [UsesBoth],
      }),
    );
    expect(getModelToken()).not.toBe(getEmbeddingModelToken());
    expect(ref.get(UsesBoth)).toMatchObject({ text, vectors });
    expect((await ref.get(AiService).generateText({ prompt: 'hi' })).text).toBe('hello');
    expect((await ref.get(AiService).embed({ value: 'hi' })).embedding).toEqual([3]);
  });

  it('supports injected async configuration and tokens for classes loaded later', async () => {
    @Injectable()
    class Configuration {
      readonly embeddings = new FakeEmbeddingModel([]);
    }
    @Module({ providers: [Configuration], exports: [Configuration] })
    class ConfigurationModule {}
    const configured = AiModule.forRootAsync({
      imports: [ConfigurationModule],
      inject: [Configuration],
      embeddingModelNames: ['late'],
      useFactory: async (configuration: Configuration) => ({
        embeddingModels: { late: configuration.embeddings },
      }),
    });
    @Injectable()
    class LateConsumer {
      constructor(@InjectEmbeddingModel('late') readonly model: EmbeddingModel) {}
    }
    const ref = await compile(
      Test.createTestingModule({
        imports: [configured],
        providers: [LateConsumer],
      }),
    );
    expect(ref.get(LateConsumer).model).toBe(ref.get(Configuration).embeddings);
    expect(await ref.resolve(getEmbeddingModelToken('late'))).toBe(
      ref.get(Configuration).embeddings,
    );
  });

  it('supports class and existing async option factories', async () => {
    @Injectable()
    class Configuration {
      readonly model = new FakeEmbeddingModel([]);
      createAiOptions() {
        return { embeddingModels: { default: this.model } };
      }
    }
    const byClass = await compile(
      Test.createTestingModule({
        imports: [AiModule.forRootAsync({ useClass: Configuration })],
      }),
    );
    expect(byClass.get(AiEmbeddingModels).get()).toBe(byClass.get(Configuration).model);
    @Module({ providers: [Configuration], exports: [Configuration] })
    class ConfigurationModule {}
    const existing = await compile(
      Test.createTestingModule({
        imports: [
          AiModule.forRootAsync({
            imports: [ConfigurationModule],
            useExisting: Configuration,
          }),
        ],
      }),
    );
    expect(existing.get(AiEmbeddingModels).get()).toBe(existing.get(Configuration).model);
  });

  it('respects module visibility, explicit global configuration, and re-exports', async () => {
    @Injectable()
    class Consumer {
      constructor(@InjectEmbeddingModel() readonly model: EmbeddingModel) {}
    }
    @Module({ providers: [Consumer] })
    class FeatureModule {}
    const model = new FakeEmbeddingModel([]);
    await expect(
      Test.createTestingModule({
        imports: [AiModule.forRoot({ embeddingModels: { default: model } }), FeatureModule],
      }).compile(),
    ).rejects.toThrow(/Consumer/);
    const global = await compile(
      Test.createTestingModule({
        imports: [
          AiModule.forRoot({
            embeddingModels: { default: model },
            isGlobal: true,
          }),
          FeatureModule,
        ],
      }),
    );
    expect(global.get(Consumer).model).toBe(model);
    @Module({
      imports: [AiModule.forRoot({ embeddingModels: { default: model } })],
      exports: [AiModule],
    })
    class EmbeddingsModule {}
    @Module({ imports: [EmbeddingsModule], providers: [Consumer] })
    class ReexportingModule {}
    const reexported = await compile(Test.createTestingModule({ imports: [ReexportingModule] }));
    expect(reexported.get(Consumer).model).toBe(model);
  });

  it('rejects empty configuration, wrong model kinds, and missing injection names', async () => {
    for (const options of [{}, { models: {}, embeddingModels: {} }]) {
      await expect(
        Test.createTestingModule({
          imports: [AiModule.forRoot(options)],
        }).compile(),
      ).rejects.toThrow(AiConfigurationError);
    }
    await expect(
      Test.createTestingModule({
        imports: [
          AiModule.forRoot({
            embeddingModels: {
              default: new FakeModel([]) as unknown as EmbeddingModel,
            },
          }),
        ],
      }).compile(),
    ).rejects.toThrow(/embedding model/i);
    @Injectable()
    class TypoConsumer {
      constructor(@InjectEmbeddingModel('documants') readonly model: EmbeddingModel) {}
    }
    await expect(
      Test.createTestingModule({
        imports: [
          AiModule.forRoot({
            embeddingModels: { documents: new FakeEmbeddingModel([]) },
          }),
        ],
        providers: [TypoConsumer],
      }).compile(),
    ).rejects.toThrow(/TypoConsumer.*InjectEmbeddingModel.*documants/);
  });

  it('does not implicitly use the first registered embedding model', async () => {
    const model = new FakeEmbeddingModel([]);
    const ref = await compile(
      Test.createTestingModule({
        imports: [AiModule.forRoot({ embeddingModels: { documents: model } })],
      }),
    );
    await expect(ref.get(AiService).embed({ value: 'hi' })).rejects.toThrow(AiConfigurationError);
    await expect(ref.get(AiService).embed({ model: 'missing', value: 'hi' })).rejects.toThrow(
      AiConfigurationError,
    );
    expect(model.calls).toEqual([]);
  });

  it('validates embedding model settings and pricing during startup', async () => {
    const model = new FakeEmbeddingModel([]);
    for (const maxParallelCalls of [0, -1, 1.5, Infinity, NaN]) {
      await expect(
        Test.createTestingModule({
          imports: [
            AiModule.forRoot({
              embeddingModels: { default: model },
              embeddingDefaults: { maxParallelCalls },
            }),
          ],
        }).compile(),
      ).rejects.toThrow(AiConfigurationError);
      await expect(
        Test.createTestingModule({
          imports: [
            AiModule.forRoot({
              embeddingModels: { default: { model, maxParallelCalls } },
            }),
          ],
        }).compile(),
      ).rejects.toThrow(AiConfigurationError);
    }
    for (const input of [-1, Infinity, -Infinity, NaN, '1', null, undefined]) {
      await expect(
        Test.createTestingModule({
          imports: [
            AiModule.forRoot({
              embeddingModels: { default: { model, pricing: { input } as any } },
            }),
          ],
        }).compile(),
      ).rejects.toMatchObject({
        name: 'AiConfigurationError',
        message:
          'AiModule: the embedding model "default" pricing.input must be a finite nonnegative number.',
      });
    }
    for (const pricing of [null, [], Object.assign([], { input: 1 }), '1', 1]) {
      await expect(
        Test.createTestingModule({
          imports: [
            AiModule.forRoot({
              embeddingModels: { default: { model, pricing: pricing as any } },
            }),
          ],
        }).compile(),
      ).rejects.toMatchObject({
        name: 'AiConfigurationError',
        message: 'AiModule: the embedding model "default" pricing must contain an input price.',
      });
    }
    expect(model.calls).toEqual([]);
  });

  it('keeps embedding and language overrides independent, preserving pricing', async () => {
    const configured = new FakeEmbeddingModel([]);
    const replacement = new FakeEmbeddingModel([{ embeddings: [[5]], usage: { tokens: 7 } }]);
    const language = new FakeModel(['language replacement']);
    const ref = await compile(
      overrideEmbeddingModels(
        overrideModels(
          Test.createTestingModule({
            imports: [
              AiModule.forRoot({
                models: { default: new FakeModel([]) },
                embeddingModels: {
                  default: { model: configured, pricing: { input: 1_000_000 } },
                },
              }),
            ],
          }),
          { default: language },
        ),
        { default: replacement },
      ),
    );
    const result = await ref.get(AiService).embed({ value: 'hello' });
    expect(result.embedding).toEqual([5]);
    expect(result.cost).toBe(7);
    expect(configured.calls).toEqual([]);
    expect((await ref.get(AiService).generateText({ prompt: 'hello' })).text).toBe(
      'language replacement',
    );
    await expect(
      overrideEmbeddingModels(
        Test.createTestingModule({
          imports: [AiModule.forRoot({ embeddingModels: { default: configured } })],
        }),
        { unknown: replacement },
      ).compile(),
    ).rejects.toThrow(/overrideEmbeddingModels.*unknown/);
  });
});
