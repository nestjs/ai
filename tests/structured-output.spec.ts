/**
 * Structured output: `generateObject()` with a schema, what the model is sent, validation and transforms, failures,
 * and an object answered after tool calls.
 */
import { Injectable } from '@nestjs/common';
import type { TestingModule } from '@nestjs/testing';
import * as z from 'zod';
import { AgentFunction, AiConfigurationError, AiOutputError, type AiUsageEvent } from '../lib/index.js';
import { FakeModel } from '../lib/testing/index.js';
import { createAi } from './support/module.js';

const Invoice = z.object({
  number: z.string().describe('The invoice number'),
  total: z.number().describe('The total, in cents'),
  dueDate: z.iso.date().transform((value) => new Date(`${value}T00:00:00Z`)),
  lines: z.array(z.object({ description: z.string(), amount: z.number() })),
});

@Injectable()
class CustomerTools {
  @AgentFunction({ description: 'Look a customer up by email', inputSchema: z.object({ email: z.string() }) })
  customer({ email }: { email: string }) {
    return { email, id: 'cus_42' };
  }
}

describe('structured output', () => {
  let moduleRef: TestingModule | undefined;

  afterEach(async () => {
    await moduleRef?.close();
    moduleRef = undefined;
  });

  it('answers a typed object, validated and transformed by the schema', async () => {
    const model = new FakeModel([
      { object: { number: 'INV-7', total: 4_200, dueDate: '2026-12-01', lines: [{ description: 'Salmon kibble', amount: 4_200 }] } },
    ]);
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } });
    moduleRef = ref;

    const { object, usage } = await ai.generateObject({
      schema: Invoice,
      schemaName: 'invoice',
      instructions: 'Extract the invoice from the email.',
      prompt: 'Invoice INV-7: 42.00 for salmon kibble, due December 1st.',
    });

    expect(object.dueDate).toEqual(new Date('2026-12-01T00:00:00Z'));
    expect(object).toMatchObject({ number: 'INV-7', total: 4_200, lines: [{ description: 'Salmon kibble', amount: 4_200 }] });
    expect(usage.totalTokens).toBe(20);
    // The model was asked for JSON of the schema's input side.
    expect(model.lastCall!.options.responseFormat).toMatchObject({
      type: 'json',
      name: 'invoice',
      schema: {
        type: 'object',
        properties: { number: { type: 'string', description: 'The invoice number' }, dueDate: { type: 'string', format: 'date' } },
        required: ['number', 'total', 'dueDate', 'lines'],
      },
    });
  });

  it("rejects an answer that is not JSON, or does not match, with the model's text", async () => {
    const { moduleRef: ref, ai } = await createAi({ models: { default: new FakeModel(['Sure! Here it is.', { object: { number: 7 } }]) } });
    moduleRef = ref;

    const notJson = (await ai.generateObject({ schema: Invoice, prompt: 'Extract' }).catch((e: unknown) => e)) as AiOutputError;
    expect(notJson).toBeInstanceOf(AiOutputError);
    expect(notJson.text).toBe('Sure! Here it is.');

    const mismatch = (await ai.generateObject({ schema: Invoice, prompt: 'Extract' }).catch((e: unknown) => e)) as AiOutputError;
    expect(mismatch).toBeInstanceOf(AiOutputError);
    expect(mismatch.message).toContain("The model's answer does not match the schema");
    expect(mismatch.message).toContain('number: Invalid input: expected string');
    expect(mismatch.text).toBe('{"number":7}');
  });

  it('reports the usage of an answer that is not the object', async () => {
    const events: AiUsageEvent[] = [];
    const model = new FakeModel([
      { text: 'Sure! Here it is.', usage: { inputTokens: 5, outputTokens: 1 } },
      { object: { number: 7 }, usage: { inputTokens: 5, outputTokens: 2 } },
      { text: '', finishReason: 'length', usage: { inputTokens: 5, outputTokens: 3 } },
    ]);
    const { moduleRef: ref, ai } = await createAi({
      models: { default: { model, pricing: { input: 1_000_000, output: 1_000_000 } } },
      onUsage: (event) => void events.push(event),
    });
    moduleRef = ref;

    for (let i = 0; i < 3; i++) {
      await expect(ai.generateObject({ schema: Invoice, prompt: 'Extract' })).rejects.toBeInstanceOf(AiOutputError);
    }
    expect(model.calls).toHaveLength(3);
    expect(events.map(({ usage, cost, finishReason }) => ({ tokens: usage.totalTokens, cost, finishReason }))).toEqual([
      { tokens: 6, cost: 6, finishReason: 'stop' },
      { tokens: 7, cost: 7, finishReason: 'stop' },
      { tokens: 8, cost: 8, finishReason: 'length' },
    ]);
  });

  it('answers the object after the tool calls it needed', async () => {
    const model = new FakeModel([
      { toolCalls: [{ name: 'customer', input: { email: 'tom@example.com' } }] },
      (call) => ({ object: { category: 'billing', customerId: (call.toolResults[0]!.output as { id: string }).id } }),
    ]);
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } }, { providers: [CustomerTools] });
    moduleRef = ref;

    const { object, steps } = await ai.generateObject({
      schema: z.object({ category: z.enum(['billing', 'shipping', 'other']), customerId: z.string() }),
      prompt: 'From tom@example.com: I was charged twice.',
      tools: ['customer'],
    });
    expect(object).toEqual({ category: 'billing', customerId: 'cus_42' });
    expect(steps.map((step) => step.finishReason)).toEqual(['tool-calls', 'stop']);
  });

  it('rejects when maxSteps runs out before the answer, and refuses a schema it cannot send', async () => {
    const model = new FakeModel(() => ({ toolCalls: [{ name: 'customer', input: { email: 'a@b.c' } }] }));
    const { moduleRef: ref, ai } = await createAi({ models: { default: model } }, { providers: [CustomerTools] });
    moduleRef = ref;

    await expect(ai.generateObject({ schema: Invoice, prompt: 'Extract', tools: ['customer'], maxSteps: 2 })).rejects.toThrow(
      'still calling tools after 2 step(s)',
    );
    await expect(ai.generateObject({ schema: { parse() {} } as never, prompt: 'Extract' })).rejects.toThrow(AiConfigurationError);
  });
});
