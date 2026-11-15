/**
 * `@nestjs/ai/testing`: the fake model on its own and with `ai`, `providerError()`, and `callTool()`, which calls an
 * agent function as a model does.
 */
import { ForbiddenException, Injectable, UseGuards, type CanActivate, type ExecutionContext } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import { APICallError, generateText, streamText } from 'ai';
import * as z from 'zod';
import { AgentFunction, AiExecutionContext, AiModule, AiToolError, type AiContextType } from '../lib/index.js';
import { callTool, FakeModel, fakeModel, providerError } from '../lib/testing/index.js';

@Injectable()
class SupportOnly implements CanActivate {
  canActivate(context: ExecutionContext) {
    const call = AiExecutionContext.create(context).getContext<unknown, { role?: string }>();
    if (context.getType<AiContextType>() !== 'ai' || call.user?.role !== 'support') {
      throw new ForbiddenException('Support only');
    }
    return true;
  }
}

@Injectable()
class TicketTools {
  @AgentFunction({ name: 'close-ticket', description: 'Close a ticket', inputSchema: z.object({ id: z.number() }) })
  @UseGuards(SupportOnly)
  close({ id }: { id: number }) {
    return { closed: id };
  }
}

describe('@nestjs/ai/testing', () => {
  describe('FakeModel', () => {
    it('answers in order with text, tool calls and objects, and records the calls', async () => {
      const model = fakeModel('first', { toolCalls: [{ name: 'lookup', input: { id: 1 }, id: 'call_a' }] }, { object: { ok: true } });

      expect((await generateText({ model, instructions: 'Be brief.', prompt: 'One' })).text).toBe('first');
      const second = await generateText({ model, prompt: 'Two' });
      expect(second.finishReason).toBe('tool-calls');
      expect(second.toolCalls).toEqual([expect.objectContaining({ toolCallId: 'call_a', toolName: 'lookup', input: { id: 1 } })]);
      expect((await generateText({ model, prompt: 'Three' })).text).toBe('{"ok":true}');

      expect(model.calls.map((call) => call.lastUserMessage)).toEqual(['One', 'Two', 'Three']);
      expect(model.calls[0]!.instructions).toBe('Be brief.');
      await expect(generateText({ model, prompt: 'Four', maxRetries: 0 })).rejects.toThrow('no answer left for call 4');
    });

    it('answers every call with a function of the call, and takes more answers with reply()', async () => {
      const echo = new FakeModel((call) => `you said: ${call.lastUserMessage}`);
      expect((await generateText({ model: echo, prompt: 'hi' })).text).toBe('you said: hi');
      expect((await generateText({ model: echo, prompt: 'again' })).text).toBe('you said: again');

      const model = new FakeModel(['a']).reply('b');
      expect((await generateText({ model, prompt: '1' })).text).toBe('a');
      expect((await generateText({ model, prompt: '2' })).text).toBe('b');
    });

    it('streams its text in word-sized chunks, with the usage it was given', async () => {
      const model = fakeModel({ text: 'one two three', usage: { inputTokens: 3, outputTokens: 7 } });
      const result = streamText({ model, prompt: 'Count' });
      const parts: string[] = [];
      for await (const part of result.textStream) {
        parts.push(part);
      }
      expect(parts).toEqual(['one ', 'two ', 'three']);
      expect(await result.usage).toMatchObject({ inputTokens: 3, outputTokens: 7 });
      expect(model.lastCall!.stream).toBe(true);
    });

    it('fails as a provider does, and waits as long as it is told unless the call is aborted', async () => {
      const error = providerError(429, { retryAfter: 2 });
      expect(error).toBeInstanceOf(APICallError);
      expect(error).toMatchObject({ statusCode: 429, isRetryable: true });
      expect(providerError(400).isRetryable).toBe(false);

      const model = fakeModel(error, { text: 'slow', delay: 5_000 });
      await expect(generateText({ model, prompt: 'a', maxRetries: 0 })).rejects.toBe(error);
      const controller = new AbortController();
      const slow = generateText({ model, prompt: 'b', maxRetries: 0, abortSignal: controller.signal });
      controller.abort(new Error('stop'));
      await expect(slow).rejects.toThrow('stop');
    });
  });

  describe('providerError()', () => {
    it('uses retryAfter for the retry-after header', () => {
      expect(providerError(429, { retryAfter: 2 }).responseHeaders).toEqual({ 'retry-after': '2' });
    });

    it('uses custom response headers as provided', () => {
      const responseHeaders = { 'retry-after-ms': '2000', 'x-provider': 'fake' };
      expect(providerError(429, { responseHeaders }).responseHeaders).toEqual(responseHeaders);
    });

    it('merges custom response headers with retryAfter', () => {
      const responseHeaders = { 'retry-after-ms': '2000' };
      expect(
        providerError(429, { retryAfter: 2, responseHeaders }).responseHeaders,
      ).toEqual({ 'retry-after': '2', 'retry-after-ms': '2000' });
    });

    it('lets an explicit retry-after header override retryAfter', () => {
      const responseHeaders = { 'retry-after': '5' };
      expect(
        providerError(429, { retryAfter: 2, responseHeaders }).responseHeaders,
      ).toEqual(responseHeaders);
    });

    it('leaves responseHeaders undefined when neither header option is given', () => {
      expect(providerError(429).responseHeaders).toBeUndefined();
    });
  });

  describe('callTool()', () => {
    it('runs an agent function through its schema and enhancers, as the caller', async () => {
      const moduleRef = await Test.createTestingModule({
        imports: [AiModule.forRoot({ models: { default: new FakeModel([]) } })],
        providers: [TicketTools],
      }).compile();
      moduleRef.useLogger(false);

      expect(await callTool(moduleRef, 'close-ticket', { id: 7 }, { caller: { user: { role: 'support' } } })).toEqual({ closed: 7 });
      const refused = await callTool(moduleRef, 'close-ticket', { id: 7 }, { caller: { user: { role: 'customer' } } }).catch((e: unknown) => e);
      expect(refused).toBeInstanceOf(AiToolError);
      expect(refused).toMatchObject({ message: 'Support only', toolName: 'close-ticket' });
      await expect(callTool(moduleRef, 'close-ticket', { id: 'seven' })).rejects.toThrow('Invalid input: id:');
      await expect(callTool(moduleRef, 'open-ticket')).rejects.toThrow('There is no agent function named "open-ticket"');
      await moduleRef.close();
    });
  });
});
