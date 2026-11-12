import { generateText, NoObjectGeneratedError, Output, streamText, type FinishReason, type ModelMessage, type Schema, type UIMessageChunk } from 'ai';
import { AiOutputError, AiToolError } from '../errors/ai.errors.js';
import type { AiStep, AiToolCall, AiToolResult } from '../interfaces/ai-result.interface.js';
import type { ResolvedModel } from '../models/ai-models.service.js';
import { costOf, toUsage } from '../utils/usage.util.js';
import type { StepAttempt } from './model-policy.js';
import type { CallPlan } from './call-plan.js';

/** One model call's outcome, as the loop reads it. */
export interface StepOutcome {
  step: Omit<AiStep, 'attempts'>;
  /** The assistant (and tool) messages the step added to the conversation. */
  responseMessages: ModelMessage[];
  /** The parsed answer, for structured output on a final step. */
  output?: unknown;
  durationMs: number;
}

/** Structured output: the `ai` schema of the answer, with its name and description. */
export interface StepOutput {
  schema: Schema<unknown>;
  name?: string;
  description?: string;
}

/** Whether the loop goes on after a step: the model asked for tools, and they ran. */
export function wantsAnotherStep(outcome: StepOutcome): boolean {
  return outcome.step.finishReason === 'tool-calls' && outcome.step.toolResults.length > 0;
}

function callSettings(plan: CallPlan, model: ResolvedModel, messages: ModelMessage[], attempt: StepAttempt, output?: StepOutput) {
  return {
    model: model.model,
    ...(plan.instructions !== undefined && { instructions: plan.instructions }),
    messages,
    ...(plan.tools && { tools: plan.tools }),
    ...(plan.toolChoice !== undefined && { toolChoice: plan.toolChoice }),
    ...(output && {
      output: Output.object({
        schema: output.schema,
        ...(output.name !== undefined && { name: output.name }),
        ...(output.description !== undefined && { description: output.description }),
      }),
    }),
    ...plan.settingsFor(model),
    // Retries are the call's resilience policies'.
    maxRetries: 0,
    abortSignal: attempt.signal,
  };
}

/** One model call with `generateText()`: its tool calls run inside it. */
export async function generateStep(
  plan: CallPlan,
  index: number,
  messages: ModelMessage[],
  attempt: StepAttempt,
  output?: StepOutput,
): Promise<StepOutcome> {
  const started = performance.now();
  // A tool that starts commits the attempt (`onExecute`): another attempt would run it again.
  plan.running = attempt;
  let result: Awaited<ReturnType<typeof generateText>>;
  try {
    result = await generateText(callSettings(plan, attempt.model, messages, attempt, output) as Parameters<typeof generateText>[0]);
  } catch (error) {
    // The model answered, and its answer is not the object: no retry or other model is tried.
    if (output && NoObjectGeneratedError.isInstance(error)) {
      const text = error.text ?? '';
      throw answeredWith(toOutputError(error, text), {
        step: describeStep(index, attempt.model, { text, content: [], finishReason: error.finishReason ?? 'other', usage: error.usage }),
        responseMessages: [],
        durationMs: performance.now() - started,
      });
    }
    throw error;
  } finally {
    plan.running = undefined;
  }
  const outcome: StepOutcome = {
    step: describeStep(index, attempt.model, {
      text: result.text,
      content: result.content,
      finishReason: result.finishReason,
      usage: result.usage,
    }),
    responseMessages: [...result.responseMessages] as ModelMessage[],
    durationMs: performance.now() - started,
  };
  if (output && !wantsAnotherStep(outcome)) {
    try {
      outcome.output = result.output;
    } catch (error) {
      throw answeredWith(toOutputError(error, result.text), outcome);
    }
  }
  return outcome;
}

/** The model calls that answered, and whose answer is not the object: their tokens were spent all the same. */
const answered = new WeakMap<AiOutputError, StepOutcome>();

function answeredWith(error: AiOutputError, outcome: StepOutcome): AiOutputError {
  answered.set(error, outcome);
  return error;
}

/** The outcome of the model call that answered with what `error` (an `AiOutputError`) rejects. */
export function answeredOutcome(error: unknown): StepOutcome | undefined {
  return error instanceof AiOutputError ? answered.get(error) : undefined;
}

/** An answer that is not JSON, or does not match the schema. */
function toOutputError(error: unknown, text: string): AiOutputError {
  const cause = (error as { cause?: unknown })?.cause ?? error;
  return new AiOutputError(
    `The model's answer does not match the schema: ${(cause as Error)?.message ?? String(cause)}`,
    text,
    { cause: error },
  );
}

/**
 * One model call with `streamText()`, its UI message chunks sent with `attempt.emit()` as they come. Until the first
 * chunk with content arrives the attempt can still fail and be retried: the chunks before it (`start`, `start-step`)
 * are held. The first one sent commits the attempt, and a failure after it is final. The step's `finish` chunk is
 * held for the loop, which writes one when the last step ends.
 */
export async function streamStep(
  plan: CallPlan,
  index: number,
  messages: ModelMessage[],
  attempt: StepAttempt,
): Promise<StepOutcome & { finish?: UIMessageChunk }> {
  const emit = (chunk: UIMessageChunk) => attempt.emit(chunk);
  plan.running = attempt;
  try {
    return await streamAttempt(plan, index, messages, attempt, emit);
  } finally {
    plan.running = undefined;
  }
}

async function streamAttempt(
  plan: CallPlan,
  index: number,
  messages: ModelMessage[],
  attempt: StepAttempt,
  emit: (chunk: UIMessageChunk) => void,
): Promise<StepOutcome & { finish?: UIMessageChunk }> {
  const started = performance.now();
  let failure: unknown;
  const result = streamText({
    ...callSettings(plan, attempt.model, messages, attempt),
    // Errors are read from the chunks; `ai` would log them to the console.
    onError: () => {},
  } as Parameters<typeof streamText>[0]);
  const chunks = result.toUIMessageStream({
    sendStart: index === 0,
    sendFinish: true,
    onError: (error: unknown) => {
      // A tool's error is what the model was told; anything else is the stream's failure.
      if (error instanceof AiToolError) {
        return error.message;
      }
      failure = error;
      return 'An error occurred.';
    },
  });

  const held: UIMessageChunk[] = [];
  let opened = false;
  let finish: UIMessageChunk | undefined;
  const reader = (chunks as ReadableStream<UIMessageChunk>).getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value.type === 'error') {
        // `failure` was set by `onError` for this chunk.
        throw failure ?? new Error(value.errorText);
      }
      if (value.type === 'abort') {
        attempt.signal.throwIfAborted();
        continue;
      }
      if (value.type === 'finish') {
        finish = value;
        continue;
      }
      if (!opened && (value.type === 'start' || value.type === 'start-step')) {
        held.push(value);
        continue;
      }
      if (!opened) {
        opened = true;
        held.splice(0).forEach(emit);
      }
      emit(value);
    }
  } finally {
    reader.releaseLock();
  }
  attempt.signal.throwIfAborted();
  if (!opened) {
    // An empty answer (no content at all): what it held still opens the message.
    held.splice(0).forEach(emit);
  }

  const [text, content, finishReason, usage, responseMessages] = await Promise.all([
    result.text,
    result.content,
    result.finishReason,
    result.usage,
    result.responseMessages,
  ]);
  return {
    step: describeStep(index, attempt.model, { text, content, finishReason, usage }),
    responseMessages: [...responseMessages] as ModelMessage[],
    durationMs: performance.now() - started,
    ...(finish && { finish }),
  };
}

type ContentPart = { type: string; toolCallId?: string; toolName?: string; input?: unknown; output?: unknown; error?: unknown };

function describeStep(
  index: number,
  model: ResolvedModel,
  result: { text: string; content: ReadonlyArray<unknown>; finishReason: FinishReason; usage: Parameters<typeof toUsage>[0] },
): Omit<AiStep, 'attempts'> {
  const toolCalls: AiToolCall[] = [];
  const toolResults: AiToolResult[] = [];
  for (const part of result.content as ContentPart[]) {
    const call = { toolCallId: part.toolCallId ?? '', toolName: part.toolName ?? '', input: part.input };
    if (part.type === 'tool-call') {
      toolCalls.push(call);
    } else if (part.type === 'tool-result') {
      toolResults.push({ ...call, output: part.output });
    } else if (part.type === 'tool-error') {
      toolResults.push({ ...call, error: String(part.error) });
    }
  }
  const usage = toUsage(result.usage);
  const cost = costOf(usage, model.pricing);
  return {
    index,
    model: model.info,
    text: result.text,
    toolCalls,
    toolResults,
    finishReason: result.finishReason,
    usage,
    ...(cost !== undefined && { cost }),
  };
}
