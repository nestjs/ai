import type { ModelMessage, UIMessageChunk } from 'ai';
import { AiBudgetExceededError } from '../errors/ai.errors.js';
import type { AiStep, AiTextResult } from '../interfaces/ai-result.interface.js';
import type { AiUsage, AiUsageEvent } from '../interfaces/ai-usage.interface.js';
import { addCost, addUsage, ZERO_USAGE } from '../utils/usage.util.js';
import { runStep, type StepAttempt } from './model-policy.js';
import type { CallPlan } from './call-plan.js';
import { answeredOutcome, generateStep, streamStep, wantsAnotherStep, type StepOutcome, type StepOutput } from './steps.js';

export type LoopMode =
  | { stream: false; output?: StepOutput }
  | { stream: true; emit: (chunk: UIMessageChunk) => void };

export interface LoopResult {
  result: AiTextResult;
  /** The last step's parsed answer (structured output). */
  output?: unknown;
  /** The last step's `finish` chunk (streams). */
  finish?: UIMessageChunk;
}

/**
 * The agent loop: a model call, then, while the model asks for tools (which run inside the call), another one with
 * the conversation so far, up to `maxSteps`. Each model call runs under its resilience policies (retry, timeout,
 * fallback); the budget is checked before each. `report` receives a usage event per model call that answered.
 */
export async function runLoop(plan: CallPlan, mode: LoopMode, report: (event: AiUsageEvent) => void): Promise<LoopResult> {
  const messages: ModelMessage[] = [...plan.messages];
  const responseMessages: ModelMessage[] = [];
  const steps: AiStep[] = [];
  let usage: AiUsage = ZERO_USAGE;
  let cost: number | undefined;
  let last: (StepOutcome & { finish?: UIMessageChunk }) | undefined;

  for (let index = 0; index < plan.maxSteps; index++) {
    assertWithinBudget(plan, usage, cost);
    const conversation = [...messages];
    const run = async (attempt: StepAttempt) => {
      let outcome: StepOutcome & { finish?: UIMessageChunk };
      try {
        outcome = mode.stream
          ? await streamStep(plan, index, conversation, attempt)
          : await generateStep(plan, index, conversation, attempt, mode.output);
      } catch (error) {
        // An answer that is not the object still cost its tokens.
        const answered = answeredOutcome(error);
        if (answered) {
          report(usageEvent(plan, attempt, answered, mode.stream));
        }
        throw error;
      }
      report(usageEvent(plan, attempt, outcome, mode.stream));
      return outcome;
    };
    const { value, attempts } = await runStep(plan, run, mode.stream ? (chunk) => mode.emit(chunk as UIMessageChunk) : undefined);

    last = value;
    steps.push({ ...value.step, attempts });
    usage = addUsage(usage, value.step.usage);
    cost = addCost(cost, value.step.cost);
    messages.push(...value.responseMessages);
    responseMessages.push(...value.responseMessages);
    if (!wantsAnotherStep(value)) {
      break;
    }
  }

  const final = last!;
  return {
    result: {
      text: final.step.text,
      finishReason: final.step.finishReason,
      steps,
      usage,
      ...(cost !== undefined && { cost }),
      toolCalls: steps.flatMap((step) => step.toolCalls),
      toolResults: steps.flatMap((step) => step.toolResults),
      responseMessages,
      model: final.step.model,
    },
    ...('output' in final && { output: final.output }),
    ...(final.finish && { finish: final.finish }),
  };
}

/** The usage event of a model call that answered. */
export function usageEvent(plan: CallPlan, attempt: StepAttempt, outcome: StepOutcome, stream: boolean): AiUsageEvent {
  return {
    ...(plan.label !== undefined && { label: plan.label }),
    model: attempt.model.info,
    step: outcome.step.index,
    attempt: attempt.attempt,
    fallback: attempt.fallback,
    stream,
    usage: outcome.step.usage,
    ...(outcome.step.cost !== undefined && { cost: outcome.step.cost }),
    durationMs: Math.round(outcome.durationMs),
    finishReason: outcome.step.finishReason,
  };
}

/** Throws when the call has spent its budget. */
export function assertWithinBudget(plan: Pick<CallPlan, 'budget'>, usage: AiUsage, cost: number | undefined): void {
  const budget = plan.budget;
  if (!budget) {
    return;
  }
  const overTokens = budget.maxTokens !== undefined && usage.totalTokens >= budget.maxTokens;
  const overCost = budget.maxCost !== undefined && (cost ?? 0) >= budget.maxCost;
  if (overTokens || overCost) {
    throw new AiBudgetExceededError(usage, cost, budget);
  }
}
