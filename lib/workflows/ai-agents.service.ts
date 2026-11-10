import { Injectable } from '@nestjs/common';
import { ModuleRef } from '@nestjs/core';
import { NonRetryableStepError, WorkflowClient, WorkflowSignal, type WorkflowContext } from '@nestjs/workflows';
import type { ModelMessage } from 'ai';
import { AI_MODULE_OPTIONS } from '../ai.constants.js';
import { AiConfigurationError, AiError, AiToolError } from '../errors/ai.errors.js';
import type { AiCallOptions } from '../interfaces/ai-call-options.interface.js';
import type { AiCallDefaults, AiModuleOptions } from '../interfaces/ai-module-options.interface.js';
import type { AiStep, AiTextResult, AiToolResult } from '../interfaces/ai-result.interface.js';
import type { AiUsage } from '../interfaces/ai-usage.interface.js';
import { assertWithinBudget } from '../runtime/agent-loop.js';
import type { StepOutcome } from '../runtime/steps.js';
import { AiService } from '../services/ai.service.js';
import { AiTools } from '../tools/ai-tools.service.js';
import { INTERNAL_ERROR_MESSAGE } from '../utils/errors.util.js';
import { addCost, addUsage, validateBudget, ZERO_USAGE } from '../utils/usage.util.js';
import type {
  AiAgentRunOptions,
  AiAgentRunResult,
  AiPendingApproval,
  AiToolApproval,
} from './ai-agent.interfaces.js';

/** The signal a tool call that waits for approval takes, keyed by the call (`AiPendingApproval.key`). */
export const TOOL_APPROVAL = new WorkflowSignal<AiToolApproval>('nestjs-ai.tool-approval');

const DEFAULT_MAX_STEPS = 25;
const DEFAULT_STEPS_PER_INSTANCE = 10;

/** A run's options, resolved once and stored with it: tools as names. Internal. */
export interface AgentSettings extends Omit<AiAgentRunOptions, 'tools' | 'prompt' | 'messages'> {
  tools: string[];
  approve: string[];
  maxSteps: number;
  stepsPerInstance: number;
}

/** Where a run is: what each instance replays to, and what a child instance starts from. Internal. */
export interface AgentState {
  /** The instance `run()` was called in: approval keys name it. */
  runId: string;
  index: number;
  messages: ModelMessage[];
  responseMessages: ModelMessage[];
  steps: AiStep[];
  usage: AiUsage;
  cost?: number;
}

/** What the pending approvals of an instance are, in its custom status. */
interface AgentStatus {
  pendingApprovals?: AiPendingApproval[];
}

/**
 * The agent loop as a durable workflow: inside a `@Workflow()`'s `run()`, each model call is a step and each tool
 * call (an agent function's) another, so that a restart resumes where the run was, without calling a model or an agent
 * function again for what is journaled. Calls of the functions named in `approve` wait for a person
 * (`waitForSignal()`), for days if need be. After `stepsPerInstance` model calls the run goes on in a child workflow.
 *
 * ```ts
 * @Workflow('refund-request')
 * export class RefundRequest {
 *   constructor(private readonly agents: AiAgents) {}
 *
 *   run(ctx: WorkflowContext, { email, customerId }: RefundInput) {
 *     return this.agents.run(ctx, {
 *       prompt: email,
 *       tools: ['find-order', 'refund'],
 *       approve: ['refund'],
 *       caller: { user: { id: customerId } },
 *     });
 *   }
 * }
 * ```
 */
@Injectable()
export class AiAgents {
  constructor(
    private readonly moduleRef: ModuleRef,
    private readonly client: WorkflowClient,
  ) {}

  /** Runs the agent loop in the workflow `ctx` is the context of; resolves with its result. */
  run(ctx: WorkflowContext, options: AiAgentRunOptions): Promise<AiAgentRunResult> {
    const settings = this.settings(options);
    const messages = initialMessages(options);
    return this.resume(ctx, settings, {
      runId: ctx.workflowId,
      index: 0,
      messages,
      responseMessages: [],
      steps: [],
      usage: ZERO_USAGE,
    });
  }

  /** Approves a tool call waiting for it; `key` is `AiPendingApproval.key`. The first answer for a call counts. */
  async approve(key: string, { by }: { by?: unknown } = {}): Promise<void> {
    await this.answer(key, { approved: true, ...(by !== undefined && { by }) });
  }

  /** Rejects a tool call waiting for approval: the agent function does not run, and the model is told `reason`. */
  async reject(key: string, { reason, by }: { reason?: string; by?: unknown } = {}): Promise<void> {
    await this.answer(key, { approved: false, ...(reason !== undefined && { reason }), ...(by !== undefined && { by }) });
  }

  /** The tool calls of run `runId` that wait for approval now, in it or in the child instances it goes on in. */
  async pendingApprovals(runId: string): Promise<AiPendingApproval[]> {
    const pending: AiPendingApproval[] = [];
    const visit = async (id: string) => {
      const details = await this.client.getStatus(id, { children: true });
      if (!details) {
        return;
      }
      pending.push(...((details.customStatus as AgentStatus | null)?.pendingApprovals ?? []));
      for (const child of details.children ?? []) {
        if (child.status !== 'completed' && child.status !== 'failed' && child.status !== 'cancelled') {
          await visit(child.id);
        }
      }
    };
    await visit(runId);
    return pending;
  }

  /** @internal Runs from `state` on: `run()`'s instance, and each child instance a run goes on in. */
  async resume(ctx: WorkflowContext, settings: AgentSettings, state: AgentState): Promise<AiAgentRunResult> {
    const last = state.index + settings.stepsPerInstance;
    while (state.index < settings.maxSteps) {
      if (state.index >= last) {
        const result = await ctx.executeChild(AGENT_WORKFLOW, { settings, state }, { id: `${state.runId}:from-${state.index}` });
        return result as AiAgentRunResult;
      }
      assertWithinBudget(settings, state.usage, state.cost);

      const index = state.index;
      const { outcome, attempts } = await ctx.step(`model-${index}`, (step) =>
        this.ai()
          .callModel(callOptions(settings), { index, messages: state.messages, signal: step.signal })
          .catch((error: unknown) => {
            // A mistake in the options fails the same way every time: not worth a durable retry.
            throw error instanceof AiConfigurationError ? new NonRetryableStepError(error.message) : error;
          }),
      );
      const step = outcome.step as StepOutcome['step'];
      const results = await this.runTools(ctx, settings, state, index, step);

      state.messages.push(...(outcome.responseMessages as ModelMessage[]));
      state.responseMessages.push(...(outcome.responseMessages as ModelMessage[]));
      if (results.length > 0) {
        const message = toolMessage(results);
        state.messages.push(message);
        state.responseMessages.push(message);
      }
      state.steps.push({ ...step, toolResults: [...step.toolResults, ...results], attempts });
      state.usage = addUsage(state.usage, step.usage);
      state.cost = addCost(state.cost, step.cost);
      state.index++;
      if (step.finishReason !== 'tool-calls' || step.toolCalls.length === 0) {
        break;
      }
    }
    return resultOf(state);
  }

  /**
   * The step's valid tool calls (`ai` answered the invalid ones), one durable step each, after their approval when
   * they need one. An agent function's failure is a result the model reads, not the step's failure: the step is
   * retried only after a crash, with the same `idempotencyKey` in the function's context.
   */
  private async runTools(
    ctx: WorkflowContext,
    settings: AgentSettings,
    state: AgentState,
    index: number,
    step: StepOutcome['step'],
  ): Promise<AiToolResult[]> {
    // A call's id names its approval, its wait and its step: two calls with one id can't be told apart.
    const ids = new Set<string>();
    for (const call of step.toolCalls) {
      if (ids.has(call.toolCallId)) {
        throw new AiError(
          `The model's response at step ${index} has more than one tool call with the id "${call.toolCallId}", ` +
            'so the run stops before any of them runs.',
        );
      }
      ids.add(call.toolCallId);
    }

    const answered = new Set(step.toolResults.map((result) => result.toolCallId));
    const calls = step.toolCalls.filter((call) => !answered.has(call.toolCallId));

    // Every call of the step that needs approval is listed at once; they are answered in any order.
    const waiting = calls
      .filter((call) => settings.approve.includes(call.toolName))
      .map((call) => ({
        key: approvalKey(state.runId, index, call.toolCallId),
        runId: state.runId,
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        input: call.input,
      }));
    if (waiting.length > 0) {
      ctx.setStatus({ pendingApprovals: waiting } satisfies AgentStatus);
    }

    const results: AiToolResult[] = [];
    for (const call of calls) {
      const approval = waiting.find((pending) => pending.toolCallId === call.toolCallId);
      if (approval) {
        const decision = await ctx.waitForSignal(`approval-${index}-${call.toolCallId}`, TOOL_APPROVAL, {
          key: approval.key,
          ...(settings.approvalTimeout !== undefined && { timeout: settings.approvalTimeout }),
        });
        waiting.splice(waiting.indexOf(approval), 1);
        ctx.setStatus(waiting.length > 0 ? ({ pendingApprovals: waiting } satisfies AgentStatus) : null);
        if (!decision?.approved) {
          const error = decision ? `The call was rejected${decision.reason ? `: ${decision.reason}` : ''}` : 'No one approved the call in time';
          results.push({ ...call, error });
          continue;
        }
      }
      results.push(
        (await ctx.step(`tool-${index}-${call.toolCallId}`, async (toolStep) => {
          const tools = this.tools();
          const entry = tools.tools.get(call.toolName);
          if (!entry) {
            return { ...call, error: `There is no tool named "${call.toolName}"` };
          }
          try {
            const output = await tools.call(entry, call.input, settings.caller ?? {}, {
              toolCallId: call.toolCallId,
              messages: state.messages,
              abortSignal: toolStep.signal,
              idempotencyKey: toolStep.idempotencyKey,
            });
            return { ...call, output };
          } catch (error) {
            return { ...call, error: error instanceof AiToolError ? error.message : INTERNAL_ERROR_MESSAGE };
          }
        })) as AiToolResult,
      );
    }
    return results;
  }

  private settings(options: AiAgentRunOptions): AgentSettings {
    if (typeof options !== 'object' || options === null) {
      throw new AiConfigurationError('AiAgents.run() takes options: { prompt } or { messages }, and the rest.');
    }
    const { tools, prompt: _prompt, messages: _messages, ...rest } = options;
    const names = tools?.length ? this.tools().resolve(tools).map((entry) => entry.name) : [];
    const approve = options.approve ?? [];
    for (const name of approve) {
      if (!names.includes(name)) {
        throw new AiConfigurationError(`\`approve\` names "${name}", which is not one of the run's \`tools\`.`);
      }
    }
    const defaults = this.defaults();
    const maxSteps = options.maxSteps ?? defaults.maxSteps ?? DEFAULT_MAX_STEPS;
    const stepsPerInstance = options.stepsPerInstance ?? DEFAULT_STEPS_PER_INSTANCE;
    for (const [name, value] of Object.entries({ maxSteps, stepsPerInstance })) {
      if (!Number.isInteger(value) || value < 1) {
        throw new AiConfigurationError(
          `\`${name}\` must be a whole number of at least 1. Got ${String(value)}.`,
        );
      }
    }
    const budget = options.budget ?? defaults.budget;
    validateBudget(budget);
    return {
      ...rest,
      tools: names,
      approve,
      maxSteps,
      stepsPerInstance,
      ...(budget && { budget }),
    };
  }

  /** The module's `defaults`, which a run's `maxSteps` and `budget` fall back to, as a call's do. */
  private defaults(): AiCallDefaults {
    return this.moduleRef.get<AiModuleOptions>(AI_MODULE_OPTIONS, { strict: false }).defaults ?? {};
  }

  private answer(key: string, decision: AiToolApproval) {
    return this.client.signal(TOOL_APPROVAL, decision, { key, id: `${key}:answer` });
  }

  private ai(): AiService {
    return this.moduleRef.get(AiService, { strict: false });
  }

  private tools(): AiTools {
    return this.moduleRef.get(AiTools, { strict: false });
  }
}

/** The workflow a run goes on in after `stepsPerInstance` model calls (`AiAgentSegment`). */
export const AGENT_WORKFLOW = 'nestjs-ai.agent';

// The step's index keeps the key unique when a provider reuses tool call ids across responses (TGI gives each one "0").
export function approvalKey(runId: string, index: number, toolCallId: string): string {
  return `${runId}:${index}:${toolCallId}`;
}

function initialMessages(options: AiAgentRunOptions): ModelMessage[] {
  if (options.prompt !== undefined && options.messages !== undefined) {
    throw new AiConfigurationError('Pass `prompt` or `messages`, not both.');
  }
  if (typeof options.prompt === 'string') {
    return [{ role: 'user', content: options.prompt }];
  }
  if (Array.isArray(options.messages) && options.messages.length > 0) {
    return [...options.messages];
  }
  throw new AiConfigurationError('Pass `prompt` (a string) or `messages` (a non-empty array of model messages).');
}

function callOptions(settings: AgentSettings): AiCallOptions {
  const { approve: _approve, approvalTimeout: _timeout, stepsPerInstance: _steps, maxSteps: _max, budget: _budget, caller, tools, ...rest } = settings;
  return { ...rest, tools, caller: { ...(caller?.user !== undefined && { user: caller.user }), ...(caller?.scopes && { scopes: caller.scopes }) } };
}

/** The results of tool calls as the tool message the model reads next. */
function toolMessage(results: AiToolResult[]): ModelMessage {
  return {
    role: 'tool',
    content: results.map((result) => ({
      type: 'tool-result' as const,
      toolCallId: result.toolCallId,
      toolName: result.toolName,
      output:
        result.error !== undefined
          ? { type: 'error-text' as const, value: result.error }
          : typeof result.output === 'string'
            ? { type: 'text' as const, value: result.output }
            : { type: 'json' as const, value: (result.output ?? null) as never },
    })),
  };
}

function resultOf(state: AgentState): AiTextResult {
  const final = state.steps[state.steps.length - 1]!;
  return {
    text: final.text,
    finishReason: final.finishReason,
    steps: state.steps,
    usage: state.usage,
    ...(state.cost !== undefined && { cost: state.cost }),
    toolCalls: state.steps.flatMap((step) => step.toolCalls),
    toolResults: state.steps.flatMap((step) => step.toolResults),
    responseMessages: state.responseMessages,
    model: final.model,
  };
}
