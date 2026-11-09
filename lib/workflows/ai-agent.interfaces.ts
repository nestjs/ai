import type { Duration } from '@nestjs/workflows';
import type { ModelMessage } from 'ai';
import type { AiToolChoice } from '../interfaces/ai-call-options.interface.js';
import type { AiBudget } from '../interfaces/ai-module-options.interface.js';
import type { AiTextResult } from '../interfaces/ai-result.interface.js';
import type { AiToolSelection } from '../interfaces/ai-tool.interface.js';

/** Who a durable run acts for: data, stored with the run, since a request does not survive a restart. */
export interface AiAgentCaller<TUser = unknown> {
  /** JSON: an id, the claims the agent functions and their guards need. */
  user?: TUser;
  scopes?: string[];
}

/**
 * The options of `AiAgents.run()`: those of a call that a run stores and resumes with, so JSON only. Models and agent
 * functions are named; what the module configures for a model (pricing, fallback, retries) applies as for any call.
 */
export interface AiAgentRunOptions {
  /** A model's name. Default `'default'`. */
  model?: string;
  instructions?: string;
  prompt?: string;
  messages?: ModelMessage[];
  /** Agent function names and provider classes (all their agent functions), stored as names. */
  tools?: AiToolSelection;
  /**
   * The agent functions, among `tools`, whose calls wait for a person's approval (`AiAgents.approve()` or `reject()`)
   * before they run. A rejected call, or one nobody answered within `approvalTimeout`, is a tool error the model reads.
   */
  approve?: string[];
  /** How long a call waits for its approval. Default: until it gets one (or the workflow's own timeout). */
  approvalTimeout?: Duration;
  caller?: AiAgentCaller;
  /**
   * Model calls the run may make while the model asks for tools. A whole number of at least 1.
   * Default: `defaults.maxSteps`, else 25.
   */
  maxSteps?: number;
  /**
   * Model calls per workflow instance: after that many, the run goes on in a child workflow,
   * which keeps each instance's journal small (a journal is replayed on every resume, and capped).
   * A whole number of at least 1. Default 10.
   */
  stepsPerInstance?: number;
  /** `maxTokens` and `maxCost`, over the whole run. Default: the module's `defaults.budget`. */
  budget?: AiBudget;
  /** Names the run's model calls in usage events. */
  label?: string;
  toolChoice?: AiToolChoice;
  /** Models tried in order when the model fails after its retries, by name; `false` for none. */
  fallback?: string | string[] | false;
  maxOutputTokens?: number;
  temperature?: number;
  providerOptions?: Record<string, Record<string, any>>;
}

/** What `AiAgents.run()` resolves with: a call's result, as JSON. */
export type AiAgentRunResult = AiTextResult;

/** A person's answer to a tool call that waits for approval: the `TOOL_APPROVAL` signal's payload. */
export interface AiToolApproval {
  approved: boolean;
  /** Told to the model when the call is rejected. */
  reason?: string;
  /** Who answered, for your records. */
  by?: unknown;
}

/** A tool call waiting for approval, as `AiAgents.pendingApprovals()` lists it. */
export interface AiPendingApproval {
  /**
   * What `approve()` and `reject()` take. Read it from `pendingApprovals()` rather than build it: its format can change
   * between versions, and a key read before an update does not answer a call that waits after it.
   */
  key: string;
  /** The run (the instance `AiAgents.run()` runs in). */
  runId: string;
  toolCallId: string;
  toolName: string;
  input: unknown;
}
