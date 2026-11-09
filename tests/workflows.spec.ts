/**
 * `@nestjs/ai/workflows`: the agent loop as a durable workflow, on `@nestjs/workflows`' in-memory store with a
 * manual clock. A "restart" closes the application and boots another on the same store, as a database outlives a
 * process: model calls and tool calls are journaled steps, approvals are signals that can come days later, and a
 * replay never calls a model or a tool again for what is journaled.
 */
import { Injectable } from '@nestjs/common';
import { Test, type TestingModule } from '@nestjs/testing';
import {
  InMemoryWorkflowStore,
  ManualWorkflowClock,
  Workflow,
  WorkflowClient,
  WorkflowStorage,
  WorkflowsModule,
  WorkflowWorker,
  type WorkflowContext,
} from '@nestjs/workflows';
import * as z from 'zod';
import { AgentFunction, AiModule, type AgentFunctionContext, type AiCallDefaults } from '../lib/index.js';
import { FakeModel, type FakeAnswer, type FakeModelCall } from '../lib/testing/index.js';
import { AiAgents, AiWorkflowsModule } from '../lib/workflows/index.js';

/** What the tools did, across "processes". */
class World {
  readonly calls: Array<{ tool: string; input: unknown; user: unknown; idempotencyKey?: string }> = [];
}

@Injectable()
class StoreTools {
  constructor(private readonly world: World) {}

  @AgentFunction({ name: 'find-order', description: 'Find an order', inputSchema: z.object({ orderId: z.string() }) })
  find({ orderId }: { orderId: string }, ctx: AgentFunctionContext) {
    this.world.calls.push({ tool: 'find-order', input: { orderId }, user: ctx.user, idempotencyKey: ctx.idempotencyKey });
    return { id: orderId, total: 4_200, status: 'delivered' };
  }

  @AgentFunction({ name: 'refund', description: 'Refund an order', inputSchema: z.object({ orderId: z.string() }) })
  refund({ orderId }: { orderId: string }, ctx: AgentFunctionContext) {
    this.world.calls.push({ tool: 'refund', input: { orderId }, user: ctx.user, idempotencyKey: ctx.idempotencyKey });
    return { refunded: orderId };
  }
}

@Workflow('refund-request')
class RefundRequest {
  constructor(private readonly agents: AiAgents) {}

  run(
    ctx: WorkflowContext,
    { email, customer, stepsPerInstance, maxSteps }: { email: string; customer: string; stepsPerInstance?: number; maxSteps?: number },
  ) {
    return this.agents.run(ctx, {
      instructions: 'You handle refund requests of the cat store.',
      prompt: email,
      tools: [StoreTools],
      approve: ['refund'],
      caller: { user: { name: customer } },
      ...(stepsPerInstance !== undefined && { stepsPerInstance }),
      ...(maxSteps !== undefined && { maxSteps }),
    });
  }
}

/** The fake model of a refund request: look the order up, ask for the refund, then answer with what happened. */
const refundAnswers = (): FakeAnswer[] => [
  { toolCalls: [{ name: 'find-order', input: { orderId: 'ord_1' } }] },
  { toolCalls: [{ name: 'refund', input: { orderId: 'ord_1' } }] },
  (call) => `Refund: ${JSON.stringify(call.toolResults[0]!.output)}`,
];

describe('durable agent runs', () => {
  let store: InMemoryWorkflowStore;
  let clock: ManualWorkflowClock;
  let world: World;
  const apps: TestingModule[] = [];

  beforeEach(() => {
    store = new InMemoryWorkflowStore();
    clock = new ManualWorkflowClock();
    world = new World();
  });

  afterEach(async () => {
    for (const app of apps.splice(0)) {
      await app.close();
    }
  });

  /** One "process": the application on the shared store and clock. */
  async function boot(model: FakeModel, defaults: AiCallDefaults = {}) {
    const app = await Test.createTestingModule({
      imports: [
        WorkflowsModule.forRoot({ clock, worker: { enabled: false, shutdownTimeout: 50 } }),
        AiModule.forRoot({ models: { default: model }, defaults: { retry: false, ...defaults } }),
        AiWorkflowsModule,
      ],
      providers: [{ provide: World, useValue: world }, StoreTools, RefundRequest],
    }).compile();
    app.useLogger(false);
    app.get(WorkflowStorage).registerSource(store);
    await app.init();
    apps.push(app);
    return { app, client: app.get(WorkflowClient), worker: app.get(WorkflowWorker), agents: app.get(AiAgents) };
  }

  async function restart(app: TestingModule, model: FakeModel) {
    await app.close();
    apps.splice(apps.indexOf(app), 1);
    return boot(model);
  }

  it('journals each model call and each tool call as a step, and resolves with the answer', async () => {
    const model = new FakeModel([{ toolCalls: [{ name: 'find-order', input: { orderId: 'ord_1' } }] }, 'It was delivered.']);
    const { client, worker } = await boot(model);

    await client.start(RefundRequest, { email: 'Where is ord_1?', customer: 'alice' }, { id: 'run-1' });
    await worker.drain();

    const status = await client.getStatus('run-1', { journal: true });
    expect(status).toMatchObject({ status: 'completed', output: { text: 'It was delivered.', finishReason: 'stop' } });
    expect(status!.journal!.map((entry) => `${entry.kind} ${entry.name}`)).toEqual([
      'step model-0',
      'step tool-0-call-1',
      'step model-1',
    ]);
    expect(world.calls).toEqual([
      { tool: 'find-order', input: { orderId: 'ord_1' }, user: { name: 'alice' }, idempotencyKey: 'run-1:tool-0-call-1' },
    ]);
    expect(model.calls.map((call) => call.tools)).toEqual([
      ['find-order', 'refund'],
      ['find-order', 'refund'],
    ]);
    expect(model.calls[0]!.instructions).toBe('You handle refund requests of the cat store.');
  });

  it('waits for an approval across a restart, days later, and runs the tool once it comes', async () => {
    const model = new FakeModel(refundAnswers());
    const first = await boot(model);
    await first.client.start(RefundRequest, { email: 'Please refund ord_1', customer: 'alice' }, { id: 'run-2' });
    await first.worker.drain();

    expect(await first.client.getStatus('run-2')).toMatchObject({ status: 'suspended' });
    const [pending] = await first.agents.pendingApprovals('run-2');
    expect(pending).toEqual({ key: 'run-2:1:call-2', runId: 'run-2', toolCallId: 'call-2', toolName: 'refund', input: { orderId: 'ord_1' } });
    expect(world.calls.map((call) => call.tool)).toEqual(['find-order']);

    // The process goes away; the approval comes three days later, to another one.
    const second = await restart(first.app, model);
    clock.advance('3d');
    await second.agents.approve(pending!.key, { by: 'sam' });
    await second.worker.drain();

    expect(await second.client.getStatus('run-2')).toMatchObject({
      status: 'completed',
      output: { text: 'Refund: {"refunded":"ord_1"}' },
      customStatus: null,
    });
    expect(world.calls.map((call) => call.tool)).toEqual(['find-order', 'refund']);
    // Each model call was made once: the replay read the first two from the journal.
    expect(model.calls).toHaveLength(3);
    expect(await second.agents.pendingApprovals('run-2')).toEqual([]);
  });

  it('does not run a rejected call, and tells the model why', async () => {
    const model = new FakeModel(refundAnswers());
    const { client, worker, agents } = await boot(model);
    await client.start(RefundRequest, { email: 'Please refund ord_1', customer: 'bob' }, { id: 'run-3' });
    await worker.drain();

    const [pending] = await agents.pendingApprovals('run-3');
    await agents.reject(pending!.key, { reason: 'The order is past the refund window' });
    // A second answer for the same call does not count.
    await agents.approve(pending!.key);
    await worker.drain();

    expect(await client.getStatus('run-3')).toMatchObject({ status: 'completed' });
    expect(world.calls.map((call) => call.tool)).toEqual(['find-order']);
    expect(model.lastCall!.toolResults).toEqual([
      { toolName: 'refund', toolCallId: 'call-2', output: 'The call was rejected: The order is past the refund window' },
    ]);
    const output = (await client.getStatus('run-3'))!.output as { toolResults: unknown[] };
    expect(output.toolResults).toContainEqual(expect.objectContaining({ toolName: 'refund', error: 'The call was rejected: The order is past the refund window' }));
  });

  it.each([
    { where: 'in one workflow', runId: 'run-4', stepsPerInstance: undefined },
    { where: 'across child workflows', runId: 'run-4-child', stepsPerInstance: 1 },
  ])('waits for its own approval when a later step reuses a tool call id, $where', async ({ runId, stepsPerInstance }) => {
    // text-generation-inference, for example, gives every tool call the id "0".
    const model = new FakeModel([
      { toolCalls: [{ name: 'refund', id: '0', input: { orderId: 'ord_1' } }] },
      { toolCalls: [{ name: 'refund', id: '0', input: { orderId: 'ord_2' } }] },
      'Both orders were refunded.',
    ]);
    const { client, worker, agents } = await boot(model);
    await client.start(
      RefundRequest,
      { email: 'Please refund ord_1 and ord_2', customer: 'bob', ...(stepsPerInstance !== undefined && { stepsPerInstance }) },
      { id: runId },
    );
    await worker.drain();

    const [first] = await agents.pendingApprovals(runId);
    await agents.approve(first!.key);
    await worker.drain();
    const [second] = await agents.pendingApprovals(runId);
    expect(second).toMatchObject({ toolCallId: '0', input: { orderId: 'ord_2' } });
    await agents.approve(second!.key);
    await worker.drain();

    expect(await client.getStatus(runId)).toMatchObject({ status: 'completed', output: { text: 'Both orders were refunded.' } });
    expect(world.calls.map((call) => call.input)).toEqual([{ orderId: 'ord_1' }, { orderId: 'ord_2' }]);
    expect(await agents.pendingApprovals(runId)).toEqual([]);
  });

  it('replays a finished tool from the journal when the process dies in the next model call', async () => {
    let hang = true;
    const answer = (call: FakeModelCall): FakeAnswer => {
      if (call.toolResults.length === 0) {
        return { toolCalls: [{ name: 'find-order', input: { orderId: 'ord_1' } }] };
      }
      // The first time, the model call outlives the process.
      if (hang) {
        hang = false;
        return { text: 'never', delay: 60_000 };
      }
      return 'It was delivered.';
    };
    const model = new FakeModel(answer);
    const first = await boot(model);
    await first.client.start(RefundRequest, { email: 'Where is ord_1?', customer: 'alice' }, { id: 'run-4' });
    void first.worker.drain();
    await vi.waitFor(() => expect(model.calls).toHaveLength(2));

    const second = await restart(first.app, model);
    clock.advance('31s'); // the dead process's lease runs out
    await second.worker.drain();

    expect(await second.client.getStatus('run-4')).toMatchObject({ status: 'completed', output: { text: 'It was delivered.' } });
    // The tool ran once; the model call it was in the middle of was made again.
    expect(world.calls.map((call) => call.tool)).toEqual(['find-order']);
    expect(model.calls).toHaveLength(3);
  });

  it('goes on in child workflows every stepsPerInstance model calls, keeping each journal short', async () => {
    const lookups = ['ord_1', 'ord_2', 'ord_3', 'ord_4'];
    const model = new FakeModel((call) => {
      const done = model.calls.length - 1;
      return done < lookups.length ? { toolCalls: [{ name: 'find-order', input: { orderId: lookups[done] } }] } : `Checked ${call.toolResults.length} order(s).`;
    });
    const { client, worker } = await boot(model);
    await client.start(RefundRequest, { email: 'Check all my orders', customer: 'alice', stepsPerInstance: 2 }, { id: 'run-5' });
    await worker.drain();

    const root = await client.getStatus('run-5', { journal: true, children: true });
    expect(root).toMatchObject({ status: 'completed', output: { text: 'Checked 1 order(s).' } });
    expect(root!.journal!.map((entry) => `${entry.kind} ${entry.name}`)).toEqual([
      'step model-0',
      'step tool-0-call-1',
      'step model-1',
      'step tool-1-call-2',
      // The child's start, and its result.
      'child $child:run-5:from-2',
      'signal $result:run-5:from-2',
    ]);
    const child = await client.getStatus('run-5:from-2', { journal: true });
    expect(child!.journal!.map((entry) => entry.name)).toEqual([
      'model-2',
      'tool-2-call-3',
      'model-3',
      'tool-3-call-4',
      '$child:run-5:from-4',
      '$result:run-5:from-4',
    ]);
    expect((root!.output as { steps: unknown[] }).steps).toHaveLength(5);
    expect(world.calls.map((call) => (call.input as { orderId: string }).orderId)).toEqual(lookups);
  });

  /** A model that looks the order up again and again, and never answers. */
  const lookingForever = () => new FakeModel(() => ({ toolCalls: [{ name: 'find-order', input: { orderId: 'ord_1' } }] }));

  it('accepts one model call per run and per workflow instance', async () => {
    const model = lookingForever();
    const { client, worker } = await boot(model);
    await client.start(
      RefundRequest,
      { email: 'Where is ord_1?', customer: 'alice', maxSteps: 1, stepsPerInstance: 1 },
      { id: 'one-step' },
    );
    await worker.drain();

    const status = await client.getStatus('one-step', { children: true });
    expect(status).toMatchObject({
      status: 'completed', output: { finishReason: 'tool-calls' },
    });
    expect(status!.children ?? []).toHaveLength(0);
    expect(model.calls).toHaveLength(1);
    expect(world.calls).toHaveLength(1);
  });

  it.each(['maxSteps', 'stepsPerInstance'] as const)(
    'fails an invalid %s without starting a model or child workflow',
    async (name) => {
      const model = lookingForever();
      const { client, worker } = await boot(model);
      await client.start(
        RefundRequest,
        { email: 'Where is ord_1?', customer: 'alice', [name]: 0 },
        { id: 'invalid-limit' },
      );
      await worker.drain({ maxRounds: 2 });

      const status = await client.getStatus('invalid-limit', {
        journal: true, children: true,
      });
      expect(status).toMatchObject({
        status: 'failed',
        error: { message: `\`${name}\` must be a whole number of at least 1. Got 0.` },
      });
      expect(status!.journal).toHaveLength(0);
      expect(status!.children ?? []).toHaveLength(0);
      expect(model.calls).toHaveLength(0);
      expect(world.calls).toHaveLength(0);
    },
  );

  it("stops after the module's defaults.maxSteps, unless the run sets its own", async () => {
    const model = lookingForever();
    const { client, worker } = await boot(model, { maxSteps: 2 });

    await client.start(RefundRequest, { email: 'Where is ord_1?', customer: 'alice' }, { id: 'run-6' });
    await worker.drain();
    expect(await client.getStatus('run-6')).toMatchObject({ status: 'completed', output: { finishReason: 'tool-calls' } });
    expect(model.calls).toHaveLength(2);

    await client.start(RefundRequest, { email: 'Where is ord_1?', customer: 'alice', maxSteps: 3 }, { id: 'run-7' });
    await worker.drain();
    expect(await client.getStatus('run-7')).toMatchObject({ status: 'completed' });
    expect(model.calls).toHaveLength(5);
  });

  it("stops before the model call that would spend more than the module's defaults.budget", async () => {
    // Each fake model call uses 20 tokens: the third would start with 40 spent.
    const model = lookingForever();
    const { client, worker } = await boot(model, { budget: { maxTokens: 30 } });

    await client.start(RefundRequest, { email: 'Where is ord_1?', customer: 'alice' }, { id: 'run-8' });
    await worker.drain();

    const status = await client.getStatus('run-8');
    expect(status).toMatchObject({ status: 'failed' });
    expect(status!.error!.message).toContain("The call's budget is spent (40 tokens)");
    expect(model.calls).toHaveLength(2);
  });
});
