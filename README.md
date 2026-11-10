<p align="center">
  <a href="http://nestjs.com/" target="blank"><img src="https://nestjs.com/img/logo-small.svg" width="120" alt="Nest Logo" /></a>
</p>

[travis-image]: https://api.travis-ci.org/nestjs/nest.svg?branch=master
[travis-url]: https://travis-ci.org/nestjs/nest
[linux-image]: https://img.shields.io/travis/nestjs/nest/master.svg?label=linux
[linux-url]: https://travis-ci.org/nestjs/nest

  <p align="center">A progressive <a href="http://nodejs.org" target="blank">Node.js</a> framework for building efficient and scalable server-side applications.</p>
    <p align="center">
<a href="https://www.npmjs.com/~nestjscore"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>
<a href="https://www.npmjs.com/~nestjscore"><img src="https://img.shields.io/npm/l/@nestjs/core.svg" alt="Package License" /></a>
<a href="https://www.npmjs.com/~nestjscore"><img src="https://img.shields.io/npm/dm/@nestjs/core.svg" alt="NPM Downloads" /></a>
<a href="https://discord.gg/G7Qnnhy" target="_blank"><img src="https://img.shields.io/badge/discord-online-brightgreen.svg" alt="Discord"/></a>
<a href="https://opencollective.com/nest#backer"><img src="https://opencollective.com/nest/backers/badge.svg" alt="Backers on Open Collective" /></a>
<a href="https://opencollective.com/nest#sponsor"><img src="https://opencollective.com/nest/sponsors/badge.svg" alt="Sponsors on Open Collective" /></a>
  <a href="https://paypal.me/kamilmysliwiec"><img src="https://img.shields.io/badge/Donate-PayPal-dc3d53.svg"/></a>
  <a href="https://twitter.com/nestframework"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow"></a>
</p>
  <!--[![Backers on Open Collective](https://opencollective.com/nest/backers/badge.svg)](https://opencollective.com/nest#backer)
  [![Sponsors on Open Collective](https://opencollective.com/nest/sponsors/badge.svg)](https://opencollective.com/nest#sponsor)-->

## Description

Language models for [Nest](https://github.com/nestjs/nest): named models through dependency injection, model calls with timeouts, retries and fallback models (run on [`@nestjs/resilience`](https://github.com/nestjs/resilience) policies, or your own), budgets and usage events, structured output validated by a schema, the application's own provider methods as agent functions its agents call (through Nest's guards, interceptors, pipes and exception filters, as the user the agent acts for), answers streamed to the browser on Express, Fastify and the Node adapter, and durable agent runs on [`@nestjs/workflows`](https://github.com/nestjs/workflows) (`@nestjs/ai/workflows`) that survive restarts and wait for a person's approval. You write an agent function with `@AgentFunction()`, and the model is offered it as a tool; a method can be an MCP tool too, served by [`@nestjs/mcp`](https://github.com/nestjs/mcp), with its `@Tool()` stacked on it. Embedding models have their own named registry, dependency injection, resilient single and batch calls, and usage events. Comes with fake language and embedding models for tests. Built on the [AI SDK](https://ai-sdk.dev) (`ai`) and its provider packages.

## Installation

```bash
$ npm i --save @nestjs/ai @nestjs/resilience ai @ai-sdk/anthropic zod
```

Any `@ai-sdk/*` provider package works (`@ai-sdk/openai`, `@ai-sdk/google`, ...). Zod 4.2 or later for the schemas; any other [Standard Schema](https://standardschema.dev) library that produces JSON Schema works too. Node.js 22 or later.

Each major version of `@nestjs/ai` supports one major version of the AI SDK: this one, `ai` 7.

## Quick Start

Register the models, and call them:

```ts
import { anthropic } from '@ai-sdk/anthropic';
import { Injectable, Module } from '@nestjs/common';
import { AgentFunction, AiModule, AiService, type AgentFunctionContext } from '@nestjs/ai';
import * as z from 'zod';

@Injectable()
export class OrdersTools {
  constructor(private readonly orders: OrdersService) {}

  @AgentFunction({
    name: 'find-order',
    description: 'Find an order by its id',
    inputSchema: z.object({ orderId: z.string() }),
  })
  find({ orderId }: { orderId: string }, ctx: AgentFunctionContext) {
    return this.orders.findFor(ctx.request?.user, orderId);
  }
}

@Injectable()
export class SupportService {
  constructor(private readonly ai: AiService) {}

  async answer(question: string, request: unknown) {
    const { text } = await this.ai.generateText({
      prompt: question,
      tools: ['find-order'],
      caller: { request },
    });
    return text;
  }
}

@Module({
  imports: [
    AiModule.forRoot({
      models: {
        default: anthropic('claude-opus-5-5'),
        fast: anthropic('claude-haiku-5-5'),
      },
      defaults: { timeout: '30s', retry: { attempts: 3 } },
    }),
  ],
  providers: [OrdersService, OrdersTools, SupportService],
})
export class AppModule {}
```

A method that MCP clients call too also carries `@nestjs/mcp`'s `@Tool()`, with options of its own: the two decorators don't read each other's, and one schema constant serves both. `generateObject({ schema })` answers a typed object, and `streamText()` an answer a controller sends to the browser with `pipeUIMessageStreamToResponse(res)`, as the AI SDK's `useChat()` reads it. Agent function calls run through the guards, interceptors, pipes and filters of the class and the method, with the context type `'ai'`, so `AiExecutionContext` reads them in enhancers (on a method with both decorators, they see `'mcp'` for MCP clients' calls), and `@nestjs/authentication` (0.1.4 or later) and `@nestjs/authorization` (0.1.2 or later) authenticate and authorize them. `@nestjs/ai/testing` has `FakeModel`, which answers with text, tool calls, objects or errors, without a network.

The [sample](sample) is a cat store: it triages support emails, and runs a chat assistant that calls the same `find-order` method its MCP server serves. `npm run build && npx tsc -p sample/tsconfig.json`, then `node sample/dist/main.js` (with `ANTHROPIC_API_KEY` set, Claude answers; without it, fake models do).

Read [Overview](https://docs.nestjs.com/ai/overview) to get started and for the module's options, [Models](https://docs.nestjs.com/ai/models) for retries, timeouts, fallback models and budgets, [Tools](https://docs.nestjs.com/ai/tools) for agent functions, the call context, guards and authorization, [Agents](https://docs.nestjs.com/ai/agents) for the agent loop, streaming to the browser and durable runs, [Embeddings](https://docs.nestjs.com/ai/embeddings) for embedding models and batches, and [Testing](https://docs.nestjs.com/ai/testing) for the fake models.

Model `pricing` sets prices per million tokens for cost reporting and `budget.maxCost`. Its `input`
and `output` prices are required, and `cachedInput` is optional (default: `input`). Every price must
be finite and nonnegative; invalid pricing fails at startup, before a provider is called. Zero
prices are supported:

```ts
AiModule.forRoot({
  models: {
    default: {
      model: anthropic('claude-opus-5-5'),
      pricing: { input: 4, output: 20, cachedInput: 0.2 },
    },
  },
});
```

## Support

Nest is an MIT-licensed open source project. It can grow thanks to the sponsors and support by the amazing backers. If you'd like to join them, please [read more here](https://docs.nestjs.com/support).

## Stay in touch

- Author - [Kamil Myśliwiec](https://twitter.com/kammysliwiec)
- Website - [https://nestjs.com](https://nestjs.com/)
- Twitter - [@nestframework](https://twitter.com/nestframework)

## License

Nest is [MIT licensed](LICENSE).
