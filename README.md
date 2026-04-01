# AI Agent SDK

[![npm](https://img.shields.io/npm/v/@nordinian/ai-agent-sdk.svg?style=flat-square)](https://www.npmjs.com/package/@nordinian/ai-agent-sdk) ![Node.js](https://img.shields.io/badge/Node.js-18%2B-brightgreen?style=flat-square) ![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

A multi-model agent SDK for building autonomous AI agents. Use **Claude, Gemini, GPT, DeepSeek, or 100+ other models** to drive the same powerful agent loop — with 61+ built-in tools for file editing, shell execution, code search, web access, MCP, and more.

## Motivation

This project was inspired by Anthropic's official [`claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-typescript). The official SDK is excellent but has two limitations: it requires spawning a local CLI subprocess, and it only supports Claude models.

We wanted an SDK that:
- **Runs the full agent loop in-process** — no subprocess, deploy anywhere (serverless, Docker, CI/CD)
- **Supports any LLM** — swap between Claude, Gemini, GPT, DeepSeek, or local models with one line
- **Keeps 100% of the agentic capabilities** — tools, permissions, memory, context compression, multi-agent, MCP

## Quick start

```sh
bun add @nordinian/ai-agent-sdk
# or: npm install @nordinian/ai-agent-sdk
```

### With Claude

```typescript
import { createAgent } from '@nordinian/ai-agent-sdk'

const agent = createAgent({
  model: 'claude-sonnet-4-6',
  env: { ANTHROPIC_API_KEY: 'your-key' },
})
const result = await agent.prompt('Read package.json and tell me the project name')
console.log(result.text)
```

### With Gemini

```typescript
const agent = createAgent({
  model: 'gemini-2.5-flash',
  env: { GEMINI_API_KEY: 'your-key' },
})
const result = await agent.prompt('Find all TODO comments in this codebase')
console.log(result.text)
```

### With any OpenAI-compatible API

```typescript
// OpenAI
createAgent({ model: 'gpt-4o', env: { OPENAI_API_KEY: 'key' } })

// DeepSeek
createAgent({ model: 'deepseek-chat', env: { DEEPSEEK_API_KEY: 'key' } })

// Groq
createAgent({ model: 'llama-3.3-70b-versatile', env: { GROQ_API_KEY: 'key' } })

// Ollama (local)
createAgent({ model: 'ollama/llama3', env: { OLLAMA_BASE_URL: 'http://localhost:11434/v1' } })

// Any OpenAI-compatible endpoint
createAgent({ model: 'openai-compat/my-model', env: { OPENAI_API_KEY: 'key', OPENAI_BASE_URL: 'https://my-api.com/v1' } })
```

All models share the same 61+ built-in tools. The provider layer handles format translation transparently.

## Supported providers

| Provider | Models | Auth |
|----------|--------|------|
| **Anthropic** | `claude-*` | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini-*` (native `@google/genai` SDK) | `GEMINI_API_KEY` |
| **Google Vertex AI** | `gemini-*` via Vertex | `GOOGLE_CLOUD_PROJECT` + ADC |
| **OpenAI** | `gpt-*`, `o1-*`, `o3-*`, `o4-*` | `OPENAI_API_KEY` |
| **DeepSeek** | `deepseek-*` | `DEEPSEEK_API_KEY` |
| **Groq** | `llama-*`, `mixtral-*`, `qwen-*` | `GROQ_API_KEY` |
| **Mistral** | `mistral-*`, `codestral-*` | `MISTRAL_API_KEY` |
| **Ollama** | `ollama/*` (any local model) | `OLLAMA_BASE_URL` |
| **Custom** | `openai-compat/*` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |

## API

### `createAgent(options)`

Create a reusable agent with persistent session state.

```typescript
const agent = createAgent({ model: 'gemini-2.5-pro' })

// Blocking
const result = await agent.prompt('Explain the architecture of this project')
console.log(result.text)

// Streaming
for await (const event of agent.query('Refactor the error handling')) {
  // handle streaming events
}

// Session persists across calls
agent.getMessages()  // conversation history
agent.clear()        // reset
```

### `query({ prompt, options })`

One-shot query, compatible with the official `claude-agent-sdk` API.

```typescript
import { query } from '@nordinian/ai-agent-sdk'

for await (const message of query({
  prompt: 'Find and fix the bug in auth.py',
  options: {
    model: 'gemini-2.5-flash',
    allowedTools: ['Read', 'Edit', 'Bash'],
  },
})) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) {
      if ('text' in block) console.log(block.text)
    }
  }
}
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `string` | `claude-sonnet-4-6` | Model ID — any supported provider |
| `apiKey` | `string` | `env.ANTHROPIC_API_KEY` | API key (Anthropic) |
| `cwd` | `string` | `process.cwd()` | Working directory for tools |
| `systemPrompt` | `string` | — | Custom system prompt |
| `tools` | `Tool[]` | All built-in | Available tools |
| `allowedTools` | `string[]` | — | Tool whitelist (e.g. `['Read', 'Glob']`) |
| `permissionMode` | `string` | `bypassPermissions` | `acceptEdits` / `bypassPermissions` / `plan` / `default` |
| `maxTurns` | `number` | `100` | Max agentic turns |
| `mcpServers` | `object` | — | MCP server configurations |
| `agents` | `object` | — | Custom subagent definitions |
| `thinking` | `object` | — | Extended thinking configuration |
| `env` | `object` | — | Environment variables for provider auth |
| `geminiGrounding` | `boolean` | — | Enable Google Search grounding (Gemini only) |

### Environment variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic | Claude API key |
| `ANTHROPIC_MODEL` | Anthropic | Default model |
| `GEMINI_API_KEY` | Gemini | Google AI API key |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI | GCP region (default: `us-central1`) |
| `GOOGLE_VERTEX_AI` | Vertex AI | Set to `'true'` to use Vertex AI |
| `OPENAI_API_KEY` | OpenAI | OpenAI API key |
| `OPENAI_BASE_URL` | OpenAI-compat | Base URL for any OpenAI-compatible API |
| `DEEPSEEK_API_KEY` | DeepSeek | DeepSeek API key |
| `GROQ_API_KEY` | Groq | Groq API key |
| `MISTRAL_API_KEY` | Mistral | Mistral API key |
| `OLLAMA_BASE_URL` | Ollama | Ollama URL (default: `http://localhost:11434/v1`) |

## Built-in tools

| Tool | Description |
|------|-------------|
| `Read` | Read files with line numbers, images, PDFs |
| `Write` | Create or overwrite files |
| `Edit` | Precise string replacement in files |
| `Bash` | Execute shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents with regex (ripgrep) |
| `WebFetch` | Fetch and parse web content |
| `WebSearch` | Search the web |
| `Agent` | Spawn subagents for parallel work |
| `NotebookEdit` | Edit Jupyter notebooks |
| `SendMessage` | Inter-agent messaging |
| `TeamCreate` / `TeamDelete` | Multi-agent teams |
| `EnterWorktree` / `ExitWorktree` | Git worktree isolation |
| `ListMcpResources` / `ReadMcpResource` | MCP resource access |
| `TaskCreate` / `TaskUpdate` / `TaskList` | Task management |

[See all 61+ tools in the source.](./src/tools)

## Architecture

```
Your code
    │
    ▼
createAgent({ model: 'gemini-2.5-flash' })
    │
    ▼
┌─────────────────────────────┐
│       QueryEngine           │  ← Full agent loop (in-process)
│  system prompt, memory,     │
│  context compression,       │
│  tool execution, permissions│
└──────────┬──────────────────┘
           │
    ┌──────▼──────┐
    │  LLMProvider │  ← Provider abstraction layer
    │   Registry   │
    └──┬────┬────┬─┘
       │    │    │
  Anthropic Gemini OpenAI-compat
  (native) (native) (universal)
       │    │    │
    Claude  Gemini  GPT / DeepSeek /
                    Groq / Mistral /
                    Ollama / 100+
```

The engine uses Anthropic's message format internally. Each provider translates to/from its native format at the API boundary. This means the entire engine (permissions, memory, context compression, multi-agent, MCP) works unchanged regardless of which model you choose.

### What's under the hood

| Component | Description |
|-----------|-------------|
| **Provider Layer** | Pluggable LLM providers with auto-routing by model name |
| **System Prompt** | Full prompt construction with boundary caching |
| **Permission System** | 4-layer pipeline: rules, low-risk skip, whitelist, AI classifier |
| **Memory System** | Auto-memory with 4 types, background organizer |
| **Context Compression** | 9-segment structured extraction |
| **Multi-Agent** | Leader/Teammate teams, Git worktree isolation |
| **MCP Client** | stdio, SSE, HTTP transports |
| **Tool Execution** | Concurrent batching for read-only, serial for mutations |
| **Loop Detection** | Hash-based tool repetition + content chanting detection |
| **Context Overflow Guard** | Pre-flight token estimation before API calls |
| **Provider Token Estimation** | Per-provider heuristic token counting (ASCII/CJK-aware) |

### Agent Robustness (v0.2.1)

Built-in modules to keep agents productive during long tasks:

```typescript
import {
  LoopDetector,
  loopBreakMessage,
  checkContextOverflow,
  estimateTextTokens,
  getProviderContextWindow,
} from '@nordinian/ai-agent-sdk'

// Loop detection — detects infinite tool-calling loops
const detector = new LoopDetector()
const result = detector.checkAssistantMessage(assistantMessage)
if (result.detected) {
  console.log(loopBreakMessage(result))  // inject corrective message
  detector.clearDetection()               // allow retry with new approach
}

// Context overflow guard — pre-check before API calls
const check = checkContextOverflow('gemini-2.5-flash', messages, systemPrompt, tools)
if (check.status === 'overflow') {
  console.log(check.message)  // "Context overflow: estimated 1.2M tokens..."
}

// Token estimation — provider-aware heuristics
estimateTextTokens('Hello world', 'gemini')      // ASCII: 0.25 tokens/char
estimateTextTokens('你好世界', 'gemini')           // CJK: 1.3 tokens/char
getProviderContextWindow('gemini-2.5-pro')         // 1,048,576
getProviderContextWindow('gpt-4o')                 // 128,000
```

**Auto-compact** automatically triggers for all providers when context approaches the limit:
- **Anthropic models**: Full summarization-based compaction (uses Claude)
- **Non-Anthropic models**: Truncation-based compaction (preserves last 30% of conversation)

## Custom providers

Register your own LLM provider:

```typescript
import { registerProvider, type LLMProvider } from '@nordinian/ai-agent-sdk'

class MyProvider implements LLMProvider {
  readonly type = 'openai-compat' as const
  supportsModel(model: string) { return model.startsWith('my-') }
  async createMessage(params) { /* ... */ }
  async *createMessageStream(params) { /* ... */ }
}

registerProvider(new MyProvider())
```

All inputs and outputs use Anthropic message format. Your provider handles translation internally.

## MCP integration

```typescript
const agent = createAgent({
  model: 'gemini-2.5-flash',
  mcpServers: {
    filesystem: {
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'],
    },
  },
})
```

## Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Simple Query](./examples/01-simple-query.ts) | Streaming with `createAgent().query()` |
| 02 | [Multi-Tool](./examples/02-multi-tool.ts) | Glob + Bash orchestration |
| 03 | [Multi-Turn](./examples/03-multi-turn.ts) | Session persistence across turns |
| 04 | [Prompt API](./examples/04-prompt-api.ts) | Blocking `agent.prompt()` |
| 05 | [System Prompt](./examples/05-custom-system-prompt.ts) | Custom system prompt |
| 06 | [MCP Server](./examples/06-mcp-server.ts) | MCP stdio transport |
| 07 | [Custom Tools](./examples/07-custom-tools.ts) | User-defined tools |
| 08 | [Official API Compat](./examples/08-official-api-compat.ts) | `query()` drop-in compatible |
| 09 | [Subagents](./examples/09-subagents.ts) | Agent delegation |
| 10 | [Permissions](./examples/10-permissions.ts) | Read-only agent |
| 11 | [Gemini](./examples/11-gemini-basic.ts) | Gemini as base model |
| 12 | [Vertex AI](./examples/12-gemini-vertex-ai.ts) | Vertex AI authentication |

```sh
npx tsx examples/11-gemini-basic.ts
```

## Acknowledgments

This project was inspired by and builds upon the architecture of [`@anthropic-ai/claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-typescript). We extend it with a multi-model provider layer that enables the same agentic capabilities across different LLM providers.

## License

MIT
