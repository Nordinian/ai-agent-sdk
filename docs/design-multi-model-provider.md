# Multi-Model Provider Design Document

> Date: 2026-03-31
> Status: Implemented
> Author: Kinson + Claude

## 1. Goal

Add a multi-model Provider abstraction layer to AI Agent SDK, enabling Gemini (and other models in the future) to drive the full Agent Loop with 100% of Claude Code's 61+ tool capabilities.

**First non-Anthropic Provider:** Google Gemini (native `@google/genai` SDK), supporting both Gemini API and Vertex AI.

## 2. Summary

- **What:** LLMProvider interface + GeminiNativeProvider + AnthropicProvider wrapper + OpenAICompatProvider
- **Why:** Remove Anthropic single-vendor lock-in, support multi-model
- **For whom:** SDK users (developers), via `createAgent({ model: 'gemini-2.5-pro' })`
- **Key constraint:** 1M-line engine code untouched, changes only at the API call boundary (< 0.06% intrusion rate)
- **Non-goals:** No changes to existing Anthropic user experience

## 3. Assumptions

- Gemini 2.5 Pro/Flash function calling is capable enough to drive 61+ tools
- `@google/genai` SDK streaming API is stable (validated by gemini-cli)
- Message format translation is reversible (Anthropic <-> Gemini without semantic loss)
- Gemini's tool count limit (~128) is sufficient for built-in tools + MCP tools

## 4. Architecture

```
                    ┌──────────────────────────┐
                    │   AI Agent SDK Internals  │
                    │  (Anthropic msg format)   │
                    │   1M LOC, untouched       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   LLMProvider Interface   │
                    │   (API call boundary)     │
                    └────┬───────┬────────┬────┘
                         │       │        │
              ┌──────────▼──┐ ┌──▼─────┐ ┌▼──────────────┐
              │ Anthropic   │ │ Gemini │ │ OpenAICompat   │
              │ Provider    │ │ Native │ │ Provider       │
              │ (existing)  │ │ Provider│ │ (universal)    │
              │ minimal chg │ │@google/ │ │ openai npm     │
              └─────────────┘ │ genai  │ │ → 100+ models  │
                              └────────┘ └────────────────┘
                              full feat    DeepSeek/Groq/
                              thinking     Mistral/Ollama
                              grounding    any compat API
                              Vertex AI
```

## 5. LLMProvider Interface

```typescript
// src/services/api/provider.ts

export interface LLMProvider {
  readonly type: LLMProviderType
  supportsModel(model: string): boolean
  createMessage(params: LLMCreateParams): Promise<LLMResponse>
  createMessageStream(params: LLMCreateParams): AsyncGenerator<LLMStreamEvent>
  initialize?(): Promise<void>
  shutdown?(): Promise<void>
}
```

**Key decision:** Interface inputs and outputs remain in Anthropic format. Translation logic is encapsulated within each Provider.

## 6. Provider Registry

```typescript
// src/services/api/registry.ts
const providers: LLMProvider[] = []

export function registerProvider(provider: LLMProvider): void
export function resolveProvider(model: string): LLMProvider
```

Model name matching:
- `gemini-*`, `vertex/*` → GeminiProvider
- `gpt-*`, `o1-*`, `deepseek-*`, etc. → OpenAICompatProvider
- `claude-*`, `anthropic/*` → AnthropicProvider (default fallback)

## 7. Message Format Translation

### Anthropic → Gemini

| Anthropic | Gemini |
|-----------|--------|
| `role: 'assistant'` | `role: 'model'` |
| `{ type: 'text', text }` | `{ text }` |
| `{ type: 'tool_use', id, name, input }` | `{ functionCall: { name, args } }` |
| `{ type: 'tool_result', tool_use_id, content }` | `{ functionResponse: { name, response } }` |
| `{ type: 'thinking', thinking }` | `{ thought: true, text }` |
| `{ type: 'image', source }` | `{ inlineData: { mimeType, data } }` |
| `system: string \| SystemBlock[]` | `config.systemInstruction` (multi-segment merged into one) |

**Key detail:** `tool_result` translation requires an `id → name` mapping table, because Anthropic associates by id while Gemini associates by name.

### Gemini → Anthropic

| Gemini | Anthropic |
|--------|-----------|
| `finishReason: 'STOP'` | `stop_reason: 'end_turn'` |
| `finishReason: 'MAX_TOKENS'` | `stop_reason: 'max_tokens'` |
| Response with `functionCall` | `stop_reason: 'tool_use'` (critical — engine enters tool branch on this) |
| `usageMetadata.promptTokenCount` | `usage.input_tokens` |
| `usageMetadata.candidatesTokenCount` | `usage.output_tokens` |

## 8. Tool Schema Translation

Core difference: JSON Schema `type` field lowercase → Gemini uppercase enum.

```typescript
function convertSchemaToGemini(schema: JSONSchema): GeminiSchema {
  // Recursive conversion: 'string' → 'STRING', 'object' → 'OBJECT'
  // Handle oneOf/anyOf: merge into a loose object (Gemini doesn't support union types)
  // Remove Gemini-unsupported fields: $schema, additionalProperties, examples
}
```

Tool count limit: truncate low-priority MCP tools when exceeding 128.

## 9. Streaming Protocol Adaptation

Gemini simple chunk sequence → wrapped into Anthropic's 7 fine-grained event types:

```
chunk.text         → content_block_start + content_block_delta(text_delta) + content_block_stop
chunk.functionCall → content_block_start(tool_use) + content_block_delta(input_json_delta) + content_block_stop
chunk.thought      → content_block_start(thinking) + content_block_delta(thinking_delta) + content_block_stop
finishReason       → message_delta(stop_reason, usage) + message_stop
first chunk        → message_start
```

## 10. Error Handling

Gemini errors mapped to Anthropic SDK error format, reusing existing withRetry.ts logic:

| Gemini Error | Mapped To | withRetry Behavior |
|-------------|-----------|-------------------|
| `RESOURCE_EXHAUSTED` (429) | status 429 | rate limit retry |
| `UNAVAILABLE` (503) | status 529 | overloaded retry |
| `UNAUTHENTICATED` (401) | status 401 | auth refresh |
| token limit exceeded | status 400 + PTL message | prompt too long handling |

## 11. Gemini-Specific Configuration

- **Safety settings:** default `BLOCK_NONE` (agent needs to handle code and shell)
- **Thinking tokens:** mapped to Anthropic thinking block
- **Grounding:** Google Search integration (opt-in via `geminiGrounding`)
- **Context caching:** via `geminiCachedContent` parameter
- **Vertex AI:** auto-switch via `@google/genai` vertexai config
- **Model context window:** gemini-2.5-pro/flash = 1M tokens

## 12. Change List

| File | Type | Lines |
|------|------|-------|
| `src/services/api/provider.ts` | New | ~73 |
| `src/services/api/registry.ts` | New | ~77 |
| `src/services/api/providers/anthropic.ts` | New | ~50 |
| `src/services/api/providers/gemini.ts` | New | ~443 |
| `src/services/api/providers/gemini-translator.ts` | New | ~315 |
| `src/services/api/providers/gemini-tool-translator.ts` | New | ~239 |
| `src/services/api/providers/openai-compat.ts` | New | ~750 |
| `src/services/api/claude.ts` | Modified | ~150 lines added |
| `src/agent.ts` | Modified | ~20 lines |
| `src/utils/model/providers.ts` | Modified | ~10 lines |
| `src/setup-globals.ts` | Modified | ~5 lines |
| `src/sdk.ts` | Modified | ~5 lines |
| `package.json` | Modified | +2 deps |
| **Total** | **~1,947 new + ~200 modified** | **< 0.06% intrusion** |

## 13. User API

```typescript
// Gemini API
const agent = createAgent({
  model: 'gemini-2.5-pro',
  env: { GEMINI_API_KEY: 'your-key' },
})

// Vertex AI
const agent = createAgent({
  model: 'gemini-2.5-flash',
  env: { GOOGLE_VERTEX_AI: 'true', GOOGLE_CLOUD_PROJECT: 'my-project' },
})

// OpenAI
const agent = createAgent({
  model: 'gpt-4o',
  env: { OPENAI_API_KEY: 'your-key' },
})

// Anthropic (100% backward compatible)
const agent = createAgent({ model: 'claude-sonnet-4-6' })

// All share the same 61+ tools
const result = await agent.prompt('Find and fix the bug in auth.py')
```

## 14. Decision Log

| # | Decision | Alternatives | Rationale |
|---|----------|-------------|-----------|
| D1 | Hybrid approach C | Pure translator / Full refactor | Pragmatic, lowest intrusion rate |
| D2 | Native @google/genai | OpenAI compat endpoint | Compat endpoint beta, unstable |
| D3 | Registry pattern | Env vars / Hardcoded | Auto-routing by model name |
| D4 | Interface keeps Anthropic format | Intermediate format | Zero engine changes |
| D5 | Errors mapped to Anthropic format | Rewrite withRetry | Reuse mature logic |
| D6 | Safety defaults to most permissive | User-configurable | Agent needs code/shell ops |
| D7 | id↔name mapping in translator | Change engine format | Minimal intrusion |
| D8 | oneOf flattened to merged object | Skip / Take first | Preserves most information |
| D9 | v1 skips grounding/caching | Full features | YAGNI |
| D10 | OpenAICompat as separate module | Build simultaneously | Decoupled delivery |

## 15. Future Extensions

- **Gemini grounding:** Google Search integration (implemented, opt-in)
- **Gemini context caching:** long conversation performance optimization (implemented, opt-in)
- **Model routing:** auto-select optimal model based on task type
