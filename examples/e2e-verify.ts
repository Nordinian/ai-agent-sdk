#!/usr/bin/env npx tsx
/**
 * E2E Verification Script — Open Agent SDK v0.3.0
 *
 * Tests the full multi-model provider system without making real API calls.
 * Validates: provider routing, alias resolution, token estimation,
 * context overflow guard, loop detection, and multimodal translation.
 *
 * Run: npx tsx examples/e2e-verify.ts
 */

// Import directly from modules (bypass setup-globals to avoid UI dependency chain)
import { registerProvider, resolveProvider, isNonAnthropicModel, clearProviders } from '../src/services/api/registry.js'
import { GeminiProvider } from '../src/services/api/providers/gemini.js'
import { OpenAICompatProvider } from '../src/services/api/providers/openai-compat.js'
import { AnthropicProvider } from '../src/services/api/providers/anthropic.js'
import { LoopDetector, loopBreakMessage } from '../src/services/loopDetection.js'
import { checkContextOverflow } from '../src/services/contextOverflowGuard.js'
import {
  estimateTextTokens,
  estimateMessagesTokens,
  estimateMediaTokens,
  getProviderContextWindow,
} from '../src/services/providerTokenEstimation.js'

// Register providers manually (normally done in setup-globals.ts)
clearProviders()
registerProvider(new AnthropicProvider())
registerProvider(new GeminiProvider())
registerProvider(new OpenAICompatProvider())

let passed = 0
let failed = 0

function assert(condition: boolean, name: string) {
  if (condition) {
    console.log(`  ✅ ${name}`)
    passed++
  } else {
    console.log(`  ❌ ${name}`)
    failed++
  }
}

// ============================================================================
console.log('\n🔹 Scenario 1: Provider Registration & Routing')
// ============================================================================

// Providers are auto-registered via setup-globals.ts import in sdk.ts

assert(resolveProvider('claude-sonnet-4-6').type === 'anthropic', 'Claude → AnthropicProvider')
assert(resolveProvider('gemini-2.5-flash').type === 'gemini', 'Gemini → GeminiProvider')
assert(resolveProvider('gpt-4o').type === 'openai-compat', 'GPT → OpenAICompatProvider')
assert(resolveProvider('deepseek-chat').type === 'openai-compat', 'DeepSeek → OpenAICompatProvider')
assert(resolveProvider('mistral-large-latest').type === 'openai-compat', 'Mistral → OpenAICompatProvider')
assert(resolveProvider('unknown-model').type === 'anthropic', 'Unknown → Anthropic fallback')

// Model aliases
assert(resolveProvider('gemini-auto').type === 'gemini', 'gemini-auto alias → GeminiProvider')
assert(resolveProvider('gemini-pro').type === 'gemini', 'gemini-pro alias → GeminiProvider')
assert(resolveProvider('gemini-flash').type === 'gemini', 'gemini-flash alias → GeminiProvider')
assert(resolveProvider('gemini-lite').type === 'gemini', 'gemini-lite alias → GeminiProvider')

// isNonAnthropicModel
assert(isNonAnthropicModel('gemini-2.5-pro') === true, 'Gemini is non-Anthropic')
assert(isNonAnthropicModel('gpt-4o') === true, 'GPT is non-Anthropic')
assert(isNonAnthropicModel('claude-sonnet-4-6') === false, 'Claude is Anthropic')

// ============================================================================
console.log('\n🔹 Scenario 2: Token Estimation (Provider-Aware)')
// ============================================================================

// ASCII text — all providers similar
const asciiTokens = estimateTextTokens('Hello, world!', 'gemini')
assert(asciiTokens > 0 && asciiTokens < 10, `ASCII: ${asciiTokens} tokens (expected 1-10)`)

// CJK text — Gemini higher than Anthropic
const cjkGemini = estimateTextTokens('你好世界测试', 'gemini')
const cjkAnthropic = estimateTextTokens('你好世界测试', 'anthropic')
assert(cjkGemini > cjkAnthropic, `CJK: Gemini ${cjkGemini} > Anthropic ${cjkAnthropic}`)

// Message estimation
const messages = [
  { role: 'user', content: 'What is 2+2?' },
  { role: 'assistant', content: 'The answer is 4.' },
]
const msgTokens = estimateMessagesTokens(messages, 'gemini')
assert(msgTokens > 0, `Messages: ${msgTokens} tokens`)

// Media tokens
assert(estimateMediaTokens('image/png', 'gemini') === 3000, 'Gemini image: 3000 tokens')
assert(estimateMediaTokens('application/pdf', 'gemini') === 25800, 'Gemini PDF: 25800 tokens')

// Context windows
assert(getProviderContextWindow('gemini-2.5-pro') === 1_048_576, 'Gemini Pro: 1M window')
assert(getProviderContextWindow('gpt-4o') === 128_000, 'GPT-4o: 128K window')
assert(getProviderContextWindow('deepseek-chat') === 64_000, 'DeepSeek: 64K window')
assert(getProviderContextWindow('unknown-model') === null, 'Unknown model: null')

// ============================================================================
console.log('\n🔹 Scenario 3: Context Overflow Guard')
// ============================================================================

const smallConvo = [{ role: 'user', content: 'Hi' }]
const overflowCheck = checkContextOverflow('gemini-2.5-flash', smallConvo)
assert(overflowCheck.status === 'ok', `Small convo: ${overflowCheck.status} (${overflowCheck.estimatedTokens} tokens)`)
assert(overflowCheck.contextWindow === 1_048_576, `Window: ${overflowCheck.contextWindow}`)

// Large conversation on small context window
const bigConvo = Array.from({ length: 500 }, (_, i) => ({
  role: i % 2 === 0 ? 'user' : 'assistant',
  content: 'x'.repeat(1000),
}))
const bigCheck = checkContextOverflow('deepseek-chat', bigConvo)
assert(bigCheck.status === 'overflow' || bigCheck.status === 'warning',
  `Big convo on 64K: ${bigCheck.status} (${bigCheck.estimatedTokens} / ${bigCheck.contextWindow})`)

// ============================================================================
console.log('\n🔹 Scenario 4: Loop Detection')
// ============================================================================

const detector = new LoopDetector()

// No loop on varied tool calls
for (let i = 0; i < 10; i++) {
  const r = detector.checkToolCall('read_file', { path: `/file${i}.ts` })
  assert(!r.detected, `Varied call ${i}: no loop`)
  if (i >= 4) break // Just test first 5
}

// Reset and test loop detection
detector.reset()
const loopArgs = { path: '/same/file.ts' }
for (let i = 0; i < 4; i++) {
  detector.checkToolCall('read_file', loopArgs)
}
const loopResult = detector.checkToolCall('read_file', loopArgs) // 5th identical
assert(loopResult.detected === true, `5th identical call: loop detected!`)
assert(loopResult.type === 'tool_call_repetition', `Type: ${loopResult.type}`)

// Break message
const breakMsg = loopBreakMessage(loopResult)
assert(breakMsg.includes('LOOP DETECTED'), `Break message contains LOOP DETECTED`)

// Clear and verify recovery
detector.clearDetection()
const afterClear = detector.checkToolCall('new_tool', { x: 1 })
assert(!afterClear.detected, 'After clearDetection: no loop')

// checkAssistantMessage integration
detector.reset()
for (let i = 0; i < 4; i++) {
  detector.checkToolCall('bash', { command: 'echo hello' })
}
const msgResult = detector.checkAssistantMessage({
  content: [
    { type: 'tool_use', id: 'toolu_5', name: 'bash', input: { command: 'echo hello' } },
  ],
})
assert(msgResult.detected === true, 'checkAssistantMessage triggers loop on 5th')

// ============================================================================
console.log('\n🔹 Scenario 5: Multimodal Translation (Gemini)')
// ============================================================================

import {
  toGeminiContents,
  toGeminiSystemInstruction,
  ToolCallIdMap,
} from '../src/services/api/providers/gemini-translator.js'

const idMap = new ToolCallIdMap()

// Image base64
const imgMsg = [{
  role: 'user',
  content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc' } }],
}] as any[]
const imgResult = toGeminiContents(imgMsg, idMap)
assert(imgResult[0].parts[0].inlineData?.mimeType === 'image/png', 'Image base64 → inlineData')

// Document PDF
const pdfMsg = [{
  role: 'user',
  content: [{ type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: 'pdfdata' } }],
}] as any[]
const pdfResult = toGeminiContents(pdfMsg, idMap)
assert(pdfResult[0].parts[0].inlineData?.mimeType === 'application/pdf', 'PDF → inlineData')

// Audio
const audioMsg = [{
  role: 'user',
  content: [{ type: 'audio', source: { type: 'base64', media_type: 'audio/wav', data: 'wav' } }],
}] as any[]
const audioResult = toGeminiContents(audioMsg, idMap)
assert(audioResult[0].parts[0].inlineData?.mimeType === 'audio/wav', 'Audio → inlineData')

// Video via URL
const videoMsg = [{
  role: 'user',
  content: [{ type: 'video', source: { type: 'url', url: 'gs://bucket/v.mp4', media_type: 'video/mp4' } }],
}] as any[]
const videoResult = toGeminiContents(videoMsg, idMap)
assert(videoResult[0].parts[0].fileData?.fileUri === 'gs://bucket/v.mp4', 'Video URL → fileData')

// System instruction
const sysResult = toGeminiSystemInstruction('You are a helpful assistant')
assert(sysResult!.parts[0].text === 'You are a helpful assistant', 'System instruction → text part')

// Tool use → functionCall
const toolMsg = [{
  role: 'assistant',
  content: [{ type: 'tool_use', id: 'toolu_1', name: 'read', input: { path: '/x' } }],
}] as any[]
const toolIdMap = new ToolCallIdMap()
const toolResult = toGeminiContents(toolMsg, toolIdMap)
assert(toolResult[0].parts[0].functionCall?.name === 'read', 'tool_use → functionCall')

// tool_result → functionResponse
const resultMsg = [{
  role: 'user',
  content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'file content' }],
}] as any[]
const rrResult = toGeminiContents(resultMsg, toolIdMap)
assert(rrResult[0].parts[0].functionResponse?.name === 'read', 'tool_result → functionResponse with correct name')

// ============================================================================
// Summary
// ============================================================================

console.log(`\n${'='.repeat(50)}`)
console.log(`Results: ${passed} passed, ${failed} failed, ${passed + failed} total`)
console.log(`${'='.repeat(50)}`)

if (failed > 0) {
  process.exit(1)
}
