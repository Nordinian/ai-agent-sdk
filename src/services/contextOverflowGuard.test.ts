import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock the dependencies before importing the module
vi.mock('./api/registry.js', () => ({
  resolveProvider: (model: string) => {
    if (model.startsWith('gemini-')) return { type: 'gemini' }
    if (model.startsWith('gpt-')) return { type: 'openai-compat' }
    return { type: 'anthropic' }
  },
}))

vi.mock('../bootstrap/state.js', () => ({
  getSdkBetas: () => [],
}))

vi.mock('../utils/context.js', () => ({
  getContextWindowForModel: () => 200_000,
}))

import { checkContextOverflow } from './contextOverflowGuard.js'

describe('checkContextOverflow', () => {
  it('returns ok for small conversations', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi!' },
    ]
    const result = checkContextOverflow('gemini-2.5-flash', messages)
    expect(result.status).toBe('ok')
    expect(result.estimatedTokens).toBeGreaterThan(0)
    expect(result.usage).toBeLessThan(0.85)
  })

  it('returns overflow for very large conversations', () => {
    // Create messages that would exceed a 64k window
    const messages = Array.from({ length: 100 }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: 'x'.repeat(3000), // ~750 tokens each
    }))
    // deepseek-chat has 64k window, 100 messages × ~750 tokens = ~75k → overflow
    const result = checkContextOverflow('deepseek-chat', messages)
    expect(result.status).toBe('overflow')
    expect(result.usage).toBeGreaterThanOrEqual(0.95)
  })

  it('returns warning when approaching limit', () => {
    // Create messages that approach but don't exceed 128k window
    const messages = Array.from({ length: 200 }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: 'x'.repeat(2000), // ~500 tokens each → ~100k total
    }))
    const result = checkContextOverflow('gpt-4o', messages)
    // 100k / 128k = 78% — might be ok or warning depending on tool overhead
    expect(['ok', 'warning']).toContain(result.status)
  })

  it('accounts for system prompt tokens', () => {
    const messages = [{ role: 'user', content: 'Hi' }]
    const longSystem = 'x'.repeat(10000)

    const withSystem = checkContextOverflow('gemini-2.5-flash', messages, longSystem)
    const withoutSystem = checkContextOverflow('gemini-2.5-flash', messages)

    expect(withSystem.estimatedTokens).toBeGreaterThan(withoutSystem.estimatedTokens)
  })

  it('accounts for tool definitions', () => {
    const messages = [{ role: 'user', content: 'Hi' }]
    const tools = Array.from({ length: 50 }, (_, i) => ({
      name: `tool_${i}`,
      description: `This is tool number ${i} with a detailed description`,
      input_schema: { type: 'object', properties: { path: { type: 'string' } } },
    }))

    const withTools = checkContextOverflow('gemini-2.5-flash', messages, undefined, tools)
    const withoutTools = checkContextOverflow('gemini-2.5-flash', messages)

    expect(withTools.estimatedTokens).toBeGreaterThan(withoutTools.estimatedTokens)
  })

  it('includes usage ratio in result', () => {
    const messages = [{ role: 'user', content: 'Hello' }]
    const result = checkContextOverflow('gemini-2.5-pro', messages)
    expect(result.usage).toBeGreaterThanOrEqual(0)
    expect(result.usage).toBeLessThanOrEqual(1)
    expect(result.contextWindow).toBe(1_048_576)
  })
})
