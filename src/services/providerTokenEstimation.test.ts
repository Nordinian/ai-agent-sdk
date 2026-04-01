import { describe, it, expect } from 'vitest'
import {
  estimateTextTokens,
  estimateMessagesTokens,
  estimateMediaTokens,
  getProviderContextWindow,
} from './providerTokenEstimation.js'

describe('estimateTextTokens', () => {
  it('estimates ASCII text consistently across providers', () => {
    const text = 'Hello, world!'
    // All providers use ~0.25 tokens/char for ASCII
    const gemini = estimateTextTokens(text, 'gemini')
    const anthropic = estimateTextTokens(text, 'anthropic')
    const openai = estimateTextTokens(text, 'openai-compat')

    expect(gemini).toBeGreaterThan(0)
    expect(anthropic).toBeGreaterThan(0)
    expect(openai).toBeGreaterThan(0)
    // All should be roughly similar for pure ASCII
    expect(Math.abs(gemini - anthropic)).toBeLessThan(5)
  })

  it('estimates CJK text higher for Gemini (1.3 vs 0.8)', () => {
    const cjk = '这是一段中文测试文本用于验证多语言支持'
    const gemini = estimateTextTokens(cjk, 'gemini')
    const anthropic = estimateTextTokens(cjk, 'anthropic')

    // Gemini uses 1.3 tokens/char for non-ASCII, Anthropic uses 0.8
    expect(gemini).toBeGreaterThan(anthropic)
    expect(gemini / anthropic).toBeGreaterThan(1.3)
  })

  it('uses fast path for very long strings (>100k chars)', () => {
    const longText = 'a'.repeat(150_000)
    const result = estimateTextTokens(longText, 'gemini')
    // Fast path: length * 0.25
    expect(result).toBe(Math.ceil(150_000 * 0.25))
  })

  it('returns 0 for empty string', () => {
    expect(estimateTextTokens('', 'gemini')).toBe(0)
  })

  it('falls back to default config for unknown provider', () => {
    const result = estimateTextTokens('test', 'unknown-provider')
    expect(result).toBeGreaterThan(0)
  })
})

describe('estimateMediaTokens', () => {
  it('returns image estimates per provider', () => {
    const gemini = estimateMediaTokens('image/png', 'gemini')
    const anthropic = estimateMediaTokens('image/png', 'anthropic')

    expect(gemini).toBe(3000)      // Gemini's fixed estimate
    expect(anthropic).toBe(1600)   // Anthropic's fixed estimate
  })

  it('returns PDF estimates', () => {
    const gemini = estimateMediaTokens('application/pdf', 'gemini')
    expect(gemini).toBe(25800)     // ~100 pages at 258 tokens/page
  })

  it('returns conservative estimate for unknown media', () => {
    const result = estimateMediaTokens('video/mp4', 'gemini')
    expect(result).toBe(1000)
  })
})

describe('estimateMessagesTokens', () => {
  it('estimates simple text messages', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ]
    const result = estimateMessagesTokens(messages, 'gemini')
    expect(result).toBeGreaterThan(0)
    // Should include per-message overhead (4 tokens each)
    expect(result).toBeGreaterThan(8)
  })

  it('handles array content blocks', () => {
    const messages = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Explain this code' },
          { type: 'text', text: 'function foo() {}' },
        ],
      },
    ]
    const result = estimateMessagesTokens(messages, 'anthropic')
    expect(result).toBeGreaterThan(0)
  })

  it('handles tool_use and tool_result blocks', () => {
    const messages = [
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'toolu_123',
            name: 'read_file',
            input: { path: '/foo/bar.ts' },
          },
        ],
      },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'toolu_123',
            content: 'file contents here',
          },
        ],
      },
    ]
    const result = estimateMessagesTokens(messages, 'gemini')
    expect(result).toBeGreaterThan(0)
  })

  it('handles thinking blocks', () => {
    const messages = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'Let me think about this...' },
          { type: 'text', text: 'Here is my answer.' },
        ],
      },
    ]
    const result = estimateMessagesTokens(messages)
    expect(result).toBeGreaterThan(0)
  })

  it('returns 0 for empty messages', () => {
    expect(estimateMessagesTokens([])).toBe(0)
  })
})

describe('getProviderContextWindow', () => {
  it('returns correct window for Gemini models', () => {
    expect(getProviderContextWindow('gemini-2.5-pro')).toBe(1_048_576)
    expect(getProviderContextWindow('gemini-2.5-flash-lite')).toBe(262_144)
  })

  it('returns correct window for OpenAI models', () => {
    expect(getProviderContextWindow('gpt-4o')).toBe(128_000)
  })

  it('returns correct window for DeepSeek models', () => {
    expect(getProviderContextWindow('deepseek-chat')).toBe(64_000)
  })

  it('strips vertex/ prefix', () => {
    expect(getProviderContextWindow('vertex/gemini-2.5-pro')).toBe(1_048_576)
  })

  it('returns null for unknown models', () => {
    expect(getProviderContextWindow('unknown-model-xyz')).toBeNull()
  })

  it('handles prefix-based matching for llama/qwen', () => {
    expect(getProviderContextWindow('llama-3.1-70b')).toBe(128_000)
    expect(getProviderContextWindow('qwen-72b')).toBe(128_000)
  })
})
