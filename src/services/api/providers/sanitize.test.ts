/**
 * Tests for multi-provider message sanitization.
 *
 * Covers edge cases inspired by Claude Code's production learnings:
 *   - Empty tool results causing false stop sequences (~10% rate)
 *   - Provider-specific blocks (thinking signatures) breaking cross-provider replay
 */

import { describe, it, expect } from 'vitest'
import { sanitizeEmptyToolResults, sanitizeHistoryForProvider } from './sanitize.js'

// ============================================================================
// sanitizeEmptyToolResults
// ============================================================================

describe('sanitizeEmptyToolResults', () => {
  it('fills empty string content with placeholder', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_123', content: '' },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)

    const block = messages[0].content[0]
    expect(block.content).toBe('(tool completed with no output)')
    expect(block.tool_use_id).toBe('toolu_123')
  })

  it('fills null/undefined content with placeholder', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: null },
          { type: 'tool_result', tool_use_id: 'toolu_2', content: undefined },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)

    expect(messages[0].content[0].content).toBe('(tool completed with no output)')
    expect(messages[0].content[1].content).toBe('(tool completed with no output)')
  })

  it('fills empty array content with placeholder', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: [] },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].content).toBe('(tool completed with no output)')
  })

  it('fills array of empty text blocks with placeholder', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'toolu_1',
            content: [{ type: 'text', text: '' }, { type: 'text', text: '  ' }],
          },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].content).toBe('(tool completed with no output)')
  })

  it('preserves non-empty tool results', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: 'file created' },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].content).toBe('file created')
  })

  it('preserves tool results with image content', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'toolu_1',
            content: [{ type: 'image', source: { data: 'abc' } }],
          },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    // Image block counts as non-empty — should not be replaced
    expect(messages[0].content[0].content).toEqual([{ type: 'image', source: { data: 'abc' } }])
  })

  it('preserves is_error flag on filled results', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: '', is_error: true },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].is_error).toBe(true)
    expect(messages[0].content[0].content).toBe('(tool completed with no output)')
  })

  it('skips non-user messages', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'hello' }],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].text).toBe('hello')
  })

  it('handles mixed empty and non-empty tool results', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: '' },
          { type: 'tool_result', tool_use_id: 'toolu_2', content: 'output here' },
          { type: 'tool_result', tool_use_id: 'toolu_3', content: null },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].content).toBe('(tool completed with no output)')
    expect(messages[0].content[1].content).toBe('output here')
    expect(messages[0].content[2].content).toBe('(tool completed with no output)')
  })

  it('handles whitespace-only string content', () => {
    const messages: any[] = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: '   \n\t  ' },
        ],
      },
    ]

    sanitizeEmptyToolResults(messages)
    expect(messages[0].content[0].content).toBe('(tool completed with no output)')
  })
})

// ============================================================================
// sanitizeHistoryForProvider
// ============================================================================

describe('sanitizeHistoryForProvider', () => {
  it('passes through unchanged for anthropic target', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'hmm', signature: 'sig123' },
          { type: 'text', text: 'hello' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'anthropic')
    // Should be exact same reference — no cloning
    expect(result).toBe(messages)
  })

  it('strips thinking blocks for gemini target', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'internal reasoning', signature: 'sig_abc' },
          { type: 'text', text: 'hello world' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'gemini')
    expect(result[0].content).toEqual([{ type: 'text', text: 'hello world' }])
  })

  it('strips redacted_thinking blocks for openai-compat target', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'redacted_thinking', data: 'encrypted_blob' },
          { type: 'text', text: 'result' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'openai-compat')
    expect(result[0].content).toEqual([{ type: 'text', text: 'result' }])
  })

  it('strips connector_text blocks', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'connector_text', text: 'summarized...', signature: 'csig_xyz' },
          { type: 'text', text: 'visible output' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'gemini')
    expect(result[0].content).toEqual([{ type: 'text', text: 'visible output' }])
  })

  it('injects placeholder when all blocks are provider-specific', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'deep thought', signature: 'sig' },
          { type: 'redacted_thinking', data: 'blob' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'gemini')
    expect(result[0].content).toEqual([{ type: 'text', text: '(thinking...)' }])
  })

  it('preserves user messages unchanged', () => {
    const messages: any[] = [
      { role: 'user', content: [{ type: 'text', text: 'question' }] },
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'hmm' },
          { type: 'text', text: 'answer' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'gemini')
    expect(result[0]).toBe(messages[0]) // Same reference — not cloned
    expect(result[1].content).toEqual([{ type: 'text', text: 'answer' }])
  })

  it('preserves tool_use blocks in assistant messages', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'let me read the file' },
          { type: 'text', text: 'I will read the file.' },
          { type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: '/foo' } },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'openai-compat')
    expect(result[0].content).toEqual([
      { type: 'text', text: 'I will read the file.' },
      { type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: '/foo' } },
    ])
  })

  it('does not mutate the original messages', () => {
    const original: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'secret' },
          { type: 'text', text: 'visible' },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(original, 'gemini')
    // Original should still have both blocks
    expect(original[0].content).toHaveLength(2)
    // Result should only have the text block
    expect(result[0].content).toHaveLength(1)
  })

  it('handles string content in assistant messages', () => {
    const messages: any[] = [
      { role: 'assistant', content: 'simple string response' },
    ]

    const result = sanitizeHistoryForProvider(messages, 'gemini')
    expect(result[0]).toBe(messages[0]) // Pass through
  })

  it('returns same message reference when nothing needs filtering', () => {
    const messages: any[] = [
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'only text' },
          { type: 'tool_use', id: 'toolu_1', name: 'Bash', input: {} },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'gemini')
    expect(result[0]).toBe(messages[0]) // No cloning needed
  })

  it('handles multi-turn conversation with mixed providers', () => {
    const messages: any[] = [
      { role: 'user', content: 'hello' },
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'Anthropic thinking' },
          { type: 'text', text: 'Hi!' },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: 'done' },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: 'connector_text', text: 'summary', signature: 'csig' },
          { type: 'text', text: 'Continuing...' },
          { type: 'tool_use', id: 'toolu_2', name: 'Write', input: {} },
        ],
      },
    ]

    const result = sanitizeHistoryForProvider(messages, 'openai-compat')

    // User messages unchanged
    expect(result[0]).toBe(messages[0])
    expect(result[2]).toBe(messages[2])
    // Assistant messages: thinking and connector_text stripped
    expect(result[1].content).toEqual([{ type: 'text', text: 'Hi!' }])
    expect(result[3].content).toEqual([
      { type: 'text', text: 'Continuing...' },
      { type: 'tool_use', id: 'toolu_2', name: 'Write', input: {} },
    ])
  })
})
