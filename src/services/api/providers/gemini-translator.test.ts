import { describe, it, expect } from 'vitest'
import {
  toGeminiContents,
  toGeminiSystemInstruction,
  mapFinishReason,
  mapUsage,
  ToolCallIdMap,
} from './gemini-translator.js'

describe('toGeminiContents', () => {
  const idMap = new ToolCallIdMap()

  it('converts simple text messages', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result).toHaveLength(2)
    expect(result[0].role).toBe('user')
    expect(result[0].parts[0].text).toBe('Hello')
    expect(result[1].role).toBe('model')
    expect(result[1].parts[0].text).toBe('Hi there!')
  })

  it('converts image blocks with base64', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'image',
        source: { type: 'base64', media_type: 'image/png', data: 'abc123' },
      }],
    }] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result[0].parts[0].inlineData).toEqual({
      mimeType: 'image/png',
      data: 'abc123',
    })
  })

  it('converts image blocks with URL to fileData', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'image',
        source: { type: 'url', url: 'gs://bucket/image.jpg', media_type: 'image/jpeg' },
      }],
    }] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result[0].parts[0].fileData).toEqual({
      mimeType: 'image/jpeg',
      fileUri: 'gs://bucket/image.jpg',
    })
  })

  it('converts document (PDF) blocks', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'document',
        source: { type: 'base64', media_type: 'application/pdf', data: 'pdfdata' },
      }],
    }] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result[0].parts[0].inlineData).toEqual({
      mimeType: 'application/pdf',
      data: 'pdfdata',
    })
  })

  it('converts audio blocks', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'audio',
        source: { type: 'base64', media_type: 'audio/wav', data: 'audiodata' },
      }],
    }] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result[0].parts[0].inlineData).toEqual({
      mimeType: 'audio/wav',
      data: 'audiodata',
    })
  })

  it('converts video blocks with URL', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'video',
        source: { type: 'url', url: 'gs://bucket/video.mp4', media_type: 'video/mp4' },
      }],
    }] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result[0].parts[0].fileData).toEqual({
      mimeType: 'video/mp4',
      fileUri: 'gs://bucket/video.mp4',
    })
  })

  it('converts tool_use and tool_result blocks', () => {
    const toolIdMap = new ToolCallIdMap()
    const messages = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'toolu_1', name: 'read_file', input: { path: '/foo' } }],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'file contents' }],
      },
    ] as any[]

    const result = toGeminiContents(messages, toolIdMap)
    expect(result[0].parts[0].functionCall).toEqual({ name: 'read_file', args: { path: '/foo' } })
    expect(result[1].parts[0].functionResponse).toBeDefined()
    expect(result[1].parts[0].functionResponse.name).toBe('read_file')
  })

  it('converts thinking blocks', () => {
    const messages = [{
      role: 'assistant',
      content: [{ type: 'thinking', thinking: 'Let me analyze...' }],
    }] as any[]

    const result = toGeminiContents(messages, idMap)
    expect(result[0].parts[0].thought).toBe(true)
    expect(result[0].parts[0].text).toBe('Let me analyze...')
  })

  it('merges consecutive same-role messages', () => {
    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'user', content: 'World' },
    ] as any[]

    const result = toGeminiContents(messages, idMap)
    // Should be merged into 1 message with 2 parts
    expect(result).toHaveLength(1)
    expect(result[0].parts).toHaveLength(2)
  })
})

describe('toGeminiSystemInstruction', () => {
  it('converts string system prompt', () => {
    const result = toGeminiSystemInstruction('You are helpful')
    expect(result).toBeDefined()
    expect(result!.parts[0].text).toBe('You are helpful')
  })

  it('converts array system prompt', () => {
    const result = toGeminiSystemInstruction([
      { type: 'text', text: 'Part 1' },
      { type: 'text', text: 'Part 2' },
    ] as any)
    expect(result!.parts[0].text).toContain('Part 1')
    expect(result!.parts[0].text).toContain('Part 2')
  })

  it('returns undefined for empty input', () => {
    expect(toGeminiSystemInstruction(undefined)).toBeUndefined()
    expect(toGeminiSystemInstruction('')).toBeUndefined()
  })
})

describe('mapFinishReason', () => {
  it('maps STOP to end_turn', () => {
    expect(mapFinishReason('STOP', false)).toBe('end_turn')
  })

  it('maps MAX_TOKENS to max_tokens', () => {
    expect(mapFinishReason('MAX_TOKENS', false)).toBe('max_tokens')
  })

  it('returns tool_use when function calls present', () => {
    expect(mapFinishReason('STOP', true)).toBe('tool_use')
    expect(mapFinishReason('MAX_TOKENS', true)).toBe('tool_use')
  })

  it('defaults to end_turn for unknown reasons', () => {
    expect(mapFinishReason('UNKNOWN', false)).toBe('end_turn')
    expect(mapFinishReason(undefined, false)).toBe('end_turn')
  })
})

describe('mapUsage', () => {
  it('maps Gemini usage to Anthropic format', () => {
    const result = mapUsage({
      promptTokenCount: 100,
      candidatesTokenCount: 50,
      cachedContentTokenCount: 20,
    })
    expect(result.input_tokens).toBe(100)
    expect(result.output_tokens).toBe(50)
    expect(result.cache_read_input_tokens).toBe(20)
  })

  it('handles undefined metadata', () => {
    const result = mapUsage(undefined)
    expect(result.input_tokens).toBe(0)
    expect(result.output_tokens).toBe(0)
  })
})

describe('ToolCallIdMap', () => {
  it('records and retrieves id→name mappings', () => {
    const map = new ToolCallIdMap()
    map.record('toolu_abc', 'read_file')
    expect(map.getName('toolu_abc')).toBe('read_file')
  })

  it('returns undefined for unknown ids', () => {
    const map = new ToolCallIdMap()
    expect(map.getName('nonexistent')).toBeUndefined()
  })

  it('clears all mappings', () => {
    const map = new ToolCallIdMap()
    map.record('toolu_1', 'tool_a')
    map.clear()
    expect(map.getName('toolu_1')).toBeUndefined()
  })
})
