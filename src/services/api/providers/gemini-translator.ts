// @ts-nocheck
/**
 * Gemini Message Translator
 *
 * Bidirectional translation between Anthropic message format (SDK internal)
 * and Google Gemini API format (@google/genai).
 *
 * Key differences handled:
 *   - role: 'assistant' ↔ 'model'
 *   - tool_use (id-based) ↔ functionCall (name-based)
 *   - tool_result (tool_use_id) ↔ functionResponse (name)
 *   - thinking ↔ thought
 *   - system blocks ↔ systemInstruction
 *   - usage field names
 *   - stop_reason / finishReason mapping
 */

import type {
  BetaMessageParam,
  BetaContentBlockParam,
} from '@anthropic-ai/sdk/resources/index.mjs'
import type { SystemPrompt } from '../../../utils/systemPromptType.js'

// ============================================================================
// Types — lightweight Gemini types to avoid hard dependency at import time.
// These match @google/genai's Content/Part/GenerateContentResponse shapes.
// ============================================================================

export interface GeminiContent {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

export interface GeminiPart {
  text?: string
  thought?: boolean
  inlineData?: { mimeType: string; data: string }
  functionCall?: { name: string; args: Record<string, unknown> }
  functionResponse?: { name: string; response: Record<string, unknown> }
}

export interface GeminiUsageMetadata {
  promptTokenCount?: number
  candidatesTokenCount?: number
  cachedContentTokenCount?: number
  totalTokenCount?: number
}

export interface GeminiCandidate {
  content?: { parts?: GeminiPart[] }
  finishReason?: string
}

export interface GeminiGenerateContentResponse {
  candidates?: GeminiCandidate[]
  usageMetadata?: GeminiUsageMetadata
  modelVersion?: string
}

// ============================================================================
// Anthropic → Gemini translation
// ============================================================================

/**
 * Maintains a mapping of tool_use_id → tool name across the conversation.
 * Needed because Anthropic's tool_result references by id, but Gemini's
 * functionResponse references by name.
 */
export class ToolCallIdMap {
  private idToName = new Map<string, string>()

  record(id: string, name: string): void {
    this.idToName.set(id, name)
  }

  getName(id: string): string | undefined {
    return this.idToName.get(id)
  }

  clear(): void {
    this.idToName.clear()
  }
}

/**
 * Convert an array of Anthropic messages to Gemini Content[].
 */
export function toGeminiContents(
  messages: BetaMessageParam[],
  toolCallIdMap: ToolCallIdMap,
): GeminiContent[] {
  const result: GeminiContent[] = []

  for (const msg of messages) {
    const role = msg.role === 'assistant' ? 'model' : 'user'
    const parts: GeminiPart[] = []

    // content can be a string or an array of content blocks
    const blocks = typeof msg.content === 'string'
      ? [{ type: 'text' as const, text: msg.content }]
      : (msg.content as BetaContentBlockParam[])

    for (const block of blocks) {
      const converted = toGeminiParts(block, toolCallIdMap)
      parts.push(...converted)
    }

    if (parts.length > 0) {
      result.push({ role, parts })
    }
  }

  return mergeConsecutiveSameRole(result)
}

/**
 * Convert a single Anthropic content block to Gemini Part(s).
 */
function toGeminiParts(
  block: BetaContentBlockParam | { type: string; [key: string]: any },
  toolCallIdMap: ToolCallIdMap,
): GeminiPart[] {
  switch (block.type) {
    case 'text':
      return [{ text: block.text }]

    case 'thinking':
      return [{ thought: true, text: block.thinking }]

    case 'tool_use':
      // Record id→name mapping for later tool_result translation
      toolCallIdMap.record(block.id, block.name)
      return [{
        functionCall: {
          name: block.name,
          args: block.input as Record<string, unknown>,
        },
      }]

    case 'tool_result': {
      const name = toolCallIdMap.getName(block.tool_use_id)
      if (!name) {
        // Fallback: use tool_use_id as name (shouldn't happen in practice)
        return [{
          functionResponse: {
            name: block.tool_use_id,
            response: { result: extractToolResultContent(block.content) },
          },
        }]
      }
      return [{
        functionResponse: {
          name,
          response: {
            result: extractToolResultContent(block.content),
            ...(block.is_error ? { error: true } : {}),
          },
        },
      }]
    }

    case 'image': {
      const source = block.source
      if (source?.type === 'base64') {
        return [{
          inlineData: {
            mimeType: source.media_type,
            data: source.data,
          },
        }]
      }
      // URL images: convert to text description as fallback
      return [{ text: `[Image: ${source?.url || 'unknown'}]` }]
    }

    default:
      // Unknown block types: serialize as text
      return [{ text: JSON.stringify(block) }]
  }
}

/**
 * Extract text content from a tool_result's content field.
 * Content can be a string or an array of content blocks.
 */
function extractToolResultContent(content: any): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map(c => {
        if (typeof c === 'string') return c
        if (c.type === 'text') return c.text
        return JSON.stringify(c)
      })
      .join('\n')
  }
  return String(content ?? '')
}

/**
 * Gemini requires alternating user/model roles.
 * Merge consecutive messages with the same role.
 */
function mergeConsecutiveSameRole(contents: GeminiContent[]): GeminiContent[] {
  if (contents.length === 0) return contents

  const merged: GeminiContent[] = [contents[0]!]

  for (let i = 1; i < contents.length; i++) {
    const current = contents[i]!
    const prev = merged[merged.length - 1]!
    if (current.role === prev.role) {
      prev.parts.push(...current.parts)
    } else {
      merged.push(current)
    }
  }

  return merged
}

/**
 * Convert Anthropic system prompt to Gemini systemInstruction.
 * Anthropic supports multi-segment system prompts; Gemini uses a single Content.
 */
export function toGeminiSystemInstruction(
  system: string | SystemPrompt[] | undefined,
): GeminiContent | undefined {
  if (!system) return undefined

  let text: string
  if (typeof system === 'string') {
    text = system
  } else if (Array.isArray(system)) {
    text = system
      .map(block => {
        if (typeof block === 'string') return block
        if (block.type === 'text') return block.text
        return ''
      })
      .filter(Boolean)
      .join('\n\n')
  } else {
    return undefined
  }

  if (!text) return undefined

  return {
    role: 'user',
    parts: [{ text }],
  }
}

// ============================================================================
// Gemini → Anthropic translation
// ============================================================================

/**
 * Map Gemini finishReason to Anthropic stop_reason.
 *
 * Critical: when functionCall parts are present, stop_reason MUST be 'tool_use'
 * — this is the signal that makes the engine enter the tool execution branch.
 */
export function mapFinishReason(
  finishReason: string | undefined,
  hasFunctionCalls: boolean,
): string {
  if (hasFunctionCalls) return 'tool_use'

  switch (finishReason) {
    case 'STOP':
      return 'end_turn'
    case 'MAX_TOKENS':
      return 'max_tokens'
    case 'SAFETY':
      return 'end_turn'
    case 'RECITATION':
      return 'end_turn'
    default:
      return 'end_turn'
  }
}

/**
 * Map Gemini usage metadata to Anthropic usage format.
 */
export function mapUsage(metadata: GeminiUsageMetadata | undefined): {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens?: number
  cache_read_input_tokens?: number
} {
  return {
    input_tokens: metadata?.promptTokenCount ?? 0,
    output_tokens: metadata?.candidatesTokenCount ?? 0,
    ...(metadata?.cachedContentTokenCount
      ? { cache_read_input_tokens: metadata.cachedContentTokenCount }
      : {}),
  }
}

/**
 * Generate an Anthropic-style message ID.
 */
export function generateMessageId(): string {
  return `msg_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`
}

/**
 * Generate an Anthropic-style tool_use ID.
 */
export function generateToolUseId(): string {
  return `toolu_${crypto.randomUUID().replace(/-/g, '').slice(0, 20)}`
}
