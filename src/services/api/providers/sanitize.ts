// @ts-nocheck
/**
 * Message Sanitization for Multi-Provider Support
 *
 * Two core functions inspired by real-world edge cases documented in Claude Code's
 * production system (ref: yage.ai/share/claude-code-engineering-cost-20260331):
 *
 *   1. sanitizeEmptyToolResults — Detect empty tool results and inject placeholder
 *      text to prevent models from misinterpreting the prompt tail as a turn boundary.
 *      Claude Code observed ~10% false-stop rates from empty tool results.
 *
 *   2. sanitizeHistoryForProvider — Strip model-specific fields (thinking signatures,
 *      redacted_thinking, connector_text) when switching providers mid-conversation.
 *      Signed blocks are model-bound; replaying them to a different provider causes
 *      400 errors or unpredictable behavior.
 */

import type { BetaMessageParam, BetaContentBlockParam } from '@anthropic-ai/sdk/resources/index.mjs'
import type { LLMProviderType } from '../provider.js'

// ============================================================================
// 1. Empty Tool Result Sanitization
// ============================================================================

/**
 * Check whether a tool_result content block is effectively empty.
 *
 * Tools can legitimately produce empty output:
 *   - Shell commands that succeed silently (mkdir, cp, chmod)
 *   - MCP servers returning content: []
 *   - REPL statements with side effects but no output
 *
 * When sent as-is, some models interpret the empty content at the prompt tail
 * as a turn boundary and emit a stop sequence, producing zero output.
 */
function isToolResultEmpty(content: any): boolean {
  if (content === undefined || content === null) return true
  if (typeof content === 'string') return content.trim() === ''
  if (Array.isArray(content)) {
    if (content.length === 0) return true
    // Array of blocks where all text content is empty
    return content.every(c => {
      if (typeof c === 'string') return c.trim() === ''
      if (c.type === 'text') return (c.text ?? '').trim() === ''
      // Non-text blocks (images, etc.) count as non-empty
      return false
    })
  }
  return false
}

/**
 * Inject placeholder text into empty tool_result blocks.
 *
 * Modifies messages in-place for efficiency (operates on the API-bound copy,
 * not the stored conversation). Returns the same array reference.
 *
 * The placeholder format matches Claude Code's pattern:
 *   "(toolName completed with no output)"
 */
export function sanitizeEmptyToolResults(messages: BetaMessageParam[]): BetaMessageParam[] {
  for (const msg of messages) {
    if (msg.role !== 'user') continue

    const blocks = msg.content
    if (!Array.isArray(blocks)) continue

    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i] as any
      if (block.type !== 'tool_result') continue

      if (isToolResultEmpty(block.content)) {
        // Inject placeholder — keep all other fields (tool_use_id, is_error, etc.)
        blocks[i] = {
          ...block,
          content: `(tool completed with no output)`,
        }
      }
    }
  }

  return messages
}

// ============================================================================
// 2. History Sanitization for Provider Switching
// ============================================================================

/**
 * Block types that carry model-bound signatures or are provider-specific.
 *
 * - thinking: May carry encrypted signatures bound to the generating model's
 *   API key. Replaying to a different provider causes 400 errors.
 * - redacted_thinking: Anthropic-specific; meaningless to other providers and
 *   would waste context window space.
 * - connector_text: Anthropic-specific anti-distillation mechanism with
 *   signed summaries. Invalid across provider boundaries.
 */
const PROVIDER_SPECIFIC_BLOCK_TYPES = new Set([
  'thinking',
  'redacted_thinking',
  'connector_text',
])

/**
 * Strip model-specific content blocks from conversation history when switching
 * between providers.
 *
 * Use cases:
 *   - User switches from Claude to Gemini mid-conversation
 *   - Provider fallback (primary model fails, retry with a different provider)
 *   - Credential rotation that changes the API key (invalidates signatures)
 *
 * The function preserves text and tool_use blocks — these are the universal
 * content that all providers understand.
 *
 * @param messages - Conversation history in Anthropic format
 * @param targetProvider - The provider that will receive these messages
 * @returns Cleaned messages (new array, original is not mutated)
 */
export function sanitizeHistoryForProvider(
  messages: BetaMessageParam[],
  targetProvider: LLMProviderType,
): BetaMessageParam[] {
  // Anthropic provider can handle its own blocks — no sanitization needed
  if (targetProvider === 'anthropic') return messages

  return messages.map(msg => {
    // Only assistant messages contain provider-specific blocks
    if (msg.role !== 'assistant') return msg

    const blocks = msg.content
    if (typeof blocks === 'string') return msg
    if (!Array.isArray(blocks)) return msg

    const cleaned = (blocks as BetaContentBlockParam[]).filter(block => {
      return !PROVIDER_SPECIFIC_BLOCK_TYPES.has(block.type)
    })

    // If all blocks were provider-specific, inject a placeholder to avoid
    // empty assistant messages (which would break alternating role validation)
    if (cleaned.length === 0) {
      return {
        ...msg,
        content: [{ type: 'text', text: '(thinking...)' }],
      }
    }

    // Only create a new object if something was actually filtered
    if (cleaned.length === blocks.length) return msg

    return { ...msg, content: cleaned }
  })
}
