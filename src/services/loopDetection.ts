// @ts-nocheck
/**
 * Loop Detection Service
 *
 * Detects infinite loops in the agent's tool-calling behavior.
 * Two detection tiers (borrowed from gemini-cli, Apache 2.0):
 *
 *   Tier 1 — Tool Call Repetition: SHA-256 hash of tool name + arguments.
 *            If the same hash appears TOOL_CALL_THRESHOLD times consecutively,
 *            it's a loop.
 *
 *   Tier 2 — Content Chanting: Sliding window over streamed text.
 *            If identical text chunks appear CONTENT_THRESHOLD times
 *            within MAX_HISTORY_LENGTH characters, it's a loop.
 *
 * Provider-agnostic: works on Anthropic-format messages (the SDK's internal
 * standard), so it detects loops regardless of which LLM provider is in use.
 *
 * Integration: called from query.ts between turns, before autocompact.
 */

import { createHash } from 'crypto'

// ============================================================================
// Constants (from gemini-cli production values)
// ============================================================================

/** Consecutive identical tool calls before triggering */
const TOOL_CALL_THRESHOLD = 5

/** Identical content chunks before triggering */
const CONTENT_THRESHOLD = 10

/** Characters per chunk for content comparison */
const CONTENT_CHUNK_SIZE = 50

/** Maximum content history to retain (chars) */
const MAX_HISTORY_LENGTH = 5000

// ============================================================================
// Types
// ============================================================================

export type LoopType = 'tool_call_repetition' | 'content_chanting'

export interface LoopDetectionResult {
  detected: boolean
  type?: LoopType
  detail?: string
  count: number
}

// ============================================================================
// LoopDetector
// ============================================================================

export class LoopDetector {
  // ── Tier 1: Tool call tracking ──
  private lastToolCallHash: string | null = null
  private toolCallRepeatCount = 0

  // ── Tier 2: Content tracking ──
  private contentHistory = ''
  private chunkCounts = new Map<string, number>()
  private inCodeBlock = false

  // ── State ──
  private loopDetected = false
  private lastLoopType?: LoopType
  private lastLoopDetail?: string
  private detectionCount = 0

  /**
   * Check a tool call for repetition (Tier 1).
   * Call this each time the model requests a tool call.
   */
  checkToolCall(name: string, input: unknown): LoopDetectionResult {
    if (this.loopDetected) {
      return this.currentResult()
    }

    const hash = this.hashToolCall(name, input)

    if (hash === this.lastToolCallHash) {
      this.toolCallRepeatCount++
    } else {
      this.lastToolCallHash = hash
      this.toolCallRepeatCount = 1
      // New tool call breaks content chanting tracking
      this.resetContentTracking()
    }

    if (this.toolCallRepeatCount >= TOOL_CALL_THRESHOLD) {
      this.loopDetected = true
      this.detectionCount++
      this.lastLoopType = 'tool_call_repetition'
      this.lastLoopDetail =
        `Tool "${name}" called ${this.toolCallRepeatCount} times ` +
        `with identical arguments`
      return this.currentResult()
    }

    return { detected: false, count: 0 }
  }

  /**
   * Check streamed text content for repetition (Tier 2).
   * Call this with each text chunk from the model's response.
   */
  checkContent(text: string): LoopDetectionResult {
    if (this.loopDetected) {
      return this.currentResult()
    }

    // Skip detection inside code blocks (repetitive code is normal)
    const fenceCount = (text.match(/```/g) || []).length
    if (fenceCount % 2 !== 0) {
      this.inCodeBlock = !this.inCodeBlock
    }
    if (this.inCodeBlock) {
      return { detected: false, count: 0 }
    }

    // Skip structural markdown elements (tables, lists, headings)
    if (/(^|\n)\s*(\|.*\||[|+-]{3,})/.test(text) ||
        /(^|\n)\s*[*-+]\s/.test(text) ||
        /(^|\n)#+\s/.test(text)) {
      this.resetContentTracking()
      return { detected: false, count: 0 }
    }

    // Append to history, truncate if needed
    this.contentHistory += text
    if (this.contentHistory.length > MAX_HISTORY_LENGTH) {
      this.contentHistory = this.contentHistory.slice(-MAX_HISTORY_LENGTH)
      // Rebuild chunk counts from truncated history
      this.rebuildChunkCounts()
    }

    // Analyze chunks
    const historyLen = this.contentHistory.length
    if (historyLen < CONTENT_CHUNK_SIZE) {
      return { detected: false, count: 0 }
    }

    // Hash the latest chunk
    const latestChunk = this.contentHistory.slice(-CONTENT_CHUNK_SIZE)
    const chunkHash = this.hashString(latestChunk)
    const count = (this.chunkCounts.get(chunkHash) ?? 0) + 1
    this.chunkCounts.set(chunkHash, count)

    if (count >= CONTENT_THRESHOLD) {
      this.loopDetected = true
      this.detectionCount++
      this.lastLoopType = 'content_chanting'
      this.lastLoopDetail =
        `Repeating content detected (${count} occurrences): ` +
        `"${latestChunk.trim().slice(0, 60)}..."`
      return this.currentResult()
    }

    return { detected: false, count: 0 }
  }

  /**
   * Check messages from the last turn for loops.
   * Convenience method that inspects Anthropic-format assistant messages.
   */
  checkAssistantMessage(message: { content: any[] }): LoopDetectionResult {
    if (!message?.content || !Array.isArray(message.content)) {
      return { detected: false, count: 0 }
    }

    for (const block of message.content) {
      if (block.type === 'tool_use') {
        const result = this.checkToolCall(block.name, block.input)
        if (result.detected) return result
      }
      if (block.type === 'text' && block.text) {
        const result = this.checkContent(block.text)
        if (result.detected) return result
      }
    }

    return { detected: false, count: 0 }
  }

  /**
   * Clear the detection flag (after injecting a corrective message).
   */
  clearDetection(): void {
    this.loopDetected = false
    this.lastLoopType = undefined
    this.lastLoopDetail = undefined
    this.toolCallRepeatCount = 0
    this.lastToolCallHash = null
    this.resetContentTracking()
  }

  /**
   * Reset all state (e.g., on new user prompt).
   */
  reset(): void {
    this.clearDetection()
    this.detectionCount = 0
  }

  // ── Private helpers ──

  private currentResult(): LoopDetectionResult {
    return {
      detected: true,
      type: this.lastLoopType,
      detail: this.lastLoopDetail,
      count: this.detectionCount,
    }
  }

  private hashToolCall(name: string, input: unknown): string {
    const raw = `${name}:${JSON.stringify(input ?? {})}`
    return createHash('sha256').update(raw).digest('hex')
  }

  private hashString(s: string): string {
    return createHash('sha256').update(s).digest('hex').slice(0, 16)
  }

  private resetContentTracking(): void {
    this.contentHistory = ''
    this.chunkCounts.clear()
    this.inCodeBlock = false
  }

  private rebuildChunkCounts(): void {
    this.chunkCounts.clear()
    for (let i = 0; i <= this.contentHistory.length - CONTENT_CHUNK_SIZE; i += CONTENT_CHUNK_SIZE) {
      const chunk = this.contentHistory.slice(i, i + CONTENT_CHUNK_SIZE)
      const hash = this.hashString(chunk)
      this.chunkCounts.set(hash, (this.chunkCounts.get(hash) ?? 0) + 1)
    }
  }
}

// ============================================================================
// Feedback message for breaking loops
// ============================================================================

/**
 * Generate a system message to inject when a loop is detected.
 * This tells the model to change its approach.
 */
export function loopBreakMessage(result: LoopDetectionResult): string {
  if (result.type === 'tool_call_repetition') {
    return (
      '[LOOP DETECTED] You have been calling the same tool with identical arguments ' +
      `${result.count > 1 ? 'multiple times' : 'repeatedly'}. ` +
      'This is not making progress. Please: ' +
      '1) Analyze why the previous attempts failed, ' +
      '2) Try a fundamentally different approach, or ' +
      '3) Report the issue to the user and ask for guidance.'
    )
  }

  return (
    '[LOOP DETECTED] Your responses contain repetitive content that is not making progress. ' +
    'Please stop repeating yourself and either: ' +
    '1) Take a different action, ' +
    '2) Provide a clear summary of what you\'ve tried and what\'s blocking progress, or ' +
    '3) Ask the user for help.'
  )
}
