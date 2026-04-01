// @ts-nocheck
/**
 * Context Overflow Guard
 *
 * Pre-flight check before sending messages to an LLM provider.
 * Estimates total token count and compares against the provider's context window.
 * Returns an actionable result: proceed, warn, or block.
 *
 * Integration point: called in claude.ts queryModelViaProvider() before
 * provider.createMessageStream().
 */

import { resolveProvider } from './api/registry.js'
import {
  estimateMessagesTokens,
  estimateTextTokens,
  getProviderContextWindow,
} from './providerTokenEstimation.js'
import { getContextWindowForModel } from '../utils/context.js'
import { getSdkBetas } from '../bootstrap/state.js'
import type { LLMProviderType } from './api/provider.js'

// ============================================================================
// Types
// ============================================================================

export type OverflowCheckResult =
  | { status: 'ok'; estimatedTokens: number; contextWindow: number; usage: number }
  | { status: 'warning'; estimatedTokens: number; contextWindow: number; usage: number; message: string }
  | { status: 'overflow'; estimatedTokens: number; contextWindow: number; usage: number; message: string }

/** Warn when estimated tokens exceed this fraction of context window */
const WARNING_THRESHOLD = 0.85

/** Block when estimated tokens exceed this fraction of context window */
const OVERFLOW_THRESHOLD = 0.95

// ============================================================================
// Main check
// ============================================================================

/**
 * Check if the request is likely to overflow the provider's context window.
 *
 * @param model - The model name (used to resolve provider and context window)
 * @param messages - Messages in Anthropic internal format
 * @param systemPrompt - System prompt text
 * @param toolDefs - Tool definitions (count towards context)
 * @returns OverflowCheckResult with status and estimates
 */
export function checkContextOverflow(
  model: string,
  messages: Array<{ role: string; content: any }>,
  systemPrompt?: string,
  toolDefs?: Array<{ name: string; description: string; input_schema?: any }>,
): OverflowCheckResult {
  // Resolve provider type for token estimation
  let providerType: LLMProviderType | undefined
  try {
    providerType = resolveProvider(model).type
  } catch {
    providerType = undefined
  }

  // Get context window — try provider-specific first, fall back to Anthropic's
  let contextWindow = getProviderContextWindow(model)
  if (contextWindow === null) {
    try {
      contextWindow = getContextWindowForModel(model, getSdkBetas())
    } catch {
      contextWindow = 200_000 // safe default
    }
  }

  // Estimate tokens
  let estimatedTokens = 0

  // 1. Messages
  estimatedTokens += estimateMessagesTokens(messages, providerType)

  // 2. System prompt
  if (systemPrompt) {
    estimatedTokens += estimateTextTokens(systemPrompt, providerType)
  }

  // 3. Tool definitions (each tool schema consumes tokens)
  if (toolDefs && toolDefs.length > 0) {
    for (const tool of toolDefs) {
      estimatedTokens += estimateTextTokens(tool.name, providerType)
      estimatedTokens += estimateTextTokens(tool.description, providerType)
      if (tool.input_schema) {
        estimatedTokens += estimateTextTokens(
          JSON.stringify(tool.input_schema),
          providerType,
        )
      }
    }
    // Per-tool overhead (~20 tokens each for structure)
    estimatedTokens += toolDefs.length * 20
  }

  const usage = estimatedTokens / contextWindow

  if (usage >= OVERFLOW_THRESHOLD) {
    return {
      status: 'overflow',
      estimatedTokens,
      contextWindow,
      usage,
      message:
        `Context overflow: estimated ${estimatedTokens.toLocaleString()} tokens ` +
        `exceeds ${Math.round(OVERFLOW_THRESHOLD * 100)}% of ${model}'s ` +
        `${contextWindow.toLocaleString()} token context window. ` +
        `Compact or truncate conversation history before continuing.`,
    }
  }

  if (usage >= WARNING_THRESHOLD) {
    return {
      status: 'warning',
      estimatedTokens,
      contextWindow,
      usage,
      message:
        `Context nearing limit: estimated ${estimatedTokens.toLocaleString()} tokens ` +
        `(${Math.round(usage * 100)}% of ${contextWindow.toLocaleString()} window). ` +
        `Consider compacting soon.`,
    }
  }

  return { status: 'ok', estimatedTokens, contextWindow, usage }
}
