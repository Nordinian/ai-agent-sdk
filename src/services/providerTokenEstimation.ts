// @ts-nocheck
/**
 * Provider-Aware Token Estimation
 *
 * Heuristic token estimation that adapts to different LLM providers.
 * Each provider has different tokenization characteristics:
 *   - Anthropic: ~4 bytes/token for English, BPE tokenizer
 *   - Gemini:    ~4 chars/token for ASCII, ~0.77 chars/token for CJK
 *   - OpenAI:    ~4 chars/token for English, tiktoken-based
 *
 * Algorithm borrowed from gemini-cli's tokenCalculation.ts (Apache 2.0).
 * Adapted to be provider-agnostic with pluggable constants.
 *
 * Usage:
 *   estimateTokenCount('Hello world', 'gemini')   // → provider-tuned estimate
 *   estimateTokenCount('Hello world')              // → generic fallback (4 bytes/token)
 */

import type { LLMProviderType } from './api/provider.js'

// ============================================================================
// Provider-specific tokenization constants
// ============================================================================

interface TokenEstimationConfig {
  /** Tokens per ASCII character (charCode 0-127) */
  asciiTokensPerChar: number
  /** Tokens per non-ASCII character (CJK, emoji, etc.) */
  nonAsciiTokensPerChar: number
  /** Fixed token estimate for inline images */
  imageTokenEstimate: number
  /** Fixed token estimate for PDF documents (~100 pages) */
  pdfTokenEstimate: number
  /** Max chars before switching to fast approximation */
  maxCharsForFullHeuristic: number
}

const PROVIDER_CONFIGS: Record<string, TokenEstimationConfig> = {
  // Gemini: from gemini-cli tokenCalculation.ts
  // ASCII chars are ~4 chars/token (0.25 tokens/char)
  // CJK/non-ASCII are often 1-2 tokens/char, use 1.3 as conservative estimate
  gemini: {
    asciiTokensPerChar: 0.25,
    nonAsciiTokensPerChar: 1.3,
    imageTokenEstimate: 3000,
    pdfTokenEstimate: 25800,
    maxCharsForFullHeuristic: 100_000,
  },

  // Anthropic: BPE tokenizer, ~4 bytes/token for English
  // Non-ASCII is less efficient, roughly 0.5-1.0 tokens/char
  anthropic: {
    asciiTokensPerChar: 0.25,
    nonAsciiTokensPerChar: 0.8,
    imageTokenEstimate: 1600,   // ~1568 for small images per Anthropic docs
    pdfTokenEstimate: 20000,
    maxCharsForFullHeuristic: 100_000,
  },

  // OpenAI: tiktoken (cl100k_base / o200k_base), similar to Anthropic
  'openai-compat': {
    asciiTokensPerChar: 0.25,
    nonAsciiTokensPerChar: 0.8,
    imageTokenEstimate: 1105,   // low-detail: 85, high-detail: 1105+ tiles
    pdfTokenEstimate: 20000,
    maxCharsForFullHeuristic: 100_000,
  },
}

// Default fallback for unknown providers
const DEFAULT_CONFIG: TokenEstimationConfig = {
  asciiTokensPerChar: 0.25,
  nonAsciiTokensPerChar: 1.0,
  imageTokenEstimate: 2000,
  pdfTokenEstimate: 20000,
  maxCharsForFullHeuristic: 100_000,
}

// ============================================================================
// Core estimation functions
// ============================================================================

/**
 * Estimate token count for a text string using provider-specific heuristics.
 *
 * For strings under maxCharsForFullHeuristic, uses per-character analysis
 * (ASCII vs non-ASCII). For longer strings, uses fast approximation (len/4).
 */
export function estimateTextTokens(
  text: string,
  providerType?: LLMProviderType | string,
): number {
  const config = PROVIDER_CONFIGS[providerType ?? ''] ?? DEFAULT_CONFIG

  if (text.length > config.maxCharsForFullHeuristic) {
    // Fast path: avoid per-char loop on very large strings
    return Math.ceil(text.length * config.asciiTokensPerChar)
  }

  let tokens = 0
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) <= 127) {
      tokens += config.asciiTokensPerChar
    } else {
      tokens += config.nonAsciiTokensPerChar
    }
  }
  return Math.ceil(tokens)
}

/**
 * Estimate tokens for media content (images, PDFs).
 */
export function estimateMediaTokens(
  mimeType: string,
  providerType?: LLMProviderType | string,
): number {
  const config = PROVIDER_CONFIGS[providerType ?? ''] ?? DEFAULT_CONFIG

  if (mimeType.startsWith('image/')) {
    return config.imageTokenEstimate
  }
  if (mimeType === 'application/pdf') {
    return config.pdfTokenEstimate
  }
  // Unknown media type — return a conservative estimate
  return 1000
}

/**
 * Estimate tokens for an Anthropic-format message array.
 * Works across all providers by inspecting content block types.
 *
 * This is the main entry point for the auto-compact system.
 */
export function estimateMessagesTokens(
  messages: Array<{ role: string; content: any }>,
  providerType?: LLMProviderType | string,
): number {
  let total = 0

  for (const msg of messages) {
    // String content
    if (typeof msg.content === 'string') {
      total += estimateTextTokens(msg.content, providerType)
      continue
    }

    // Array of content blocks
    if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        total += estimateBlockTokens(block, providerType)
      }
    }
  }

  // Add per-message overhead (~4 tokens for role/delimiters)
  total += messages.length * 4

  return total
}

/**
 * Estimate tokens for a single content block.
 */
function estimateBlockTokens(
  block: any,
  providerType?: LLMProviderType | string,
): number {
  if (!block || typeof block !== 'object') return 0

  switch (block.type) {
    case 'text':
      return estimateTextTokens(block.text ?? '', providerType)

    case 'thinking':
      return estimateTextTokens(block.thinking ?? '', providerType)

    case 'tool_use':
      // Tool name + JSON-serialized input
      return (
        estimateTextTokens(block.name ?? '', providerType) +
        estimateTextTokens(JSON.stringify(block.input ?? {}), providerType) +
        10 // overhead for tool_use structure
      )

    case 'tool_result': {
      const content = block.content
      if (typeof content === 'string') {
        return estimateTextTokens(content, providerType)
      }
      if (Array.isArray(content)) {
        let sum = 0
        for (const inner of content) {
          sum += estimateBlockTokens(inner, providerType)
        }
        return sum
      }
      return 0
    }

    case 'image': {
      const mimeType = block.source?.media_type ?? 'image/unknown'
      return estimateMediaTokens(mimeType, providerType)
    }

    case 'document': {
      const mimeType = block.source?.media_type ?? 'application/pdf'
      return estimateMediaTokens(mimeType, providerType)
    }

    default:
      // Unknown block — serialize and estimate
      return estimateTextTokens(JSON.stringify(block), providerType)
  }
}

// ============================================================================
// Context window registry
// ============================================================================

/**
 * Known context window sizes for non-Anthropic models.
 * Anthropic models use the existing getContextWindowForModel() from utils/context.ts.
 */
const CONTEXT_WINDOW_SIZES: Record<string, number> = {
  // Gemini models
  'gemini-2.5-pro': 1_048_576,
  'gemini-2.5-flash': 1_048_576,
  'gemini-2.5-flash-lite': 262_144,
  'gemini-2.0-flash': 1_048_576,
  'gemini-2.0-flash-lite': 262_144,
  'gemini-3.1-pro-preview': 1_048_576,
  'gemini-3-pro-preview': 1_048_576,
  'gemini-3-flash-preview': 1_048_576,

  // OpenAI models
  'gpt-4o': 128_000,
  'gpt-4o-mini': 128_000,
  'gpt-4-turbo': 128_000,
  'gpt-5.4': 128_000,
  'o1-preview': 128_000,
  'o1-mini': 128_000,
  'o3-mini': 200_000,
  'o4-mini': 200_000,

  // DeepSeek
  'deepseek-chat': 64_000,
  'deepseek-coder': 64_000,
  'deepseek-reasoner': 64_000,

  // Mistral
  'mistral-large-latest': 128_000,
  'mistral-small-latest': 128_000,
  'codestral-latest': 256_000,
}

/**
 * Get context window size for any model across all providers.
 * Returns null if the model is unknown (caller should use a safe default).
 */
export function getProviderContextWindow(model: string): number | null {
  // Strip prefixes like 'vertex/' or 'openai-compat/'
  const normalized = model.replace(/^(vertex|openai-compat)\//, '')

  // Exact match
  if (CONTEXT_WINDOW_SIZES[normalized]) {
    return CONTEXT_WINDOW_SIZES[normalized]
  }

  // Prefix match (e.g., 'llama-3.1-70b' → generic Llama limit)
  if (/^llama-/.test(normalized)) return 128_000
  if (/^qwen-/.test(normalized)) return 128_000
  if (/^mixtral-/.test(normalized)) return 32_000

  return null
}
