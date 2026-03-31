// @ts-nocheck
/**
 * LLMProvider — Abstract interface for LLM providers.
 *
 * All inputs and outputs use Anthropic message format (the SDK's internal standard).
 * Each provider is responsible for translating to/from its native format internally.
 */

import type {
  BetaMessageParam,
  BetaToolUnion,
  BetaMessage,
  BetaRawMessageStreamEvent,
} from '@anthropic-ai/sdk/resources/index.mjs'
import type { SystemPrompt } from '../../utils/systemPromptType.js'

export type LLMProviderType = 'anthropic' | 'gemini' | 'openai-compat'

/**
 * Parameters for creating a message, in Anthropic format.
 * This is the SDK's internal standard — providers translate as needed.
 */
export interface LLMCreateParams {
  model: string
  messages: BetaMessageParam[]
  system?: string | SystemPrompt[]
  tools?: BetaToolUnion[]
  tool_choice?: any
  max_tokens: number
  temperature?: number
  thinking?: { type: 'enabled'; budget_tokens: number }
  metadata?: Record<string, unknown>
  betas?: string[]
  [key: string]: unknown
}

/**
 * Streaming event in Anthropic format.
 * Non-Anthropic providers wrap their native stream into this format.
 */
export type LLMStreamEvent = BetaRawMessageStreamEvent

/**
 * Complete response in Anthropic format.
 */
export type LLMResponse = BetaMessage

/**
 * Abstract LLM provider interface.
 *
 * Providers receive Anthropic-formatted requests and return Anthropic-formatted
 * responses. Translation to/from the native provider format happens inside
 * each provider implementation.
 */
export interface LLMProvider {
  /** Provider identifier */
  readonly type: LLMProviderType

  /** Check whether this provider can handle the given model name */
  supportsModel(model: string): boolean

  /** Send a message and return a complete response (non-streaming) */
  createMessage(params: LLMCreateParams): Promise<LLMResponse>

  /** Send a message and return a streaming event generator */
  createMessageStream(params: LLMCreateParams): AsyncGenerator<LLMStreamEvent, void, undefined>

  /** Optional: provider-specific initialization (e.g. connect MCP, warm up) */
  initialize?(): Promise<void>

  /** Optional: provider-specific cleanup */
  shutdown?(): Promise<void>
}
