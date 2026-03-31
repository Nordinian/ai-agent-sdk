// @ts-nocheck
/**
 * AnthropicProvider — Wraps existing Anthropic API client as an LLMProvider.
 *
 * This is a thin wrapper: since the SDK's internal format IS Anthropic format,
 * no translation is needed. All existing logic in client.ts / claude.ts / withRetry.ts
 * continues to work unchanged.
 */

import type { LLMProvider, LLMCreateParams, LLMResponse, LLMStreamEvent } from '../provider.js'
import { getAnthropicClient } from '../client.js'

export class AnthropicProvider implements LLMProvider {
  readonly type = 'anthropic' as const

  /**
   * Matches Claude models and acts as the default fallback.
   * Returns false only for model names explicitly claimed by other providers.
   */
  supportsModel(model: string): boolean {
    // Explicit Anthropic/Claude patterns
    if (/^(claude-|anthropic\/)/.test(model)) return true
    // Do NOT claim models that belong to other providers
    if (/^(gemini-|vertex\/|gpt-|o[134]-|deepseek-|mistral-|codestral-|llama-|qwen-|mixtral-|openai-compat\/)/.test(model)) return false
    // Default fallback: claim anything else (backward compatible)
    return true
  }

  /**
   * Non-streaming message creation via existing Anthropic client.
   * Params are already in Anthropic format — pass through directly.
   */
  async createMessage(params: LLMCreateParams): Promise<LLMResponse> {
    const client = await getAnthropicClient({ model: params.model })
    return client.beta.messages.create(params as any) as Promise<LLMResponse>
  }

  /**
   * Streaming message creation via existing Anthropic client.
   * Events are already in Anthropic format — yield directly.
   */
  async *createMessageStream(params: LLMCreateParams): AsyncGenerator<LLMStreamEvent, void, undefined> {
    const client = await getAnthropicClient({ model: params.model })
    const stream = client.beta.messages.stream(params as any)

    for await (const event of stream) {
      yield event as LLMStreamEvent
    }
  }
}
