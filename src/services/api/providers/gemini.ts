// @ts-nocheck
/**
 * GeminiProvider — Native Google Gemini LLM provider.
 *
 * Uses @google/genai SDK to connect to Gemini API (API key) or Vertex AI
 * (service account). Translates between Anthropic message format (SDK internal)
 * and Gemini's native format.
 *
 * Reference implementations:
 *   - gemini-cli: /packages/core/src/core/contentGenerator.ts
 *   - adk-python: /src/google/adk/models/google_llm.py
 */

import type { LLMProvider, LLMCreateParams, LLMResponse, LLMStreamEvent } from '../provider.js'
import {
  toGeminiContents,
  toGeminiSystemInstruction,
  mapFinishReason,
  mapUsage,
  generateMessageId,
  generateToolUseId,
  ToolCallIdMap,
  type GeminiPart,
} from './gemini-translator.js'
import { toGeminiFunctionDeclarations } from './gemini-tool-translator.js'

// ============================================================================
// Safety settings — Agent mode requires the most permissive settings.
// Code operations and shell commands would otherwise be blocked.
// (Same approach as gemini-cli)
// ============================================================================

const AGENT_SAFETY_SETTINGS = [
  { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
  { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' },
]

// ============================================================================
// Model aliases — shorthand names resolved to concrete model IDs.
// Follows gemini-cli's alias pattern (models.ts).
// ============================================================================

const MODEL_ALIASES: Record<string, string> = {
  'gemini-auto':  'gemini-2.5-pro',
  'gemini-pro':   'gemini-2.5-pro',
  'gemini-flash': 'gemini-2.5-flash',
  'gemini-lite':  'gemini-2.5-flash-lite',
}

/**
 * Fallback chain for automatic model downgrade on 429/quota errors.
 * Each model maps to the next cheaper model to try.
 */
const FALLBACK_CHAIN: Record<string, string> = {
  'gemini-2.5-pro':        'gemini-2.5-flash',
  'gemini-2.5-flash':      'gemini-2.5-flash-lite',
  'gemini-3.1-pro-preview': 'gemini-2.5-pro',
  'gemini-3-pro-preview':  'gemini-2.5-pro',
  'gemini-3-flash-preview': 'gemini-2.5-flash',
}

// ============================================================================
// Gemini model context window limits
// ============================================================================

const GEMINI_MODEL_LIMITS: Record<string, number> = {
  'gemini-2.5-pro': 1_048_576,
  'gemini-2.5-flash': 1_048_576,
  'gemini-2.5-flash-lite': 262_144,
  'gemini-2.0-flash': 1_048_576,
  'gemini-2.0-flash-lite': 262_144,
  'gemini-3.1-pro-preview': 1_048_576,
  'gemini-3-pro-preview': 1_048_576,
  'gemini-3-flash-preview': 1_048_576,
}

// ============================================================================
// GeminiProvider
// ============================================================================

// ============================================================================
// Context caching — auto-cache long system prompts to reduce cost.
// Gemini charges ~4x less for cached tokens vs. uncached.
// Minimum content size for caching: 4096 tokens (~16K chars).
// ============================================================================

const CACHE_MIN_CHARS = 16_000  // ~4096 tokens at 4 chars/token
const CACHE_TTL_SECONDS = 3600  // 1 hour default TTL

export class GeminiProvider implements LLMProvider {
  readonly type = 'gemini' as const
  private clientInstance: any = null
  private toolCallIdMap = new ToolCallIdMap()

  // Context caching state
  private cachedContentName: string | null = null
  private cachedSystemHash: string | null = null

  supportsModel(model: string): boolean {
    // Exact alias match or prefix match
    if (MODEL_ALIASES[model]) return true
    return /^(gemini-|vertex\/)/.test(model)
  }

  /**
   * Lazily create the @google/genai client.
   * Supports both Gemini API (API key) and Vertex AI (service account).
   */
  private async getClient(): Promise<any> {
    if (this.clientInstance) return this.clientInstance

    // Dynamic import to avoid hard dependency when Gemini is not used
    const { GoogleGenAI } = await import('@google/genai')

    const isVertexAI = process.env.GOOGLE_VERTEX_AI === 'true'
      || !!process.env.GOOGLE_CLOUD_PROJECT

    this.clientInstance = new GoogleGenAI({
      apiKey: isVertexAI ? undefined : (process.env.GEMINI_API_KEY || undefined),
      vertexai: isVertexAI
        ? {
            project: process.env.GOOGLE_CLOUD_PROJECT,
            location: process.env.GOOGLE_CLOUD_LOCATION || 'us-central1',
          }
        : undefined,
    })

    return this.clientInstance
  }

  /**
   * Resolve the actual model name for the API.
   * 1. Resolve aliases (gemini-auto → gemini-2.5-pro)
   * 2. Strip prefixes (vertex/gemini-2.5-pro → gemini-2.5-pro)
   */
  private resolveModelName(model: string): string {
    const stripped = model.replace(/^vertex\//, '')
    return MODEL_ALIASES[stripped] ?? stripped
  }

  /**
   * Get the fallback model for automatic downgrade on 429 errors.
   * Returns null if no fallback is available.
   */
  private getFallbackModel(model: string): string | null {
    return FALLBACK_CHAIN[model] ?? null
  }

  /**
   * Attempt to auto-cache the system prompt if it's large enough.
   * Returns the cached content resource name, or null if caching isn't applicable.
   */
  private async tryAutoCache(
    model: string,
    systemText: string | undefined,
  ): Promise<string | null> {
    if (!systemText || systemText.length < CACHE_MIN_CHARS) return null

    // Check if we already have a valid cache for this exact system prompt
    const { createHash } = await import('crypto')
    const hash = createHash('sha256').update(systemText).digest('hex').slice(0, 16)

    if (this.cachedContentName && this.cachedSystemHash === hash) {
      return this.cachedContentName
    }

    try {
      const client = await this.getClient()
      const result = await client.caches.create({
        model,
        config: {
          contents: [{ role: 'user', parts: [{ text: systemText }] }],
          ttl: `${CACHE_TTL_SECONDS}s`,
          displayName: `open-agent-sdk-${hash}`,
        },
      })
      this.cachedContentName = result.name
      this.cachedSystemHash = hash
      return result.name
    } catch {
      // Caching not available (free tier, unsupported model) — proceed without
      return null
    }
  }

  /**
   * Build Gemini API config from LLMCreateParams.
   * Centralizes config construction for both streaming and non-streaming calls.
   * Supports Gemini-specific features via extra params:
   *   - geminiGrounding: boolean — enable Google Search grounding
   *   - geminiCachedContent: string — cached content resource name (manual)
   *   - geminiAutoCache: boolean — auto-cache system prompts (default: false)
   *   - geminiThinkingBudget: number — override thinking token budget
   *   - geminiResponseMimeType: string — e.g. 'application/json' for structured output
   *   - geminiResponseSchema: object — JSON schema for structured output
   *   - geminiCodeExecution: boolean — enable native code execution
   */
  private buildConfig(params: LLMCreateParams, tools: any[]) {
    const systemInstruction = toGeminiSystemInstruction(params.system)

    // Thinking config: explicit geminiThinkingBudget > Anthropic thinking > default
    const thinkingBudget = (params as any).geminiThinkingBudget
      ?? params.thinking?.budget_tokens
      ?? undefined

    // Google Search grounding
    const googleSearch = (params as any).geminiGrounding
      ? [{ googleSearch: {} }]
      : undefined

    // Native code execution (Gemini runs Python in sandbox)
    const codeExecution = (params as any).geminiCodeExecution
      ? [{ codeExecution: {} }]
      : undefined

    // Build tools array: function declarations + optional google search + code execution
    const toolsConfig: any[] = []
    if (tools.length > 0) {
      toolsConfig.push({ functionDeclarations: tools })
    }
    if (googleSearch) {
      toolsConfig.push(...googleSearch)
    }
    if (codeExecution) {
      toolsConfig.push(...codeExecution)
    }

    return {
      systemInstruction,
      tools: toolsConfig.length > 0 ? toolsConfig : undefined,
      maxOutputTokens: params.max_tokens,
      temperature: params.temperature,
      safetySettings: AGENT_SAFETY_SETTINGS,
      ...(thinkingBudget ? { thinkingConfig: { thinkingBudget } } : {}),
      ...((params as any).geminiCachedContent
        ? { cachedContent: (params as any).geminiCachedContent }
        : {}),
      ...((params as any).geminiResponseMimeType
        ? { responseMimeType: (params as any).geminiResponseMimeType }
        : {}),
      ...((params as any).geminiResponseSchema
        ? { responseSchema: (params as any).geminiResponseSchema }
        : {}),
    }
  }

  /**
   * Non-streaming message creation.
   * Translates request/response between Anthropic and Gemini formats.
   */
  async createMessage(params: LLMCreateParams): Promise<LLMResponse> {
    const client = await this.getClient()
    const model = this.resolveModelName(params.model)

    // Auto-cache system prompt if enabled and not already manually cached
    if ((params as any).geminiAutoCache && !(params as any).geminiCachedContent) {
      const systemText = typeof params.system === 'string' ? params.system
        : Array.isArray(params.system) ? params.system.map((b: any) => b.text || b).join('\n\n')
        : undefined
      const cachedName = await this.tryAutoCache(model, systemText)
      if (cachedName) {
        ;(params as any).geminiCachedContent = cachedName
      }
    }

    const contents = toGeminiContents(params.messages, this.toolCallIdMap)
    const tools = toGeminiFunctionDeclarations(params.tools ?? [])
    const config = this.buildConfig(params, tools)

    const response = await client.models.generateContent({
      model,
      contents,
      config,
    })

    return this.translateCompleteResponse(response, model)
  }

  /**
   * Streaming message creation.
   * Wraps Gemini's chunk stream into Anthropic's fine-grained event protocol.
   */
  async *createMessageStream(params: LLMCreateParams): AsyncGenerator<LLMStreamEvent, void, undefined> {
    const client = await this.getClient()
    const model = this.resolveModelName(params.model)

    // Auto-cache system prompt if enabled
    if ((params as any).geminiAutoCache && !(params as any).geminiCachedContent) {
      const systemText = typeof params.system === 'string' ? params.system
        : Array.isArray(params.system) ? params.system.map((b: any) => b.text || b).join('\n\n')
        : undefined
      const cachedName = await this.tryAutoCache(model, systemText)
      if (cachedName) {
        ;(params as any).geminiCachedContent = cachedName
      }
    }

    const contents = toGeminiContents(params.messages, this.toolCallIdMap)
    const tools = toGeminiFunctionDeclarations(params.tools ?? [])
    const config = this.buildConfig(params, tools)

    let stream: AsyncIterable<any>
    let activeModel = model
    try {
      const result = await client.models.generateContentStream({
        model: activeModel,
        contents,
        config,
      })
      stream = result
    } catch (error: any) {
      // Auto-fallback on 429 (quota exhausted) — try cheaper model
      const status = error.status || error.code || error.httpStatusCode
      if (status === 429 || status === 'RESOURCE_EXHAUSTED') {
        const fallback = this.getFallbackModel(activeModel)
        if (fallback) {
          try {
            activeModel = fallback
            const result = await client.models.generateContentStream({
              model: activeModel,
              contents,
              config,
            })
            stream = result
          } catch (fallbackError: any) {
            throw this.toAnthropicError(fallbackError)
          }
        } else {
          throw this.toAnthropicError(error)
        }
      } else {
        throw this.toAnthropicError(error)
      }
    }

    // ── Translate Gemini chunks → Anthropic stream events ──

    const messageId = generateMessageId()
    let contentIndex = 0
    let hasEmittedStart = false
    let totalUsage = { input_tokens: 0, output_tokens: 0 }
    let functionCallCount = 0

    try {
      for await (const chunk of stream) {
        // 1. Emit message_start on first chunk
        if (!hasEmittedStart) {
          yield {
            type: 'message_start',
            message: {
              id: messageId,
              type: 'message',
              role: 'assistant',
              model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            },
          } as any
          hasEmittedStart = true
        }

        // 2. Process each part in this chunk
        const parts: GeminiPart[] = chunk.candidates?.[0]?.content?.parts ?? []

        for (const part of parts) {
          // ── Text content ──
          if (part.text !== undefined && !part.thought) {
            yield {
              type: 'content_block_start',
              index: contentIndex,
              content_block: { type: 'text', text: '' },
            } as any
            yield {
              type: 'content_block_delta',
              index: contentIndex,
              delta: { type: 'text_delta', text: part.text },
            } as any
            yield {
              type: 'content_block_stop',
              index: contentIndex,
            } as any
            contentIndex++
          }

          // ── Thinking content ──
          if (part.thought && part.text) {
            yield {
              type: 'content_block_start',
              index: contentIndex,
              content_block: { type: 'thinking', thinking: '' },
            } as any
            yield {
              type: 'content_block_delta',
              index: contentIndex,
              delta: { type: 'thinking_delta', thinking: part.text },
            } as any
            yield {
              type: 'content_block_stop',
              index: contentIndex,
            } as any
            contentIndex++
          }

          // ── Executable code (Gemini code execution) ──
          if (part.executableCode) {
            yield {
              type: 'content_block_start',
              index: contentIndex,
              content_block: { type: 'text', text: '' },
            } as any
            yield {
              type: 'content_block_delta',
              index: contentIndex,
              delta: { type: 'text_delta', text: `\n\`\`\`${part.executableCode.language || 'python'}\n${part.executableCode.code}\n\`\`\`\n` },
            } as any
            yield { type: 'content_block_stop', index: contentIndex } as any
            contentIndex++
          }

          // ── Code execution result ──
          if (part.codeExecutionResult) {
            yield {
              type: 'content_block_start',
              index: contentIndex,
              content_block: { type: 'text', text: '' },
            } as any
            const outcome = part.codeExecutionResult.outcome === 'OUTCOME_OK' ? '' : `[${part.codeExecutionResult.outcome}] `
            yield {
              type: 'content_block_delta',
              index: contentIndex,
              delta: { type: 'text_delta', text: `\n**Output:**\n\`\`\`\n${outcome}${part.codeExecutionResult.output}\n\`\`\`\n` },
            } as any
            yield { type: 'content_block_stop', index: contentIndex } as any
            contentIndex++
          }

          // ── Function call ──
          if (part.functionCall) {
            functionCallCount++
            const toolUseId = generateToolUseId()

            // Record in the id↔name map for future tool_result translation
            this.toolCallIdMap.record(toolUseId, part.functionCall.name)

            yield {
              type: 'content_block_start',
              index: contentIndex,
              content_block: {
                type: 'tool_use',
                id: toolUseId,
                name: part.functionCall.name,
                input: {},
              },
            } as any
            yield {
              type: 'content_block_delta',
              index: contentIndex,
              delta: {
                type: 'input_json_delta',
                partial_json: JSON.stringify(part.functionCall.args ?? {}),
              },
            } as any
            yield {
              type: 'content_block_stop',
              index: contentIndex,
            } as any
            contentIndex++
          }
        }

        // 3. Track usage
        if (chunk.usageMetadata) {
          totalUsage = mapUsage(chunk.usageMetadata)
        }

        // 4. Check for finish
        const finishReason = chunk.candidates?.[0]?.finishReason
        if (finishReason) {
          const stopReason = mapFinishReason(finishReason, functionCallCount > 0)
          yield {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: totalUsage.output_tokens },
          } as any
          yield { type: 'message_stop' } as any
        }
      }

      // Edge case: stream ended without a finishReason
      if (hasEmittedStart && contentIndex === 0) {
        // Empty response — emit a blank text block
        yield {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'text', text: '' },
        } as any
        yield {
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: '(No response from model)' },
        } as any
        yield { type: 'content_block_stop', index: 0 } as any
        yield {
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', stop_sequence: null },
          usage: { output_tokens: 0 },
        } as any
        yield { type: 'message_stop' } as any
      }
    } catch (error: any) {
      throw this.toAnthropicError(error)
    }
  }

  /**
   * Translate a complete (non-streaming) Gemini response to Anthropic format.
   */
  private translateCompleteResponse(response: any, model: string): LLMResponse {
    const candidate = response.candidates?.[0]
    const parts: GeminiPart[] = candidate?.content?.parts ?? []
    const content: any[] = []
    let hasFunctionCalls = false

    for (const part of parts) {
      if (part.text !== undefined && !part.thought) {
        content.push({ type: 'text', text: part.text })
      }
      if (part.thought && part.text) {
        content.push({ type: 'thinking', thinking: part.text })
      }
      if (part.executableCode) {
        content.push({
          type: 'text',
          text: `\n\`\`\`${part.executableCode.language || 'python'}\n${part.executableCode.code}\n\`\`\`\n`,
        })
      }
      if (part.codeExecutionResult) {
        const outcome = part.codeExecutionResult.outcome === 'OUTCOME_OK' ? '' : `[${part.codeExecutionResult.outcome}] `
        content.push({
          type: 'text',
          text: `\n**Output:**\n\`\`\`\n${outcome}${part.codeExecutionResult.output}\n\`\`\`\n`,
        })
      }
      if (part.functionCall) {
        hasFunctionCalls = true
        const toolUseId = generateToolUseId()
        this.toolCallIdMap.record(toolUseId, part.functionCall.name)
        content.push({
          type: 'tool_use',
          id: toolUseId,
          name: part.functionCall.name,
          input: part.functionCall.args ?? {},
        })
      }
    }

    const usage = mapUsage(response.usageMetadata)
    const stopReason = mapFinishReason(candidate?.finishReason, hasFunctionCalls)

    return {
      id: generateMessageId(),
      type: 'message',
      role: 'assistant',
      model,
      content,
      stop_reason: stopReason,
      stop_sequence: null,
      usage,
    } as any
  }

  /**
   * Map Gemini errors to Anthropic SDK error format.
   * This allows the existing withRetry.ts logic to handle them unchanged.
   */
  private toAnthropicError(error: any): Error {
    const status = error.status || error.code || error.httpStatusCode
    const message = error.message || 'Gemini API error'

    const mapped: any = new Error(`[GeminiProvider] ${message}`)
    mapped.status = status

    // Map Gemini error codes to Anthropic-compatible status codes
    switch (status) {
      case 429:
      case 'RESOURCE_EXHAUSTED':
        mapped.status = 429
        mapped.headers = { 'retry-after': String(error.retryDelay || 30) }
        break
      case 503:
      case 'UNAVAILABLE':
        mapped.status = 529
        break
      case 401:
      case 'UNAUTHENTICATED':
        mapped.status = 401
        break
      case 403:
      case 'PERMISSION_DENIED':
        mapped.status = 403
        break
      case 400:
      case 'INVALID_ARGUMENT':
        mapped.status = 400
        if (message.toLowerCase().includes('token')) {
          mapped.message = `prompt is too long: ${message}`
        }
        break
      case 504:
      case 'DEADLINE_EXCEEDED':
        mapped.status = 504
        break
      default:
        mapped.status = typeof status === 'number' ? status : 500
    }

    mapped.error = { type: 'error', message: mapped.message }
    return mapped
  }
}

/**
 * Get the context window limit for a Gemini model.
 */
export function getGeminiContextLimit(model: string): number {
  const cleanModel = model.replace(/^vertex\//, '')
  return GEMINI_MODEL_LIMITS[cleanModel] ?? 1_048_576 // default 1M
}
