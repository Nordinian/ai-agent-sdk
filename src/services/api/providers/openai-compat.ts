// @ts-nocheck
/**
 * OpenAICompatProvider — OpenAI-compatible LLM provider.
 *
 * Supports OpenAI, DeepSeek, Groq, Mistral, Ollama, and any OpenAI-compatible
 * API via baseURL configuration. Translates between Anthropic message format
 * (SDK internal) and OpenAI chat completion format.
 *
 * Uses dynamic import of the `openai` npm package to avoid hard dependency
 * when this provider is not in use.
 */

import type { LLMProvider, LLMCreateParams, LLMResponse, LLMStreamEvent } from '../provider.js'
import type {
  BetaMessageParam,
  BetaContentBlockParam,
} from '@anthropic-ai/sdk/resources/index.mjs'
import type { SystemPrompt } from '../../../utils/systemPromptType.js'

// ============================================================================
// Provider routing — match model name to API key + base URL
// ============================================================================

interface ProviderConfig {
  apiKey: string
  baseURL: string
  /** Actual model name to send to the API (strip prefix if needed) */
  model: string
}

/**
 * Resolve provider configuration from environment variables and model name.
 */
function resolveProviderConfig(model: string): ProviderConfig {
  // Explicit prefix: openai-compat/my-model → strip prefix, use OPENAI env
  if (model.startsWith('openai-compat/')) {
    const actualModel = model.slice('openai-compat/'.length)
    return {
      apiKey: process.env.OPENAI_API_KEY || '',
      baseURL: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
      model: actualModel,
    }
  }

  // DeepSeek models
  if (/^deepseek-/.test(model)) {
    return {
      apiKey: process.env.DEEPSEEK_API_KEY || process.env.OPENAI_API_KEY || '',
      baseURL: process.env.OPENAI_BASE_URL || 'https://api.deepseek.com/v1',
      model,
    }
  }

  // Mistral / Codestral models
  if (/^(mistral-|codestral-)/.test(model)) {
    return {
      apiKey: process.env.MISTRAL_API_KEY || process.env.OPENAI_API_KEY || '',
      baseURL: process.env.OPENAI_BASE_URL || 'https://api.mistral.ai/v1',
      model,
    }
  }

  // Llama / Qwen / Mixtral — typically via Groq, Together, or Ollama
  if (/^(llama-|qwen-|mixtral-)/.test(model)) {
    // Check for Ollama first (local, no key needed)
    if (process.env.OLLAMA_BASE_URL) {
      return {
        apiKey: 'ollama', // Ollama doesn't need a real key but the SDK requires one
        baseURL: process.env.OLLAMA_BASE_URL,
        model,
      }
    }
    // Groq
    if (process.env.GROQ_API_KEY) {
      return {
        apiKey: process.env.GROQ_API_KEY,
        baseURL: process.env.OPENAI_BASE_URL || 'https://api.groq.com/openai/v1',
        model,
      }
    }
    // Fallback to OPENAI_* env vars (e.g. Together AI)
    return {
      apiKey: process.env.OPENAI_API_KEY || '',
      baseURL: process.env.OPENAI_BASE_URL || 'http://localhost:11434/v1',
      model,
    }
  }

  // GPT / O1 / O3 / O4 models → OpenAI
  return {
    apiKey: process.env.OPENAI_API_KEY || '',
    baseURL: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
    model,
  }
}

// ============================================================================
// Message translation: Anthropic → OpenAI
// ============================================================================

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | OpenAIContentPart[] | null
  tool_calls?: OpenAIToolCall[]
  tool_call_id?: string
  name?: string
}

interface OpenAIContentPart {
  type: 'text' | 'image_url'
  text?: string
  image_url?: { url: string; detail?: string }
}

interface OpenAIToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters?: Record<string, unknown>
  }
}

/**
 * Convert Anthropic system prompt to OpenAI system message(s).
 */
function translateSystem(system: string | SystemPrompt[] | undefined): OpenAIMessage[] {
  if (!system) return []

  let text: string
  if (typeof system === 'string') {
    text = system
  } else if (Array.isArray(system)) {
    text = (system as any[])
      .map(block => {
        if (typeof block === 'string') return block
        if (block.type === 'text') return block.text
        return ''
      })
      .filter(Boolean)
      .join('\n\n')
  } else {
    return []
  }

  if (!text) return []
  return [{ role: 'system', content: text }]
}

/**
 * Convert Anthropic messages to OpenAI messages.
 */
function translateMessages(messages: BetaMessageParam[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = []

  for (const msg of messages) {
    const blocks = typeof msg.content === 'string'
      ? [{ type: 'text' as const, text: msg.content }]
      : (msg.content as BetaContentBlockParam[])

    if (msg.role === 'assistant') {
      // Collect text parts and tool_calls from assistant content blocks
      const textParts: string[] = []
      const toolCalls: OpenAIToolCall[] = []

      for (const block of blocks) {
        switch (block.type) {
          case 'text':
            textParts.push(block.text)
            break
          case 'tool_use':
            toolCalls.push({
              id: block.id,
              type: 'function',
              function: {
                name: block.name,
                arguments: JSON.stringify(block.input ?? {}),
              },
            })
            break
          case 'thinking':
            // OpenAI doesn't have thinking — skip
            break
          default:
            // Unknown block: serialize as text
            textParts.push(JSON.stringify(block))
        }
      }

      const assistantMsg: OpenAIMessage = { role: 'assistant' }
      if (textParts.length > 0) {
        assistantMsg.content = textParts.join('')
      } else {
        assistantMsg.content = null
      }
      if (toolCalls.length > 0) {
        assistantMsg.tool_calls = toolCalls
      }
      result.push(assistantMsg)
    } else {
      // User role — may contain text, images, and tool_result blocks
      // Tool results must be separate messages with role: 'tool'
      const contentParts: OpenAIContentPart[] = []

      for (const block of blocks) {
        switch (block.type) {
          case 'text':
            contentParts.push({ type: 'text', text: block.text })
            break

          case 'image': {
            const source = (block as any).source
            if (source?.type === 'base64') {
              contentParts.push({
                type: 'image_url',
                image_url: {
                  url: `data:${source.media_type};base64,${source.data}`,
                },
              })
            } else if (source?.url) {
              contentParts.push({
                type: 'image_url',
                image_url: { url: source.url },
              })
            }
            break
          }

          case 'tool_result': {
            // Flush any accumulated content parts as a user message first
            if (contentParts.length > 0) {
              result.push({
                role: 'user',
                content: contentParts.length === 1 && contentParts[0].type === 'text'
                  ? contentParts[0].text
                  : [...contentParts],
              })
              contentParts.length = 0
            }

            // Emit tool result as a separate message
            result.push({
              role: 'tool',
              tool_call_id: (block as any).tool_use_id,
              content: extractToolResultContent((block as any).content),
            })
            break
          }

          default:
            contentParts.push({ type: 'text', text: JSON.stringify(block) })
        }
      }

      // Emit remaining content parts as a user message
      if (contentParts.length > 0) {
        result.push({
          role: 'user',
          content: contentParts.length === 1 && contentParts[0].type === 'text'
            ? contentParts[0].text
            : contentParts,
        })
      }
    }
  }

  return result
}

/**
 * Extract text content from a tool_result's content field.
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
 * Convert Anthropic tool definitions to OpenAI function tools.
 * Anthropic's input_schema is already JSON Schema — pass directly as parameters.
 */
function translateTools(tools: any[]): OpenAITool[] {
  const result: OpenAITool[] = []

  for (const tool of tools) {
    // Skip Anthropic-specific tool types
    if (tool.type === 'computer_20241022' || tool.type === 'bash_20241022' || tool.type === 'text_editor_20241022') {
      continue
    }

    result.push({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description || '',
        ...(tool.input_schema ? { parameters: tool.input_schema } : {}),
      },
    })
  }

  return result
}

// ============================================================================
// Response translation: OpenAI → Anthropic
// ============================================================================

/**
 * Map OpenAI finish_reason to Anthropic stop_reason.
 *
 * CRITICAL: 'tool_calls' must map to 'tool_use' — this is the signal that
 * makes the agent loop enter the tool execution branch.
 */
function mapFinishReason(finishReason: string | null | undefined, hasToolCalls: boolean): string {
  // If there are tool calls, always return 'tool_use' regardless of finish_reason
  if (hasToolCalls) return 'tool_use'

  switch (finishReason) {
    case 'stop':
      return 'end_turn'
    case 'length':
      return 'max_tokens'
    case 'tool_calls':
      return 'tool_use'
    case 'content_filter':
      return 'end_turn'
    default:
      return 'end_turn'
  }
}

/**
 * Map OpenAI usage to Anthropic usage format.
 */
function mapUsage(usage: any): { input_tokens: number; output_tokens: number } {
  return {
    input_tokens: usage?.prompt_tokens ?? 0,
    output_tokens: usage?.completion_tokens ?? 0,
  }
}

/**
 * Generate an Anthropic-style message ID.
 */
function generateMessageId(): string {
  return `msg_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`
}

/**
 * Generate an Anthropic-style tool_use ID.
 */
function generateToolUseId(): string {
  return `toolu_${crypto.randomUUID().replace(/-/g, '').slice(0, 20)}`
}

// ============================================================================
// OpenAICompatProvider
// ============================================================================

export class OpenAICompatProvider implements LLMProvider {
  readonly type = 'openai-compat' as const
  private clientCache = new Map<string, any>()

  supportsModel(model: string): boolean {
    return /^(gpt-|o1-|o3-|o4-|deepseek-|mistral-|codestral-|llama-|qwen-|mixtral-|openai-compat\/)/.test(model)
  }

  /**
   * Lazily create an OpenAI client for the given model's provider config.
   * Clients are cached by baseURL to avoid re-creation.
   */
  private async getClient(model: string): Promise<{ client: any; resolvedModel: string }> {
    const config = resolveProviderConfig(model)
    const cacheKey = `${config.baseURL}:${config.apiKey}`

    let client = this.clientCache.get(cacheKey)
    if (!client) {
      // Dynamic import to avoid hard dependency when OpenAI is not used
      const { default: OpenAI } = await import('openai')
      client = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseURL,
      })
      this.clientCache.set(cacheKey, client)
    }

    return { client, resolvedModel: config.model }
  }

  /**
   * Non-streaming message creation.
   * Translates request/response between Anthropic and OpenAI formats.
   */
  async createMessage(params: LLMCreateParams): Promise<LLMResponse> {
    const { client, resolvedModel } = await this.getClient(params.model)

    const systemMsgs = translateSystem(params.system)
    const userMsgs = translateMessages(params.messages)
    const messages = [...systemMsgs, ...userMsgs]
    const tools = translateTools(params.tools ?? [])

    const requestParams: any = {
      model: resolvedModel,
      messages,
      max_tokens: params.max_tokens,
    }

    if (params.temperature !== undefined) {
      requestParams.temperature = params.temperature
    }

    if (tools.length > 0) {
      requestParams.tools = tools
    }

    let response: any
    try {
      response = await client.chat.completions.create(requestParams)
    } catch (error: any) {
      throw this.toAnthropicError(error)
    }

    return this.translateCompleteResponse(response, params.model)
  }

  /**
   * Streaming message creation.
   * Wraps OpenAI's SSE stream into Anthropic's fine-grained event protocol.
   */
  async *createMessageStream(params: LLMCreateParams): AsyncGenerator<LLMStreamEvent, void, undefined> {
    const { client, resolvedModel } = await this.getClient(params.model)

    const systemMsgs = translateSystem(params.system)
    const userMsgs = translateMessages(params.messages)
    const messages = [...systemMsgs, ...userMsgs]
    const tools = translateTools(params.tools ?? [])

    const requestParams: any = {
      model: resolvedModel,
      messages,
      max_tokens: params.max_tokens,
      stream: true,
      stream_options: { include_usage: true },
    }

    if (params.temperature !== undefined) {
      requestParams.temperature = params.temperature
    }

    if (tools.length > 0) {
      requestParams.tools = tools
    }

    let stream: AsyncIterable<any>
    try {
      stream = await client.chat.completions.create(requestParams)
    } catch (error: any) {
      throw this.toAnthropicError(error)
    }

    // ── Translate OpenAI SSE chunks → Anthropic stream events ──

    const messageId = generateMessageId()
    let contentIndex = 0
    let hasEmittedStart = false
    let hasEmittedTextBlock = false
    let totalUsage = { input_tokens: 0, output_tokens: 0 }

    // Track tool call state across chunks (OpenAI streams tool calls incrementally)
    const activeToolCalls = new Map<number, {
      id: string
      name: string
      arguments: string
      blockIndex: number
      started: boolean
    }>()

    try {
      for await (const chunk of stream) {
        const choice = chunk.choices?.[0]

        // 1. Emit message_start on first chunk
        if (!hasEmittedStart) {
          yield {
            type: 'message_start',
            message: {
              id: messageId,
              type: 'message',
              role: 'assistant',
              model: params.model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            },
          } as any
          hasEmittedStart = true
        }

        // 2. Track usage (may come in a final chunk with choices=[])
        if (chunk.usage) {
          totalUsage = mapUsage(chunk.usage)
        }

        if (!choice) continue

        const delta = choice.delta

        // 3. Text content delta
        if (delta?.content) {
          if (!hasEmittedTextBlock) {
            yield {
              type: 'content_block_start',
              index: contentIndex,
              content_block: { type: 'text', text: '' },
            } as any
            hasEmittedTextBlock = true
          }
          yield {
            type: 'content_block_delta',
            index: contentIndex,
            delta: { type: 'text_delta', text: delta.content },
          } as any
        }

        // 4. Tool call deltas (OpenAI streams these incrementally)
        if (delta?.tool_calls) {
          // Close text block before first tool call
          if (hasEmittedTextBlock) {
            yield { type: 'content_block_stop', index: contentIndex } as any
            contentIndex++
            hasEmittedTextBlock = false
          }

          for (const tc of delta.tool_calls) {
            const tcIndex = tc.index ?? 0

            if (!activeToolCalls.has(tcIndex)) {
              // New tool call — initialize tracking
              activeToolCalls.set(tcIndex, {
                id: tc.id || generateToolUseId(),
                name: tc.function?.name || '',
                arguments: '',
                blockIndex: contentIndex + tcIndex,
                started: false,
              })
            }

            const tracked = activeToolCalls.get(tcIndex)!

            // Update fields as they stream in
            if (tc.id) tracked.id = tc.id
            if (tc.function?.name) tracked.name = tc.function.name
            if (tc.function?.arguments) tracked.arguments += tc.function.arguments

            // Emit content_block_start once we have the name
            if (!tracked.started && tracked.name) {
              tracked.started = true
              yield {
                type: 'content_block_start',
                index: tracked.blockIndex,
                content_block: {
                  type: 'tool_use',
                  id: tracked.id,
                  name: tracked.name,
                  input: {},
                },
              } as any
            }

            // Emit argument deltas
            if (tracked.started && tc.function?.arguments) {
              yield {
                type: 'content_block_delta',
                index: tracked.blockIndex,
                delta: {
                  type: 'input_json_delta',
                  partial_json: tc.function.arguments,
                },
              } as any
            }
          }
        }

        // 5. Check for finish
        if (choice.finish_reason) {
          // Close any open text block
          if (hasEmittedTextBlock) {
            yield { type: 'content_block_stop', index: contentIndex } as any
            contentIndex++
            hasEmittedTextBlock = false
          }

          // Close all open tool call blocks
          for (const [, tracked] of activeToolCalls) {
            if (tracked.started) {
              yield { type: 'content_block_stop', index: tracked.blockIndex } as any
            }
          }

          const hasToolCalls = activeToolCalls.size > 0
          const stopReason = mapFinishReason(choice.finish_reason, hasToolCalls)

          yield {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: totalUsage.output_tokens },
          } as any
          yield { type: 'message_stop' } as any
        }
      }

      // Edge case: stream ended without a finish_reason
      if (hasEmittedStart && contentIndex === 0 && !hasEmittedTextBlock && activeToolCalls.size === 0) {
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
   * Translate a complete (non-streaming) OpenAI response to Anthropic format.
   */
  private translateCompleteResponse(response: any, model: string): LLMResponse {
    const choice = response.choices?.[0]
    const message = choice?.message
    const content: any[] = []
    let hasToolCalls = false

    // Text content
    if (message?.content) {
      content.push({ type: 'text', text: message.content })
    }

    // Tool calls
    if (message?.tool_calls && message.tool_calls.length > 0) {
      hasToolCalls = true
      for (const tc of message.tool_calls) {
        let parsedArgs = {}
        try {
          parsedArgs = JSON.parse(tc.function.arguments || '{}')
        } catch {
          parsedArgs = {}
        }
        content.push({
          type: 'tool_use',
          id: tc.id || generateToolUseId(),
          name: tc.function.name,
          input: parsedArgs,
        })
      }
    }

    // Ensure at least one content block
    if (content.length === 0) {
      content.push({ type: 'text', text: '' })
    }

    const usage = mapUsage(response.usage)
    const stopReason = mapFinishReason(choice?.finish_reason, hasToolCalls)

    return {
      id: response.id ? `msg_${response.id}` : generateMessageId(),
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
   * Map OpenAI errors to Anthropic SDK error format.
   * This allows the existing withRetry.ts logic to handle them unchanged.
   */
  private toAnthropicError(error: any): Error {
    const status = error.status || error.statusCode || error.code
    const message = error.message || 'OpenAI-compatible API error'

    const mapped: any = new Error(`[OpenAICompatProvider] ${message}`)
    mapped.status = typeof status === 'number' ? status : 500

    switch (status) {
      case 429:
        mapped.status = 429
        mapped.headers = {
          'retry-after': String(error.headers?.['retry-after'] || 30),
        }
        break
      case 503:
        mapped.status = 529 // Anthropic overloaded equivalent
        break
      case 401:
        mapped.status = 401
        break
      case 403:
        mapped.status = 403
        break
      case 400:
        mapped.status = 400
        if (message.toLowerCase().includes('token') || message.toLowerCase().includes('length')) {
          mapped.message = `prompt is too long: ${message}`
        }
        break
      case 404:
        mapped.status = 404
        break
      case 504:
        mapped.status = 504
        break
    }

    mapped.error = { type: 'error', message: mapped.message }
    return mapped
  }
}
