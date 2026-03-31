// @ts-nocheck
/**
 * LLM Provider Registry
 *
 * Resolves a model name to the appropriate LLMProvider using pattern matching.
 * Providers are checked in registration order; the first match wins.
 * Falls back to Anthropic provider for unrecognized models (backward compatible).
 */

import type { LLMProvider } from './provider.js'

const providers: LLMProvider[] = []

/**
 * Register an LLM provider. Providers are checked in registration order.
 */
export function registerProvider(provider: LLMProvider): void {
  providers.push(provider)
}

/**
 * Resolve a model name to the appropriate provider.
 *
 * Matching order:
 *   1. First provider whose supportsModel(model) returns true
 *   2. Falls back to the first registered 'anthropic' provider
 *   3. Throws if no provider is registered at all
 *
 * @example
 *   resolveProvider('gemini-2.5-pro')   // → GeminiProvider
 *   resolveProvider('claude-sonnet-4-6') // → AnthropicProvider
 *   resolveProvider('unknown-model')     // → AnthropicProvider (fallback)
 */
export function resolveProvider(model: string): LLMProvider {
  for (const provider of providers) {
    if (provider.supportsModel(model)) {
      return provider
    }
  }

  // Fallback to Anthropic provider for backward compatibility
  const anthropicProvider = providers.find(p => p.type === 'anthropic')
  if (anthropicProvider) {
    return anthropicProvider
  }

  throw new Error(
    `No LLM provider registered for model "${model}". ` +
    `Registered providers: [${providers.map(p => p.type).join(', ')}]`
  )
}

/**
 * Check whether a model name resolves to a non-Anthropic provider.
 * Useful for feature-gating (e.g. skip prompt caching for non-Anthropic models).
 */
export function isNonAnthropicModel(model: string): boolean {
  try {
    return resolveProvider(model).type !== 'anthropic'
  } catch {
    return false
  }
}

/**
 * Get all registered providers (for debugging/introspection).
 */
export function getRegisteredProviders(): readonly LLMProvider[] {
  return providers
}

/**
 * Clear all registered providers (for testing).
 */
export function clearProviders(): void {
  providers.length = 0
}
