import { describe, it, expect } from 'vitest'
import { GeminiProvider } from './gemini.js'

describe('GeminiProvider', () => {
  const provider = new GeminiProvider()

  describe('supportsModel', () => {
    it('matches gemini- prefix models', () => {
      expect(provider.supportsModel('gemini-2.5-pro')).toBe(true)
      expect(provider.supportsModel('gemini-2.5-flash')).toBe(true)
      expect(provider.supportsModel('gemini-2.5-flash-lite')).toBe(true)
    })

    it('matches vertex/ prefix models', () => {
      expect(provider.supportsModel('vertex/gemini-2.5-pro')).toBe(true)
    })

    it('matches aliases', () => {
      expect(provider.supportsModel('gemini-auto')).toBe(true)
      expect(provider.supportsModel('gemini-pro')).toBe(true)
      expect(provider.supportsModel('gemini-flash')).toBe(true)
      expect(provider.supportsModel('gemini-lite')).toBe(true)
    })

    it('does not match non-Gemini models', () => {
      expect(provider.supportsModel('claude-sonnet-4-6')).toBe(false)
      expect(provider.supportsModel('gpt-4o')).toBe(false)
      expect(provider.supportsModel('deepseek-chat')).toBe(false)
    })
  })

  describe('resolveModelName (via supportsModel indirection)', () => {
    // We can't directly test private methods, but we verify aliases
    // are recognized and the provider claims them
    it('recognizes all defined aliases', () => {
      const aliases = ['gemini-auto', 'gemini-pro', 'gemini-flash', 'gemini-lite']
      for (const alias of aliases) {
        expect(provider.supportsModel(alias)).toBe(true)
      }
    })
  })

  describe('type', () => {
    it('identifies as gemini provider', () => {
      expect(provider.type).toBe('gemini')
    })
  })
})
