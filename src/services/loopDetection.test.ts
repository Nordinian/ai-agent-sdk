import { describe, it, expect, beforeEach } from 'vitest'
import { LoopDetector, loopBreakMessage } from './loopDetection.js'

describe('LoopDetector', () => {
  let detector: LoopDetector

  beforeEach(() => {
    detector = new LoopDetector()
  })

  describe('Tier 1: Tool Call Repetition', () => {
    it('does not trigger on different tool calls', () => {
      for (let i = 0; i < 10; i++) {
        const result = detector.checkToolCall('read_file', { path: `/file${i}.ts` })
        expect(result.detected).toBe(false)
      }
    })

    it('triggers after 5 identical tool calls', () => {
      const args = { path: '/foo/bar.ts' }
      for (let i = 0; i < 4; i++) {
        const result = detector.checkToolCall('read_file', args)
        expect(result.detected).toBe(false)
      }
      // 5th identical call triggers
      const result = detector.checkToolCall('read_file', args)
      expect(result.detected).toBe(true)
      expect(result.type).toBe('tool_call_repetition')
      expect(result.detail).toContain('read_file')
    })

    it('resets count when a different tool is called', () => {
      const args = { path: '/foo.ts' }
      for (let i = 0; i < 4; i++) {
        detector.checkToolCall('read_file', args)
      }
      // Different tool resets the count
      detector.checkToolCall('write_file', { path: '/bar.ts', content: 'x' })

      // Start counting again from 1
      for (let i = 0; i < 4; i++) {
        const result = detector.checkToolCall('read_file', args)
        expect(result.detected).toBe(false)
      }
      // 5th triggers again
      const result = detector.checkToolCall('read_file', args)
      expect(result.detected).toBe(true)
    })

    it('treats different arguments as different calls', () => {
      for (let i = 0; i < 10; i++) {
        // Same tool name but different args each time
        const result = detector.checkToolCall('bash', { command: `echo ${i}` })
        expect(result.detected).toBe(false)
      }
    })
  })

  describe('Tier 2: Content Chanting', () => {
    it('does not trigger on varied content', () => {
      for (let i = 0; i < 20; i++) {
        const result = detector.checkContent(`This is unique sentence number ${i} with enough characters to fill a chunk.`)
        expect(result.detected).toBe(false)
      }
    })

    it('triggers on repeated content chunks', () => {
      const repeatedText = 'This is a repeating sentence that will be detected.'
      let detected = false
      for (let i = 0; i < 50; i++) {
        const result = detector.checkContent(repeatedText)
        if (result.detected) {
          detected = true
          expect(result.type).toBe('content_chanting')
          break
        }
      }
      expect(detected).toBe(true)
    })

    it('skips detection inside code blocks', () => {
      detector.checkContent('```\n')
      // Repeated content inside code block should not trigger
      const repeated = 'const x = 1; const x = 1; const x = 1; const x = 1; '
      for (let i = 0; i < 20; i++) {
        const result = detector.checkContent(repeated)
        expect(result.detected).toBe(false)
      }
      detector.checkContent('\n```\n')
    })

    it('resets when markdown structure elements are detected', () => {
      // Build up some history
      const text = 'Some repeated text that could be a loop pattern here.'
      for (let i = 0; i < 5; i++) {
        detector.checkContent(text)
      }
      // Heading resets tracking
      detector.checkContent('\n## New Section\n')
      // Count restarts
      for (let i = 0; i < 5; i++) {
        const result = detector.checkContent(text)
        expect(result.detected).toBe(false)
      }
    })
  })

  describe('checkAssistantMessage', () => {
    it('checks tool_use blocks', () => {
      const args = { path: '/foo.ts' }
      // First 4 calls via direct checkToolCall
      for (let i = 0; i < 4; i++) {
        detector.checkToolCall('read_file', args)
      }
      // 5th call via checkAssistantMessage
      const result = detector.checkAssistantMessage({
        content: [
          { type: 'tool_use', id: 'toolu_5', name: 'read_file', input: args },
        ],
      })
      expect(result.detected).toBe(true)
    })

    it('checks text blocks', () => {
      const message = {
        content: [
          { type: 'text', text: 'Hello world' },
        ],
      }
      const result = detector.checkAssistantMessage(message)
      expect(result.detected).toBe(false)
    })

    it('handles null/undefined content gracefully', () => {
      expect(detector.checkAssistantMessage({ content: null as any })).toEqual({
        detected: false,
        count: 0,
      })
      expect(detector.checkAssistantMessage({ content: undefined as any })).toEqual({
        detected: false,
        count: 0,
      })
    })
  })

  describe('clearDetection / reset', () => {
    it('clearDetection allows detection to resume', () => {
      const args = { path: '/foo.ts' }
      for (let i = 0; i < 5; i++) {
        detector.checkToolCall('read_file', args)
      }
      expect(detector.checkToolCall('read_file', args).detected).toBe(true)

      detector.clearDetection()
      // After clearing, new detections can happen
      const result = detector.checkToolCall('different_tool', { x: 1 })
      expect(result.detected).toBe(false)
    })

    it('reset clears everything including detection count', () => {
      const args = { path: '/foo.ts' }
      for (let i = 0; i < 5; i++) {
        detector.checkToolCall('read_file', args)
      }
      detector.reset()

      // After reset, count is back to 0
      for (let i = 0; i < 4; i++) {
        const result = detector.checkToolCall('read_file', args)
        expect(result.detected).toBe(false)
      }
    })
  })
})

describe('loopBreakMessage', () => {
  it('generates tool_call_repetition message', () => {
    const msg = loopBreakMessage({
      detected: true,
      type: 'tool_call_repetition',
      count: 1,
      detail: 'Tool "read_file" called 5 times',
    })
    expect(msg).toContain('LOOP DETECTED')
    expect(msg).toContain('same tool')
    expect(msg).toContain('different approach')
  })

  it('generates content_chanting message', () => {
    const msg = loopBreakMessage({
      detected: true,
      type: 'content_chanting',
      count: 1,
    })
    expect(msg).toContain('LOOP DETECTED')
    expect(msg).toContain('repetitive content')
  })
})
