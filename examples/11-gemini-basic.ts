/**
 * Example: Using Gemini as the base model
 *
 * This demonstrates that all 61+ Claude Code tools work unchanged with Gemini.
 * The Provider layer translates between Anthropic and Gemini formats transparently.
 *
 * Setup:
 *   export GEMINI_API_KEY=your-api-key
 *
 * Run:
 *   npx tsx examples/11-gemini-basic.ts
 */

import { createAgent } from '../src/sdk.js'

async function main() {
  // ── Gemini API (API Key) ──
  const agent = createAgent({
    model: 'gemini-2.5-flash',
    env: {
      GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    },
    // All 61+ built-in tools are automatically available
    // Gemini drives the agent loop, calling Read, Bash, Glob, etc. as needed
  })

  console.log('=== Gemini Agent (streaming) ===\n')

  for await (const event of agent.query('Read the package.json file and tell me the project name and version')) {
    const msg = event as any
    if (msg.type === 'assistant') {
      const textBlocks = (msg.message?.content || [])
        .filter((b: any) => b.type === 'text')
        .map((b: any) => b.text)
      if (textBlocks.length > 0) {
        console.log(textBlocks.join(''))
      }
    }
    if (msg.type === 'result') {
      console.log(`\nDone. Turns: ${msg.num_turns}`)
    }
  }

  // ── Blocking prompt API ──
  console.log('\n=== Gemini Agent (blocking) ===\n')

  const result = await agent.prompt('What files are in the examples/ directory?')
  console.log(result.text)
  console.log(`\nTokens: ${result.usage.input_tokens + result.usage.output_tokens}`)
}

main().catch(console.error)
