/**
 * Example: Using Gemini via Vertex AI (Google Cloud)
 *
 * Vertex AI uses service account authentication instead of API keys.
 * The @google/genai SDK handles auth automatically via Application Default Credentials.
 *
 * Setup:
 *   export GOOGLE_CLOUD_PROJECT=your-project-id
 *   export GOOGLE_CLOUD_LOCATION=us-central1
 *   gcloud auth application-default login
 *
 * Run:
 *   npx tsx examples/12-gemini-vertex-ai.ts
 */

import { createAgent } from '../src/sdk.js'

async function main() {
  const agent = createAgent({
    model: 'gemini-2.5-pro',
    env: {
      GOOGLE_VERTEX_AI: 'true',
      GOOGLE_CLOUD_PROJECT: process.env.GOOGLE_CLOUD_PROJECT,
      GOOGLE_CLOUD_LOCATION: process.env.GOOGLE_CLOUD_LOCATION || 'us-central1',
    },
  })

  console.log('=== Gemini via Vertex AI ===\n')

  const result = await agent.prompt(
    'List all TypeScript files in the src/services/api/ directory and briefly describe what each one does'
  )
  console.log(result.text)
  console.log(`\nTokens: ${result.usage.input_tokens + result.usage.output_tokens}`)
  console.log(`Duration: ${result.duration_ms}ms`)
}

main().catch(console.error)
