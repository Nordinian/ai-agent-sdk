// @ts-nocheck
/**
 * Gemini Tool Schema Translator
 *
 * Converts Anthropic tool definitions (BetaToolUnion with JSON Schema input_schema)
 * to Gemini FunctionDeclaration format.
 *
 * Key differences:
 *   - JSON Schema type values: lowercase ('string') → uppercase ('STRING')
 *   - Gemini does not support: oneOf, anyOf, allOf, $ref, $schema, additionalProperties
 *   - Gemini has a tool count limit (~128 function declarations)
 */

// ============================================================================
// Types — lightweight Gemini schema types
// ============================================================================

export interface GeminiFunctionDeclaration {
  name: string
  description: string
  parameters?: GeminiSchema
}

export interface GeminiSchema {
  type: string
  description?: string
  properties?: Record<string, GeminiSchema>
  items?: GeminiSchema
  required?: string[]
  enum?: string[]
  format?: string
  nullable?: boolean
}

const GEMINI_MAX_TOOLS = 128

// ============================================================================
// Anthropic Tool → Gemini FunctionDeclaration
// ============================================================================

/**
 * Convert an array of Anthropic tool definitions to Gemini FunctionDeclarations.
 * Truncates to GEMINI_MAX_TOOLS if the count exceeds Gemini's limit.
 */
export function toGeminiFunctionDeclarations(
  tools: any[],
): GeminiFunctionDeclaration[] {
  const declarations: GeminiFunctionDeclaration[] = []

  for (const tool of tools) {
    // Skip tool types that Gemini cannot handle
    if (tool.type === 'computer_20241022' || tool.type === 'bash_20241022' || tool.type === 'text_editor_20241022') {
      continue
    }

    const declaration: GeminiFunctionDeclaration = {
      name: tool.name,
      description: tool.description || '',
    }

    // Convert input_schema (JSON Schema) to Gemini parameters
    const schema = tool.input_schema
    if (schema && typeof schema === 'object' && Object.keys(schema).length > 0) {
      declaration.parameters = convertSchemaToGemini(schema)
    }

    declarations.push(declaration)
  }

  // Enforce Gemini's tool count limit
  if (declarations.length > GEMINI_MAX_TOOLS) {
    console.warn(
      `[GeminiProvider] Tool count (${declarations.length}) exceeds Gemini limit (${GEMINI_MAX_TOOLS}). ` +
      `Truncating to first ${GEMINI_MAX_TOOLS} tools.`
    )
    return declarations.slice(0, GEMINI_MAX_TOOLS)
  }

  return declarations
}

/**
 * Recursively convert a JSON Schema object to Gemini Schema format.
 *
 * Main transformations:
 *   - type: 'string' → 'STRING' (uppercase)
 *   - oneOf/anyOf → merged into a single object with all properties optional
 *   - Unsupported keywords ($schema, $ref, additionalProperties, examples) → removed
 */
export function convertSchemaToGemini(schema: any): GeminiSchema {
  if (!schema || typeof schema !== 'object') {
    return { type: 'STRING' }
  }

  // Handle oneOf / anyOf — Gemini doesn't support union types
  if (schema.oneOf || schema.anyOf) {
    return mergeVariantsToGemini(schema.oneOf || schema.anyOf, schema.description)
  }

  // Handle allOf — merge all schemas
  if (schema.allOf) {
    const merged = mergeAllOf(schema.allOf)
    return convertSchemaToGemini(merged)
  }

  // Handle $ref — shouldn't appear after dereferencing, but fallback to string
  if (schema.$ref) {
    return { type: 'STRING', description: `Reference: ${schema.$ref}` }
  }

  const result: GeminiSchema = {}

  // Convert type to uppercase
  if (schema.type) {
    result.type = mapJsonSchemaType(schema.type)
  } else {
    result.type = 'STRING' // default
  }

  if (schema.description) {
    result.description = schema.description
  }

  if (schema.enum) {
    result.enum = schema.enum.map(String)
  }

  if (schema.format) {
    result.format = schema.format
  }

  if (schema.nullable) {
    result.nullable = true
  }

  // Recurse into object properties
  if (schema.properties) {
    result.properties = {}
    for (const [key, value] of Object.entries(schema.properties)) {
      result.properties[key] = convertSchemaToGemini(value)
    }
  }

  // Recurse into array items
  if (schema.items) {
    result.items = convertSchemaToGemini(schema.items)
  }

  // Preserve required array
  if (schema.required && Array.isArray(schema.required)) {
    result.required = schema.required
  }

  return result
}

/**
 * Map JSON Schema type strings to Gemini type enums (uppercase).
 */
function mapJsonSchemaType(type: string | string[]): string {
  // Handle array of types: ['string', 'null'] → 'STRING' with nullable
  if (Array.isArray(type)) {
    const nonNull = type.filter(t => t !== 'null')
    return mapSingleType(nonNull[0] || 'string')
  }
  return mapSingleType(type)
}

function mapSingleType(type: string): string {
  switch (type.toLowerCase()) {
    case 'string': return 'STRING'
    case 'number': return 'NUMBER'
    case 'integer': return 'INTEGER'
    case 'boolean': return 'BOOLEAN'
    case 'array': return 'ARRAY'
    case 'object': return 'OBJECT'
    case 'null': return 'STRING' // Gemini doesn't have null type
    default: return 'STRING'
  }
}

/**
 * Merge oneOf/anyOf variants into a single object schema.
 * All properties from all variants are included, all marked as optional.
 */
function mergeVariantsToGemini(variants: any[], description?: string): GeminiSchema {
  const mergedProperties: Record<string, GeminiSchema> = {}

  for (const variant of variants) {
    if (variant && typeof variant === 'object') {
      if (variant.properties) {
        for (const [key, value] of Object.entries(variant.properties)) {
          if (!mergedProperties[key]) {
            mergedProperties[key] = convertSchemaToGemini(value)
          }
        }
      }
      // If a variant is a simple type (not object), add as a description
      if (variant.type && variant.type !== 'object' && !variant.properties) {
        mergedProperties[`_variant_${variant.type}`] = {
          type: mapSingleType(variant.type),
          description: variant.description || `Variant of type ${variant.type}`,
        }
      }
    }
  }

  const result: GeminiSchema = {
    type: 'OBJECT',
    properties: mergedProperties,
    // No 'required' — all fields optional since they come from different variants
  }

  if (description) {
    result.description = description
  }

  return result
}

/**
 * Merge allOf schemas into a single schema.
 */
function mergeAllOf(schemas: any[]): any {
  const merged: any = { type: 'object', properties: {}, required: [] }
  for (const schema of schemas) {
    if (schema.properties) {
      Object.assign(merged.properties, schema.properties)
    }
    if (schema.required) {
      merged.required.push(...schema.required)
    }
    if (schema.description && !merged.description) {
      merged.description = schema.description
    }
  }
  if (merged.required.length === 0) delete merged.required
  return merged
}
