# Anthropic ↔ OpenAI Format Conversion

This module provides conversion functions between Anthropic and OpenAI API formats, enabling interoperability between the two AI providers.

## Implementation Status

### ✅ Working Implementation: `anthropic_openai_simple`

A simplified, JSON-based implementation that handles the core conversion use cases:

- **System messages**: Anthropic's top-level `system` field ↔ OpenAI's system message
- **Role mapping**: `user`/`assistant` roles (bidirectional)
- **Common parameters**: `model`, `temperature`, `max_tokens`
- **Message content**: Simple text-based conversations

### ❌ Advanced Implementation: `anthropic_openai` (Disabled)

The full type-safe implementation was disabled due to:
- **Builder pattern incompatibilities**: Both crates use type-state builders that don't allow conditional building
- **API evolution**: Recent changes in anthropic-ox (tool definitions, cache control, content types) 
- **Complex type mapping**: Anthropic's enum-based `Content` vs OpenAI's simpler message structure

## Usage Examples

```rust
use conversion_ox::anthropic_openai_simple::{
    simple_anthropic_to_openai, 
    simple_openai_to_anthropic
};

// Convert Anthropic-style request to OpenAI format
let openai_request = simple_anthropic_to_openai(
    "claude-3-haiku-20240307".to_string(),
    Some("You are helpful".to_string()), // system message
    vec![
        ("user".to_string(), "Hello".to_string()),
        ("assistant".to_string(), "Hi there!".to_string()),
    ],
    Some(0.7),    // temperature  
    Some(1000),   // max_tokens
)?;

// Convert OpenAI-style request to Anthropic format  
let anthropic_request = simple_openai_to_anthropic(
    "gpt-3.5-turbo".to_string(),
    vec![
        ("system".to_string(), "You are helpful".to_string()),
        ("user".to_string(), "Hello".to_string()),
    ],
    Some(0.7),
    Some(1000), 
)?;
```

## Format Differences Handled

| Feature | Anthropic | OpenAI | Conversion |
|---------|-----------|--------|-------------|
| System messages | Top-level `system` field | System role message | ✅ Bidirectional |
| Message roles | `user`, `assistant` | `system`, `user`, `assistant`, `tool` | ✅ Core roles |
| Message content | `StringOrContents` enum | Simple string | ✅ Text content only |
| Tool calls | `Content::ToolUse` blocks | `tool_calls` array | ❌ Not implemented |
| Tool results | `Content::ToolResult` | Tool role messages | ❌ Not implemented |  
| Special content | `Thinking`, `Image`, etc. | Not supported | ❌ Not implemented |

## Limitations

The current simplified implementation does **not** support:
- Tool/function calling
- Image content  
- Anthropic-specific features (thinking blocks, cache control)
- OpenAI-specific parameters (`n`, `logit_bias`, `seed`)
- Complex message threading

## Future Improvements

To implement the full type-safe version:

1. **Resolve builder compatibility**: Create wrapper functions that collect parameters before building
2. **Handle new content types**: Map Anthropic's rich content blocks to OpenAI equivalents  
3. **Tool calling support**: Convert between the different tool calling patterns
4. **Response conversion**: Implement bidirectional response format conversion
5. **Advanced features**: Support for streaming, usage statistics, stop reasons

The simplified version demonstrates the core concepts and provides a foundation for more complex implementations.