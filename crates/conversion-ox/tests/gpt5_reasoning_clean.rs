use conversion_ox::anthropic_openrouter::openrouter_to_anthropic_response;
use openrouter_ox::response::ChatCompletionResponse as OpenRouterResponse;
use anthropic_ox::message::Content as AnthropicContent;
use serde_json;

const GPT5_REASONING_RESPONSE: &str = r#"
{
  "id": "gen-1756645138-hAeJlBtx3tiKd5dwy3Wq",
  "provider": "OpenAI",
  "model": "openai/gpt-5",
  "object": "chat.completion",
  "created": 1756645138,
  "choices": [
    {
      "logprobs": null,
      "finish_reason": "stop",
      "native_finish_reason": "completed",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "refusal": null,
        "reasoning": null,
        "reasoning_details": [
          {
            "type": "reasoning.encrypted",
            "data": "encrypted-reasoning-data",
            "id": "rs_68b4471356a0819e9e7901ab5eb810c60f2f0d738099c4b6",
            "format": "openai-responses-v1",
            "index": 0
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 0,
    "total_tokens": 7
  }
}
"#;

#[test]
fn test_gpt5_empty_content_extracts_from_reasoning_data() {
    // This test validates the TDD RED->GREEN cycle for GPT-5 reasoning extraction
    // BEFORE FIX: GPT-5 returned empty content array and reasoning was lost
    // AFTER FIX: The extract_reasoning_content helper extracts from reasoning_details during deserialization
    
    let gpt5_response: OpenRouterResponse = serde_json::from_str(GPT5_REASONING_RESPONSE)
        .expect("Failed to parse GPT-5 response");

    // GREEN: After our fix, the reasoning_details are consumed during deserialization
    // and converted into actual content via extract_reasoning_content()
    assert!(!gpt5_response.choices[0].message.content.0.is_empty(), "Content should be extracted during deserialization");
    
    // Verify the content contains the reasoning placeholder
    if let Some(first_content) = gpt5_response.choices[0].message.content.0.first() {
        match first_content {
            openrouter_ox::message::ContentPart::Text(text) => {
                assert_eq!(text.text, "[Encrypted reasoning data]", "Should have extracted reasoning data placeholder");
            }
            _ => panic!("Expected text content"),
        }
    } else {
        panic!("Should have content after reasoning extraction");
    }

    // Convert to Anthropic - this should preserve the extracted content
    let anthropic_response = openrouter_to_anthropic_response(gpt5_response)
        .expect("Failed to convert GPT-5 response to Anthropic");

    // Verify Anthropic response has the extracted reasoning text
    assert!(!anthropic_response.content.is_empty(), "Anthropic content should not be empty");
    
    // Check for text content (not thinking, since it comes from OpenRouter content not reasoning field)
    let has_text_content = anthropic_response.content.iter().any(|content| {
        match content {
            AnthropicContent::Text(text) => text.text == "[Encrypted reasoning data]",
            _ => false,
        }
    });
    assert!(has_text_content, "Should have text content with reasoning data placeholder");
}