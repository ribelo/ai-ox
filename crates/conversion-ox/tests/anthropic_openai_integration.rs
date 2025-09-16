#![cfg(feature = "anthropic-openai")]

use anthropic_ox::{
    message::{
        Content as AnthropicContent, Message as AnthropicMessage, Role as AnthropicRole,
        StringOrContents, Text as AnthropicText, ThinkingContent,
    },
    request::{ChatRequest as AnthropicRequest, ThinkingConfig},
    response::{ChatResponse as AnthropicResponse, StopReason},
};

use openai_ox::{
    request::ChatRequest as OpenAIRequest,
    response::{ChatResponse as OpenAIResponse, Choice as OpenAIChoice},
    responses::{
        ReasoningItem, ResponseMessage, ResponseOutputContent, ResponseOutputItem, ResponsesInput,
        ResponsesRequest, ResponsesResponse, response::TextItem,
    },
};

use ai_ox_common::openai_format::{Message as OpenAIMessage, MessageRole as OpenAIRole};

use conversion_ox::anthropic_openai::{
    anthropic_to_openai_request, anthropic_to_openai_responses_request,
    anthropic_to_openai_responses_response, openai_responses_to_anthropic_request,
    openai_responses_to_anthropic_response, openai_to_anthropic_response,
};

#[test]
fn test_anthropic_to_openai_basic_conversion() {
    // Create Anthropic request with system message
    let anthropic_request = AnthropicRequest::builder()
        .model("claude-3-haiku-20240307")
        .system(StringOrContents::String(
            "You are a helpful programming assistant".to_string(),
        ))
        .messages(vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: StringOrContents::String("What is Rust?".to_string()),
        }])
        .temperature(0.7)
        .max_tokens(1000)
        .build();

    // Convert to OpenAI format
    let openai_request = anthropic_to_openai_request(anthropic_request).unwrap();

    // Verify conversion
    assert_eq!(openai_request.model, "claude-3-haiku-20240307");
    assert_eq!(openai_request.temperature, Some(0.7));
    assert_eq!(openai_request.max_tokens, Some(1000));
    assert_eq!(openai_request.messages.len(), 2); // system + user

    // Check system message
    assert_eq!(openai_request.messages[0].role, OpenAIRole::System);
    assert_eq!(
        openai_request.messages[0].content,
        Some("You are a helpful programming assistant".to_string())
    );

    // Check user message
    assert_eq!(openai_request.messages[1].role, OpenAIRole::User);
    assert_eq!(
        openai_request.messages[1].content,
        Some("What is Rust?".to_string())
    );
}

#[test]
fn test_openai_to_anthropic_response_conversion() {
    // Create OpenAI response
    let openai_choice = OpenAIChoice {
        index: 0,
        message: OpenAIMessage {
            role: OpenAIRole::Assistant,
            content: Some(
                "Rust is a systems programming language that focuses on safety and performance."
                    .to_string(),
            ),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        finish_reason: Some("stop".to_string()),
        logprobs: None,
    };

    let openai_response = OpenAIResponse {
        id: "chatcmpl-test123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "gpt-3.5-turbo".to_string(),
        choices: vec![openai_choice],
        usage: None,
        system_fingerprint: None,
    };

    // Convert to Anthropic format
    let anthropic_response = openai_to_anthropic_response(openai_response).unwrap();

    // Verify conversion
    assert_eq!(anthropic_response.id, "chatcmpl-test123");
    assert_eq!(anthropic_response.model, "gpt-3.5-turbo");
    assert_eq!(anthropic_response.role, AnthropicRole::Assistant);

    assert_eq!(anthropic_response.content.len(), 1);
    if let AnthropicContent::Text(text) = &anthropic_response.content[0] {
        assert_eq!(
            text.text,
            "Rust is a systems programming language that focuses on safety and performance."
        );
    } else {
        panic!("Expected text content");
    }
}

#[test]
fn test_anthropic_to_openai_to_anthropic_roundtrip() {
    // Create original Anthropic request
    let original_request = AnthropicRequest::builder()
        .model("claude-3-haiku-20240307")
        .system(StringOrContents::String(
            "You are a helpful assistant".to_string(),
        ))
        .messages(vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: StringOrContents::String("Hello, how are you?".to_string()),
        }])
        .temperature(0.5)
        .max_tokens(500)
        .build();

    // Round trip: Anthropic -> OpenAI -> simulate response -> back to Anthropic
    let openai_request = anthropic_to_openai_request(original_request).unwrap();

    // Simulate OpenAI response based on the request
    let simulated_response = simulate_openai_response_from_request(&openai_request);

    // Convert back to Anthropic
    let final_response = openai_to_anthropic_response(simulated_response).unwrap();

    // Verify round-trip preserved key information
    assert_eq!(final_response.model, "claude-3-haiku-20240307");
    assert_eq!(final_response.role, AnthropicRole::Assistant);

    assert_eq!(final_response.content.len(), 1);
    if let AnthropicContent::Text(text) = &final_response.content[0] {
        assert_eq!(text.text, "I'm doing well, thank you for asking!");
    } else {
        panic!("Expected text content in round-trip result");
    }
}

#[test]
fn test_complex_conversation_roundtrip() {
    // Create Anthropic request with multiple messages
    let original_request = AnthropicRequest::builder()
        .model("claude-3-sonnet-20240229")
        .system(StringOrContents::String(
            "You are an expert in mathematics".to_string(),
        ))
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::String("What is 15 * 23?".to_string()),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: StringOrContents::String(
                    "Let me calculate that for you: 15 * 23 = 345".to_string(),
                ),
            },
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::String(
                    "Can you show me how you calculated that?".to_string(),
                ),
            },
        ])
        .temperature(0.3)
        .build();

    // Convert to OpenAI format
    let openai_request = anthropic_to_openai_request(original_request).unwrap();

    // Verify the conversion handles multiple messages correctly
    assert_eq!(openai_request.messages.len(), 4); // system + 3 conversation messages
    assert_eq!(openai_request.temperature, Some(0.3));

    // Check system message is first
    assert_eq!(openai_request.messages[0].role, OpenAIRole::System);

    // Check conversation flow
    assert_eq!(openai_request.messages[1].role, OpenAIRole::User);
    assert_eq!(openai_request.messages[2].role, OpenAIRole::Assistant);
    assert_eq!(openai_request.messages[3].role, OpenAIRole::User);

    // Simulate response and convert back
    let simulated_response = OpenAIResponse {
        id: "complex-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "claude-3-sonnet-20240229".to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: OpenAIRole::Assistant,
                content: Some("I calculated 15 * 23 by breaking it down: (10 + 5) * 23 = 10*23 + 5*23 = 230 + 115 = 345".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
    };

    let final_response = openai_to_anthropic_response(simulated_response).unwrap();

    // Verify the final response
    assert_eq!(final_response.model, "claude-3-sonnet-20240229");
    assert_eq!(final_response.role, AnthropicRole::Assistant);
}

#[test]
fn test_content_blocks_conversion() {
    // Test Anthropic request with content blocks
    let anthropic_request = AnthropicRequest::builder()
        .model("claude-3-haiku-20240307")
        .messages(vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: StringOrContents::Contents(vec![
                AnthropicContent::Text(AnthropicText::new("First part of the message".to_string())),
                AnthropicContent::Text(AnthropicText::new(
                    "Second part of the message".to_string(),
                )),
            ]),
        }])
        .build();

    let openai_request = anthropic_to_openai_request(anthropic_request).unwrap();

    // Should combine multiple text blocks into single message
    assert_eq!(openai_request.messages.len(), 1);
    assert_eq!(
        openai_request.messages[0].content,
        Some("First part of the message\nSecond part of the message".to_string())
    );
}

#[test]
fn test_error_handling() {
    // Test empty choices in OpenAI response
    let empty_openai_response = OpenAIResponse {
        id: "empty-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "gpt-3.5-turbo".to_string(),
        choices: vec![], // Empty choices
        usage: None,
        system_fingerprint: None,
    };

    let result = openai_to_anthropic_response(empty_openai_response);
    assert!(result.is_err());

    // Test empty messages in Anthropic request
    let empty_anthropic_request = AnthropicRequest::builder()
        .model("claude-3-haiku-20240307")
        .messages(Vec::<AnthropicMessage>::new()) // Empty messages
        .build();

    let result = anthropic_to_openai_request(empty_anthropic_request);
    assert!(result.is_err());
}

/// Helper function to simulate an OpenAI response based on a request
fn simulate_openai_response_from_request(request: &OpenAIRequest) -> OpenAIResponse {
    OpenAIResponse {
        id: "simulated-response".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: request.model.clone(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: OpenAIRole::Assistant,
                content: Some("I'm doing well, thank you for asking!".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
    }
}

// =================== RESPONSES API TESTS ===================

#[test]
fn test_anthropic_to_openai_responses_request_basic() {
    // Create Anthropic request with thinking config
    let anthropic_request = AnthropicRequest::builder()
        .model("claude-3-opus-20240229")
        .system(StringOrContents::String(
            "You are a helpful AI assistant".to_string(),
        ))
        .messages(vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: StringOrContents::String("Explain quantum computing".to_string()),
        }])
        .thinking(ThinkingConfig::new(5000))
        .max_tokens(2000)
        .build();

    // Convert to OpenAI Responses format
    let responses_request = anthropic_to_openai_responses_request(anthropic_request).unwrap();

    // Verify conversion
    assert_eq!(responses_request.model, "claude-3-opus-20240229");
    assert_eq!(responses_request.max_output_tokens, Some(2000));

    // Check reasoning config was created
    assert!(responses_request.reasoning.is_some());
    let reasoning = responses_request.reasoning.unwrap();
    assert_eq!(reasoning.effort, Some("high".to_string()));
    assert_eq!(reasoning.summary, Some("auto".to_string()));

    // Check instructions field (system prompt)
    assert_eq!(
        responses_request.instructions,
        Some("You are a helpful AI assistant".to_string())
    );

    // Check messages conversion (should only have user message now)
    if let ResponsesInput::Messages(messages) = responses_request.input {
        assert_eq!(messages.len(), 1); // only user message
        assert_eq!(messages[0].role, OpenAIRole::User);
        assert_eq!(
            messages[0].content,
            Some("Explain quantum computing".to_string())
        );
    } else {
        panic!("Expected Messages input type");
    }

    // Check include field for encrypted reasoning
    assert_eq!(
        responses_request.include,
        Some(vec!["reasoning.encrypted_content".to_string()])
    );
}

#[test]
fn test_openai_responses_to_anthropic_response_with_reasoning() {
    // Create OpenAI Responses response with reasoning
    let responses_response = ResponsesResponse {
        id: "resp-123".to_string(),
        created_at: 1234567890,
        output_text: String::new(),
        error: None,
        incomplete_details: None,
        instructions: None,
        metadata: None,
        model: "o3-mini".to_string(),
        object: "response".to_string(),
        output: vec![
            ResponseOutputItem::Reasoning {
                id: "reasoning-1".to_string(),
                summary: vec![serde_json::json!("Let me think about quantum computing step by step...")],
                content: Some(vec![serde_json::json!("encrypted_reasoning_data")]),
            },
            ResponseOutputItem::Message {
                id: "msg-1".to_string(),
                content: vec![ResponseOutputContent::Text {
                    text: "Quantum computing uses quantum bits (qubits) that can exist in superposition.".to_string(),
                    annotations: vec![],
                }],
                role: "assistant".to_string(),
                status: "completed".to_string(),
            },
        ],
        parallel_tool_calls: true,
        temperature: None,
        top_p: None,
        background: None,
        conversation: None,
        max_output_tokens: None,
        previous_response_id: None,
        prompt_cache_key: None,
        max_tool_calls: None,
        service_tier: None,
        top_logprobs: None,
        reasoning: None,
        safety_identifier: None,
        status: Some("completed".to_string()),
        text: None,
        usage: None,
        tool_choice: None,
        tools: vec![],
        truncation: None,
        user: None,
    };

    // Convert to Anthropic format
    let anthropic_response = openai_responses_to_anthropic_response(responses_response).unwrap();

    // Verify conversion
    assert_eq!(anthropic_response.id, "resp-123");
    assert_eq!(anthropic_response.model, "o3-mini");
    assert_eq!(anthropic_response.role, AnthropicRole::Assistant);
    assert_eq!(anthropic_response.stop_reason, Some(StopReason::EndTurn));

    // Check content conversion
    assert_eq!(anthropic_response.content.len(), 2);

    // First should be thinking content
    if let AnthropicContent::Thinking(thinking) = &anthropic_response.content[0] {
        assert_eq!(
            thinking.text,
            "Let me think about quantum computing step by step..."
        );
    } else {
        panic!("Expected thinking content first");
    }

    // Second should be text content
    if let AnthropicContent::Text(text) = &anthropic_response.content[1] {
        assert_eq!(
            text.text,
            "Quantum computing uses quantum bits (qubits) that can exist in superposition."
        );
    } else {
        panic!("Expected text content second");
    }
}

#[test]
fn test_anthropic_to_openai_responses_response_with_thinking() {
    // Create Anthropic response with thinking content
    let anthropic_response = AnthropicResponse {
        id: "msg-456".to_string(),
        r#type: "message".to_string(),
        model: "claude-3.5-sonnet".to_string(),
        role: AnthropicRole::Assistant,
        content: vec![
            AnthropicContent::Thinking(ThinkingContent::with_signature(
                "I need to break down this complex problem...".to_string(),
                "sig_abc123".to_string(),
            )),
            AnthropicContent::Text(AnthropicText::new(
                "Here's the solution to your problem:".to_string(),
            )),
            AnthropicContent::Text(AnthropicText::new(
                "Step 1: Initialize the system".to_string(),
            )),
        ],
        stop_reason: Some(StopReason::EndTurn),
        stop_sequence: None,
        usage: anthropic_ox::response::Usage::default(),
    };

    // Convert to OpenAI Responses format
    let responses_response = anthropic_to_openai_responses_response(anthropic_response).unwrap();

    // Verify conversion
    assert_eq!(responses_response.id, "msg-456");
    assert_eq!(responses_response.model, "claude-3.5-sonnet");
    assert_eq!(responses_response.status, Some("completed".to_string()));

    // Check output items
    assert_eq!(responses_response.output.len(), 2);

    // First should be reasoning item
    if let ResponseOutputItem::Reasoning { summary, .. } = &responses_response.output[0] {
        assert_eq!(
            summary.first().unwrap().as_str().unwrap(),
            "I need to break down this complex problem..."
        );
    } else {
        panic!("Expected reasoning item first");
    }

    // Second should be message with combined text
    if let ResponseOutputItem::Message { content, .. } = &responses_response.output[1] {
        // Check that content contains the expected text
        if let Some(content_item) = content.first() {
            if let openai_ox::responses::ResponseOutputContent::Text { text, .. } = content_item {
                assert_eq!(
                    text,
                    "Here's the solution to your problem:\nStep 1: Initialize the system"
                );
            } else {
                panic!("Expected text content");
            }
        } else {
            panic!("Expected content in message");
        }
    } else {
        panic!("Expected message item second");
    }
}

#[test]
fn test_openai_responses_to_anthropic_request_text_input() {
    // Create OpenAI Responses request with simple text input
    let responses_request = ResponsesRequest::builder()
        .model("gpt-5-turbo")
        .input(ResponsesInput::Text(
            "What is machine learning?".to_string(),
        ))
        .max_output_tokens(1500)
        .build();

    // Convert to Anthropic format
    let anthropic_request = openai_responses_to_anthropic_request(responses_request).unwrap();

    // Verify conversion
    assert_eq!(anthropic_request.model, "gpt-5-turbo");
    assert_eq!(anthropic_request.max_tokens, 1500);

    // Check messages
    assert_eq!(anthropic_request.messages.len(), 1);
    assert_eq!(anthropic_request.messages[0].role, AnthropicRole::User);

    if let StringOrContents::String(content) = &anthropic_request.messages[0].content {
        assert_eq!(content, "What is machine learning?");
    } else {
        panic!("Expected string content");
    }
}

#[test]
fn test_responses_api_full_roundtrip() {
    // Create original Anthropic request
    let original_request = AnthropicRequest::builder()
        .model("claude-3-opus")
        .system(StringOrContents::String(
            "You are an expert developer".to_string(),
        ))
        .messages(vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: StringOrContents::String("Write a Python function".to_string()),
        }])
        .thinking(ThinkingConfig::new(3000))
        .max_tokens(1000)
        .build();

    // Convert to OpenAI Responses format
    let responses_request =
        anthropic_to_openai_responses_request(original_request.clone()).unwrap();

    // Convert back to Anthropic
    let roundtrip_request = openai_responses_to_anthropic_request(responses_request).unwrap();

    // Verify key fields preserved
    assert_eq!(roundtrip_request.model, original_request.model);
    assert_eq!(roundtrip_request.max_tokens, original_request.max_tokens);
    assert_eq!(
        roundtrip_request.messages.len(),
        original_request.messages.len()
    );

    // System message should be preserved
    if let Some(StringOrContents::String(system)) = roundtrip_request.system {
        assert_eq!(system, "You are an expert developer");
    } else {
        panic!("System message not preserved in roundtrip");
    }

    // Thinking config should be preserved
    assert!(roundtrip_request.thinking.is_some());
}

#[test]
fn test_responses_api_error_handling() {
    // Test empty messages in Anthropic request
    let empty_request = AnthropicRequest::builder()
        .model("claude-3")
        .messages(Vec::<AnthropicMessage>::new())
        .build();

    let result = anthropic_to_openai_responses_request(empty_request);
    assert!(result.is_err());

    // Test empty output in OpenAI response
    let empty_response = ResponsesResponse {
        id: "empty".to_string(),
        created_at: 1234567890,
        output_text: String::new(),
        error: None,
        incomplete_details: None,
        instructions: None,
        metadata: None,
        model: "o3".to_string(),
        object: "response".to_string(),
        output: vec![],
        parallel_tool_calls: true,
        temperature: None,
        top_p: None,
        background: None,
        conversation: None,
        max_output_tokens: None,
        previous_response_id: None,
        prompt_cache_key: None,
        max_tool_calls: None,
        service_tier: None,
        top_logprobs: None,
        reasoning: None,
        safety_identifier: None,
        status: Some("completed".to_string()),
        text: None,
        usage: None,
        tool_choice: None,
        tools: vec![],
        truncation: None,
        user: None,
    };

    let result = openai_responses_to_anthropic_response(empty_response);
    assert!(result.is_err());
}

#[test]
fn test_responses_api_sanitizes_instructions() {
    // Create Anthropic request with system prompt containing Claude-specific tokens
    let anthropic_request = AnthropicRequest::builder()
        .model("claude-3-opus")
        .system(StringOrContents::String(
            "ephemeral\n<!-- This is a directive comment -->\ntext\nYou are a helpful assistant\n<!-- Another comment -->\nBe concise".to_string()
        ))
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::String("Hello".to_string()),
            },
        ])
        .max_tokens(1000)
        .build();

    // Convert to OpenAI Responses format
    let responses_request = anthropic_to_openai_responses_request(anthropic_request).unwrap();

    // Verify instructions were sanitized
    assert_eq!(
        responses_request.instructions,
        Some("You are a helpful assistant\nBe concise".to_string())
    );

    // Verify other fields are correct
    assert_eq!(responses_request.store, Some(false));
    assert_eq!(responses_request.stream, Some(true));
}

#[test]
fn test_responses_api_response_roundtrip() {
    // Create original Anthropic response with thinking content
    let original_response = AnthropicResponse {
        id: "msg-original".to_string(),
        r#type: "message".to_string(),
        model: "claude-3-opus".to_string(),
        role: AnthropicRole::Assistant,
        content: vec![
            AnthropicContent::Thinking(ThinkingContent::with_signature(
                "Let me analyze this problem step by step...".to_string(),
                "sig_xyz789".to_string(),
            )),
            AnthropicContent::Text(AnthropicText::new("Based on my analysis:".to_string())),
            AnthropicContent::Text(AnthropicText::new(
                "The solution is to use recursion.".to_string(),
            )),
        ],
        stop_reason: Some(StopReason::EndTurn),
        stop_sequence: None,
        usage: anthropic_ox::response::Usage {
            input_tokens: Some(100),
            output_tokens: Some(200),
            thinking_tokens: Some(50),
        },
    };

    // Convert to OpenAI Responses format
    let responses_response =
        anthropic_to_openai_responses_response(original_response.clone()).unwrap();

    // Verify intermediate conversion
    assert_eq!(responses_response.model, "claude-3-opus");
    assert_eq!(responses_response.output.len(), 2); // reasoning + message

    // Convert back to Anthropic
    let roundtrip_response = openai_responses_to_anthropic_response(responses_response).unwrap();

    // Verify roundtrip preserved key information
    assert_eq!(roundtrip_response.id, original_response.id);
    assert_eq!(roundtrip_response.model, original_response.model);
    assert_eq!(roundtrip_response.role, original_response.role);
    assert_eq!(
        roundtrip_response.stop_reason,
        original_response.stop_reason
    );

    // Check content preservation
    assert_eq!(roundtrip_response.content.len(), 2); // thinking + combined text

    // First should be thinking content
    if let AnthropicContent::Thinking(thinking) = &roundtrip_response.content[0] {
        assert_eq!(thinking.text, "Let me analyze this problem step by step...");
        // Note: signature is preserved as encrypted_content in Responses API
    } else {
        panic!("Expected thinking content to be preserved");
    }

    // Second should be text content (combined)
    if let AnthropicContent::Text(text) = &roundtrip_response.content[1] {
        assert_eq!(
            text.text,
            "Based on my analysis:\nThe solution is to use recursion."
        );
    } else {
        panic!("Expected text content to be preserved");
    }

    // Usage should be preserved
    assert_eq!(roundtrip_response.usage.input_tokens, Some(100));
    assert_eq!(roundtrip_response.usage.output_tokens, Some(200));
}
