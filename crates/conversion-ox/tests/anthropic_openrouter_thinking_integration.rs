use anthropic_ox::{
    message::{Content as AnthropicContent, Message as AnthropicMessage, Role as AnthropicRole, ThinkingContent, StringOrContents},
    request::ChatRequest as AnthropicRequest,
};

use openrouter_ox::{
    message::{AssistantMessage, ContentPart, Message as OpenRouterMessage},
    // request::ChatRequest as OpenRouterRequest,
    response::{
        ChatCompletionResponse as OpenRouterResponse, 
        Choice, FinishReason as OpenRouterFinishReason,
        ReasoningDetail, Usage as OpenRouterUsage
    },
};

use conversion_ox::anthropic_openrouter::{
    anthropic_to_openrouter_request, openrouter_to_anthropic_response
};

#[test]
fn test_anthropic_to_openrouter_thinking_conversion() {
    // Create Anthropic request with thinking content
    let thinking_content = ThinkingContent::new("I need to carefully consider the mathematical operation. Let me think step by step about this calculation.".to_string());
    let mut thinking_with_signature = thinking_content.clone();
    thinking_with_signature.signature = Some("thinking-step-1".to_string());
    
    let anthropic_request = AnthropicRequest::builder()
        .model("anthropic/claude-3-5-sonnet")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::Contents(vec![AnthropicContent::Text(
                    anthropic_ox::message::Text::new("What is 15 * 23?".to_string())
                )]),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: StringOrContents::Contents(vec![
                    AnthropicContent::Thinking(thinking_with_signature),
                    AnthropicContent::Text(
                        anthropic_ox::message::Text::new("Let me calculate 15 * 23 = 345".to_string())
                    ),
                ]),
            },
        ])
        .build();

    // Convert to OpenRouter
    let openrouter_request = anthropic_to_openrouter_request(anthropic_request).unwrap();
    
    // Verify reasoning is enabled
    assert_eq!(openrouter_request.include_reasoning, Some(true));
    
    // Verify thinking content was converted to text in messages
    assert_eq!(openrouter_request.messages.len(), 2);
    if let OpenRouterMessage::Assistant(assistant_msg) = &openrouter_request.messages[1] {
        assert_eq!(assistant_msg.content.len(), 2);
        // First part should be the thinking text
        if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
            assert!(text_content.text.contains("I need to carefully consider"));
        } else {
            panic!("Expected text content part");
        }
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_openrouter_to_anthropic_thinking_conversion() {
    // Create OpenRouter response with reasoning
    let reasoning_detail = ReasoningDetail {
        detail_type: "reasoning.text".to_string(),
        text: "Let me work through this step by step. 15 × 23 means I need to multiply these two numbers. I can break this down: 15 × 20 = 300, and 15 × 3 = 45. So 300 + 45 = 345.".to_string(),
        format: Some("unknown".to_string()),
        index: Some(0),
    };

    let assistant_msg = AssistantMessage::text("Let me calculate 15 × 23 = 345");
    
    let choice = Choice {
        index: 0,
        message: assistant_msg,
        logprobs: None,
        finish_reason: OpenRouterFinishReason::Stop,
        native_finish_reason: None,
        reasoning: Some("Step-by-step calculation of multiplication".to_string()),
        reasoning_details: Some(vec![reasoning_detail]),
    };

    let openrouter_response = OpenRouterResponse {
        id: "gen-test-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "deepseek/deepseek-chat-v3.1".to_string(),
        choices: vec![choice],
        system_fingerprint: None,
        usage: OpenRouterUsage {
            prompt_tokens: 20,
            completion_tokens: 150,
            total_tokens: 170,
        },
    };

    // Convert to Anthropic
    let anthropic_response = openrouter_to_anthropic_response(openrouter_response).unwrap();
    
    // Verify thinking content was created
    assert_eq!(anthropic_response.content.len(), 2);
    
    // First content should be thinking
    if let AnthropicContent::Thinking(thinking) = &anthropic_response.content[0] {
        assert_eq!(thinking.text, "Let me work through this step by step. 15 × 23 means I need to multiply these two numbers. I can break this down: 15 × 20 = 300, and 15 × 3 = 45. So 300 + 45 = 345.");
    } else {
        panic!("Expected thinking content, got: {:?}", anthropic_response.content[0]);
    }
    
    // Second content should be regular text
    if let AnthropicContent::Text(text) = &anthropic_response.content[1] {
        assert_eq!(text.text, "Let me calculate 15 × 23 = 345");
    } else {
        panic!("Expected text content, got: {:?}", anthropic_response.content[1]);
    }
}

#[test]
fn test_anthropic_to_openrouter_to_anthropic_round_trip() {
    // Create original Anthropic request with thinking
    let original_thinking = ThinkingContent::new("This is a complex reasoning process that I need to work through carefully.".to_string());
    let mut thinking_with_signature = original_thinking.clone();
    thinking_with_signature.signature = Some("reasoning-001".to_string());

    let original_request = AnthropicRequest::builder()
        .model("anthropic/claude-3-5-sonnet")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::Contents(vec![AnthropicContent::Text(
                    anthropic_ox::message::Text::new("Solve this problem step by step.".to_string())
                )]),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: StringOrContents::Contents(vec![
                    AnthropicContent::Thinking(thinking_with_signature),
                    AnthropicContent::Text(
                        anthropic_ox::message::Text::new("Here is my solution to the problem.".to_string())
                    ),
                ]),
            },
        ])
        .build();

    // Round trip: Anthropic -> OpenRouter -> simulated response -> back to Anthropic
    let openrouter_request = anthropic_to_openrouter_request(original_request).unwrap();
    
    // Simulate OpenRouter response with reasoning
    let reasoning_detail = ReasoningDetail {
        detail_type: "reasoning.text".to_string(),
        text: "This is a complex reasoning process that I need to work through carefully.".to_string(),
        format: Some("unknown".to_string()),
        index: Some(0),
    };

    let choice = Choice {
        index: 0,
        message: AssistantMessage::text("Here is my solution to the problem."),
        logprobs: None,
        finish_reason: OpenRouterFinishReason::Stop,
        native_finish_reason: None,
        reasoning: Some("This is a complex reasoning process that I need to work through carefully.".to_string()),
        reasoning_details: Some(vec![reasoning_detail]),
    };

    let openrouter_response = OpenRouterResponse {
        id: "gen-round-trip-test".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "deepseek/deepseek-chat-v3.1".to_string(),
        choices: vec![choice],
        system_fingerprint: None,
        usage: OpenRouterUsage {
            prompt_tokens: 25,
            completion_tokens: 100,
            total_tokens: 125,
        },
    };

    // Convert back to Anthropic
    let final_response = openrouter_to_anthropic_response(openrouter_response).unwrap();
    
    // Verify round-trip preserved thinking content
    assert_eq!(final_response.content.len(), 2);
    
    if let AnthropicContent::Thinking(thinking) = &final_response.content[0] {
        assert_eq!(thinking.text, "This is a complex reasoning process that I need to work through carefully.");
        // Note: signature is not preserved through OpenRouter round-trip
    } else {
        panic!("Expected thinking content in round-trip result");
    }
    
    if let AnthropicContent::Text(text) = &final_response.content[1] {
        assert_eq!(text.text, "Here is my solution to the problem.");
    } else {
        panic!("Expected text content in round-trip result");
    }
}

#[test]
fn test_openrouter_to_anthropic_to_openrouter_round_trip() {
    // Create original OpenRouter response with reasoning
    let original_reasoning_detail = ReasoningDetail {
        detail_type: "reasoning.text".to_string(),
        text: "I need to analyze this request carefully and provide a thoughtful response.".to_string(),
        format: Some("unknown".to_string()),
        index: Some(0),
    };

    let original_choice = Choice {
        index: 0,
        message: AssistantMessage::text("Based on my analysis, here is the answer."),
        logprobs: None,
        finish_reason: OpenRouterFinishReason::Stop,
        native_finish_reason: None,
        reasoning: Some("I need to analyze this request carefully and provide a thoughtful response.".to_string()),
        reasoning_details: Some(vec![original_reasoning_detail]),
    };

    let original_response = OpenRouterResponse {
        id: "gen-original-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "google/gemini-2.5-flash".to_string(),
        choices: vec![original_choice],
        system_fingerprint: None,
        usage: OpenRouterUsage {
            prompt_tokens: 30,
            completion_tokens: 80,
            total_tokens: 110,
        },
    };

    // Round trip: OpenRouter -> Anthropic -> back to OpenRouter
    let anthropic_response = openrouter_to_anthropic_response(original_response).unwrap();
    
    // Convert back to OpenRouter request (simulating the flow)
    let anthropic_request = AnthropicRequest::builder()
        .model("google/gemini-2.5-flash")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::Contents(vec![AnthropicContent::Text(
                    anthropic_ox::message::Text::new("Please analyze this request.".to_string())
                )]),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: StringOrContents::Contents(anthropic_response.content),
            },
        ])
        .build();

    let final_openrouter_request = anthropic_to_openrouter_request(anthropic_request).unwrap();
    
    // Verify reasoning was preserved through round-trip
    assert_eq!(final_openrouter_request.include_reasoning, Some(true));
    
    // Verify thinking content was converted to text and preserved
    assert_eq!(final_openrouter_request.messages.len(), 2);
    if let OpenRouterMessage::Assistant(assistant_msg) = &final_openrouter_request.messages[1] {
        assert_eq!(assistant_msg.content.len(), 2);
        
        // First part should be the thinking text
        if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
            assert_eq!(text_content.text, "I need to analyze this request carefully and provide a thoughtful response.");
        } else {
            panic!("Expected thinking text content part");
        }
        
        // Second part should be the regular response
        if let ContentPart::Text(text_content) = &assistant_msg.content[1] {
            assert_eq!(text_content.text, "Based on my analysis, here is the answer.");
        } else {
            panic!("Expected regular text content part");
        }
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_full_round_trip_thinking_preservation() {
    // Test complete round-trip: Anthropic -> OpenRouter -> Response -> back to Anthropic
    let original_thinking = ThinkingContent::new("Let me think about this problem systematically. First, I need to understand what is being asked.".to_string());
    let mut thinking_with_signature = original_thinking.clone();
    thinking_with_signature.signature = Some("systematic-analysis".to_string());

    let original_request = AnthropicRequest::builder()
        .model("deepseek/deepseek-chat-v3.1")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::Contents(vec![AnthropicContent::Text(
                    anthropic_ox::message::Text::new("Explain quantum computing.".to_string())
                )]),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: StringOrContents::Contents(vec![
                    AnthropicContent::Thinking(thinking_with_signature),
                    AnthropicContent::Text(
                        anthropic_ox::message::Text::new("Quantum computing uses quantum mechanics principles to process information.".to_string())
                    ),
                ]),
            },
        ])
        .build();

    // Convert to OpenRouter and verify reasoning is enabled
    let openrouter_request = anthropic_to_openrouter_request(original_request).unwrap();
    assert_eq!(openrouter_request.include_reasoning, Some(true));

    // Simulate OpenRouter response with reasoning matching the thinking content
    let choice = Choice {
        index: 0,
        message: AssistantMessage::text("Quantum computing uses quantum mechanics principles to process information."),
        logprobs: None,
        finish_reason: OpenRouterFinishReason::Stop,
        native_finish_reason: None,
        reasoning: Some("Let me think about this problem systematically. First, I need to understand what is being asked.".to_string()),
        reasoning_details: Some(vec![ReasoningDetail {
            detail_type: "reasoning.text".to_string(),
            text: "Let me think about this problem systematically. First, I need to understand what is being asked.".to_string(),
            format: Some("unknown".to_string()),
            index: Some(0),
        }]),
    };

    let openrouter_response = OpenRouterResponse {
        id: "gen-full-round-trip".to_string(),
        object: "chat.completion".to_string(),
        created: 1234567890,
        model: "deepseek/deepseek-chat-v3.1".to_string(),
        choices: vec![choice],
        system_fingerprint: None,
        usage: OpenRouterUsage {
            prompt_tokens: 15,
            completion_tokens: 120,
            total_tokens: 135,
        },
    };

    // Convert back to Anthropic
    let final_response = openrouter_to_anthropic_response(openrouter_response).unwrap();

    // Verify complete preservation of thinking and text content
    assert_eq!(final_response.content.len(), 2);
    
    // Check thinking content was preserved
    if let AnthropicContent::Thinking(thinking) = &final_response.content[0] {
        assert_eq!(thinking.text, "Let me think about this problem systematically. First, I need to understand what is being asked.");
        // Note: signature is lost in OpenRouter round-trip (expected limitation)
    } else {
        panic!("Expected thinking content, got: {:?}", final_response.content[0]);
    }
    
    // Check regular content was preserved
    if let AnthropicContent::Text(text) = &final_response.content[1] {
        assert_eq!(text.text, "Quantum computing uses quantum mechanics principles to process information.");
    } else {
        panic!("Expected text content, got: {:?}", final_response.content[1]);
    }
}