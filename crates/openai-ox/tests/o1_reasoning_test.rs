//! Test OpenAI o1 model reasoning token handling

use openai_ox::{ChatResponse, Usage};
use serde_json::json;

#[test]
fn test_o1_reasoning_response_deserialization() {
    // Real o1-mini response with reasoning tokens
    let response_json = json!({
        "id": "chatcmpl-CAFfu0W7hpbf08cMQod4EdHHPDccU",
        "object": "chat.completion",
        "created": 1756559570,
        "model": "o1-mini-2024-09-12",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sure, let's break down the problem step by step:\n\n1. **Total Number of Sheep:**  \n   The farmer initially has **17 sheep**.\n\n2. **Understanding \"All but 9 Die\":**  \n   The phrase \"all but 9 die\" means that **all the sheep except for 9 die**. In other words, **9 sheep survive**.\n\n3. **Calculating the Number of Sheep Left:**  \n   Since 9 sheep survive, the number of sheep **left alive** is **9**.\n\n**Conclusion:**  \nThe farmer has **9 sheep left**.",
                    "refusal": null,
                    "annotations": []
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 33,
            "completion_tokens": 388,
            "total_tokens": 421,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 256,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "service_tier": "default",
        "system_fingerprint": "fp_79455e3cfb"
    });

    // Deserialize the response
    let response: ChatResponse =
        serde_json::from_value(response_json).expect("Failed to deserialize response");

    // Basic response validation
    assert_eq!(response.id, "chatcmpl-CAFfu0W7hpbf08cMQod4EdHHPDccU");
    assert_eq!(response.model, "o1-mini-2024-09-12");
    assert_eq!(response.choices.len(), 1);

    // Validate message content
    let choice = &response.choices[0];
    assert_eq!(choice.index, 0);
    assert_eq!(choice.finish_reason, Some("stop".to_string()));
    assert!(choice.message.content.is_some());
    assert!(
        choice
            .message
            .content
            .as_ref()
            .unwrap()
            .contains("9 sheep left")
    );

    // Validate usage statistics
    let usage = response.usage.expect("Usage should be present");
    assert_eq!(usage.prompt_tokens, 33);
    assert_eq!(usage.completion_tokens, 388);
    assert_eq!(usage.total_tokens, 421);

    // Validate reasoning token details - this is the key test
    let completion_details = usage
        .completion_tokens_details
        .expect("Completion tokens details should be present");
    assert_eq!(completion_details.reasoning_tokens, Some(256));
    assert_eq!(completion_details.audio_tokens, Some(0));

    // Validate prompt token details
    let prompt_details = usage
        .prompt_tokens_details
        .expect("Prompt tokens details should be present");
    assert_eq!(prompt_details.cached_tokens, Some(0));
    assert_eq!(prompt_details.audio_tokens, Some(0));
}

#[test]
fn test_o1_response_without_reasoning() {
    // Regular GPT response without reasoning tokens
    let response_json = json!({
        "id": "chatcmpl-regular",
        "object": "chat.completion",
        "created": 1756559570,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "total_tokens": 19
        }
    });

    let response: ChatResponse =
        serde_json::from_value(response_json).expect("Failed to deserialize response");

    let usage = response.usage.expect("Usage should be present");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 9);
    assert_eq!(usage.total_tokens, 19);

    // Details should be None for regular models
    assert!(usage.completion_tokens_details.is_none());
    assert!(usage.prompt_tokens_details.is_none());
}

#[test]
fn test_usage_helper_methods() {
    let usage_with_reasoning = Usage {
        prompt_tokens: 33,
        completion_tokens: 388,
        total_tokens: 421,
        prompt_tokens_details: Some(openai_ox::usage::PromptTokensDetails {
            cached_tokens: Some(10),
            audio_tokens: Some(0),
        }),
        completion_tokens_details: Some(openai_ox::usage::CompletionTokensDetails {
            reasoning_tokens: Some(256),
            audio_tokens: Some(0),
        }),
    };

    // Test helper methods
    assert!(usage_with_reasoning.is_cached()); // Has cached tokens
    assert_eq!(usage_with_reasoning.completion_ratio(), 388.0 / 33.0);

    // Test cost calculation (hypothetical pricing)
    let cost = usage_with_reasoning.calculate_cost(0.001, 0.002); // $0.001 per 1k prompt, $0.002 per 1k completion
    let expected_cost = (33.0 / 1000.0) * 0.001 + (388.0 / 1000.0) * 0.002;
    assert!((cost - expected_cost).abs() < f64::EPSILON);
}

#[test]
fn test_reasoning_token_access() {
    let mut usage = Usage::new(100, 200);

    // Initially no reasoning tokens
    assert!(usage.completion_tokens_details.is_none());

    // Add reasoning token details
    usage.completion_tokens_details = Some(openai_ox::usage::CompletionTokensDetails {
        reasoning_tokens: Some(50),
        audio_tokens: None,
    });

    // Verify reasoning tokens can be accessed
    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
        .unwrap_or(0);

    assert_eq!(reasoning_tokens, 50);
}

#[test]
fn test_usage_arithmetic() {
    let usage1 = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        prompt_tokens_details: Some(openai_ox::usage::PromptTokensDetails {
            cached_tokens: Some(5),
            audio_tokens: None,
        }),
        completion_tokens_details: Some(openai_ox::usage::CompletionTokensDetails {
            reasoning_tokens: Some(15),
            audio_tokens: None,
        }),
    };

    let usage2 = Usage {
        prompt_tokens: 5,
        completion_tokens: 10,
        total_tokens: 15,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    };

    let combined = usage1 + usage2;
    assert_eq!(combined.prompt_tokens, 15);
    assert_eq!(combined.completion_tokens, 30);
    assert_eq!(combined.total_tokens, 45);

    // Details are lost in arithmetic operations (as documented)
    assert!(combined.prompt_tokens_details.is_none());
    assert!(combined.completion_tokens_details.is_none());
}
