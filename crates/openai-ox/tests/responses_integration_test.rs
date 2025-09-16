use openai_ox::{
    OpenAI, ResponseOutputContent, ResponseOutputItem, ResponsesInput, ResponsesRequest,
    ResponsesResponse, responses::response::add_output_text,
};
use serde_json::json;
use std::env;

#[tokio::test]
async fn test_responses_api_direct_call() {
    // Skip test if no API key is set
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Skipping test: OPENAI_API_KEY not set");
            return;
        }
    };

    let client = OpenAI::new(api_key);

    // Create a minimal request
    let request = ResponsesRequest::builder()
        .model("gpt-5")
        .input(ResponsesInput::Messages(vec![
            ai_ox_common::openai_format::Message {
                role: ai_ox_common::openai_format::MessageRole::User,
                content: Some("Say 'test successful'".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ]))
        .max_output_tokens(50)
        .store(false)
        .stream(false)
        .build();

    // Add optional fields
    let mut request = request;
    request.instructions = Some("You are a test assistant".to_string());

    // Send the request
    let response = client.send_responses(&request).await;

    match response {
        Ok(resp) => {
            println!("Response ID: {}", resp.id);
            println!("Model: {}", resp.model);
            println!("Status: {:?}", resp.status);
            println!("Output text: {}", resp.output_text);

            // Verify the response structure
            assert!(!resp.id.is_empty(), "Response ID should not be empty");
            assert_eq!(resp.object, "response", "Object type should be 'response'");
            assert!(resp.created_at > 0, "Created timestamp should be positive");

            // Verify output_text was generated from output items
            if !resp.output.is_empty() {
                assert!(
                    !resp.output_text.is_empty()
                        || resp
                            .output
                            .iter()
                            .all(|item| { !matches!(item, ResponseOutputItem::Message { .. }) }),
                    "output_text should be generated from message content"
                );
            }

            // Check for message output
            let has_message = resp
                .output
                .iter()
                .any(|item| matches!(item, ResponseOutputItem::Message { .. }));

            let has_reasoning = resp
                .output
                .iter()
                .any(|item| matches!(item, ResponseOutputItem::Reasoning { .. }));

            println!("Has message output: {}", has_message);
            println!("Has reasoning output: {}", has_reasoning);

            // Verify usage stats if present
            if let Some(usage) = &resp.usage {
                assert!(usage.input_tokens > 0, "Input tokens should be positive");
                assert!(
                    usage.total_tokens >= usage.input_tokens,
                    "Total tokens should be at least input tokens"
                );
            }
        }
        Err(e) => {
            panic!("API request failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_response_deserialization() {
    // Test with actual API response structure
    let json_response = json!({
        "id": "resp_test123",
        "object": "response",
        "created_at": 1757159792,
        "status": "completed",
        "background": false,
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": 50,
        "max_tool_calls": null,
        "model": "gpt-5-2025-08-07",
        "output": [
            {
                "id": "rs_test",
                "type": "reasoning",
                "summary": []
            },
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello!",
                        "annotations": []
                    }
                ]
            }
        ],
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "prompt_cache_key": null,
        "reasoning": {
            "effort": "medium",
            "summary": null
        },
        "safety_identifier": null,
        "service_tier": "default",
        "store": false,
        "temperature": 1.0,
        "text": {
            "format": {
                "type": "text"
            },
            "verbosity": "medium"
        },
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 8,
            "input_tokens_details": {
                "cached_tokens": 0
            },
            "output_tokens": 10,
            "output_tokens_details": {
                "reasoning_tokens": 0
            },
            "total_tokens": 18
        },
        "user": null,
        "metadata": {}
    });

    // Deserialize the response
    let mut response: ResponsesResponse =
        serde_json::from_value(json_response).expect("Failed to deserialize response");

    // Add output_text like the SDK does
    add_output_text(&mut response);

    // Verify the response
    assert_eq!(response.id, "resp_test123");
    assert_eq!(response.object, "response");
    assert_eq!(response.model, "gpt-5-2025-08-07");
    assert_eq!(response.output_text, "Hello!");

    // Check output items
    assert_eq!(response.output.len(), 2);

    // Verify reasoning item
    if let ResponseOutputItem::Reasoning { id, summary, .. } = &response.output[0] {
        assert_eq!(id, "rs_test");
        assert!(summary.is_empty());
    } else {
        panic!("First output item should be reasoning");
    }

    // Verify message item
    if let ResponseOutputItem::Message {
        id,
        content,
        role,
        status,
    } = &response.output[1]
    {
        assert_eq!(id, "msg_test");
        assert_eq!(role, "assistant");
        assert_eq!(status, "completed");
        assert_eq!(content.len(), 1);

        if let ResponseOutputContent::Text { text, .. } = &content[0] {
            assert_eq!(text, "Hello!");
        } else {
            panic!("Message content should be text");
        }
    } else {
        panic!("Second output item should be message");
    }

    // Verify usage
    assert!(response.usage.is_some());
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 10);
    assert_eq!(usage.total_tokens, 18);
}

// Roundtrip test would require anthropic-ox and conversion-ox as test dependencies
// For now, we'll focus on the OpenAI-only tests
#[tokio::test]
#[ignore] // Skip this test as it requires additional dependencies
async fn test_anthropic_openai_roundtrip() {
    // This test requires anthropic-ox and conversion-ox as dependencies
    // It would test the full roundtrip conversion
    // Placeholder for actual implementation
    println!("Roundtrip test would go here");
}
