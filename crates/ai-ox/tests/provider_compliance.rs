mod common;

use ai_ox::{
    content::{
        delta::StreamEvent,
        message::{Message, MessageRole},
        part::Part,
    },
    model::request::ModelRequest,
    tool::{FunctionMetadata, Tool},
};
use futures_util::StreamExt;
use serde_json::json;
use std::collections::BTreeMap;

/// Test that all providers can handle a simple request/response interaction.
///
/// This test validates the fundamental capability that every Model implementation
/// must have: receiving a simple text request and returning a coherent response.
#[tokio::test]
async fn test_all_providers_simple_request() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping compliance test.");
        return;
    }

    for model in models {
        println!("Testing simple request with model: {}", model.name());

        let message = Message { ext: Some(BTreeMap::new()),
            role: MessageRole::User,
            content: vec![Part::Text { ext: BTreeMap::new(),
                text: "Why is the sky blue? Please respond in exactly one sentence.".to_string(),
            }],
            timestamp: Some(chrono::Utc::now()),
        };

        let request = ModelRequest {
            messages: vec![message],
            system_message: None,
            tools: None,
        };

        let result = model.request(request).await;

        assert!(result.is_ok(), "Model {} failed simple request: {:?}", model.name(), result.err());

        let response = result.unwrap();
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty(), "Model {} returned empty content", model.name());

        // Verify we got text content
        let has_text = response.message.content.iter().any(|part| {
            matches!(part, Part::Text { text, .. } if !text.is_empty())
        });
        assert!(has_text, "Model {} did not return any text content", model.name());

        // Verify usage data is reported
        assert!(response.usage.input_tokens() > 0, "Model {} reported zero input tokens", model.name());
        assert!(response.usage.output_tokens() > 0, "Model {} reported zero output tokens", model.name());

        println!("✅ Model {} passed simple request test", model.name());
    }
}

/// Test that all providers can handle streaming responses correctly.
///
/// This test validates that streaming works consistently across all providers,
/// yielding text deltas and properly terminating with a StreamStop event.
#[tokio::test]
async fn test_all_providers_simple_streaming() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping streaming compliance test.");
        return;
    }

    for model in models {
        println!("Testing streaming with model: {}", model.name());

        let message = Message { ext: Some(BTreeMap::new()),
            role: MessageRole::User,
            content: vec![Part::Text { ext: BTreeMap::new(),
                text: "Count from 1 to 5, one number per line.".to_string(),
            }],
            timestamp: Some(chrono::Utc::now()),
        };

        let request = ModelRequest {
            messages: vec![message],
            system_message: None,
            tools: None,
        };

        let mut stream = model.request_stream(request);
        let mut events = Vec::new();
        let mut received_text_delta = false;
        let mut received_stream_stop = false;

        while let Some(event_result) = stream.next().await {
            let event = event_result.expect(&format!("Model {} stream error", model.name()));

            match &event {
                StreamEvent::TextDelta(text) => {
                    assert!(!text.is_empty(), "Model {} yielded empty text delta", model.name());
                    received_text_delta = true;
                }
                StreamEvent::StreamStop(_) => {
                    received_stream_stop = true;
                    events.push(event);
                    break; // Stream should end after StreamStop
                }
                _ => {}
            }

            events.push(event);
        }

        assert!(received_text_delta, "Model {} did not yield any TextDelta events", model.name());
        assert!(received_stream_stop, "Model {} did not yield StreamStop event", model.name());

        // Verify last event is StreamStop
        assert!(
            matches!(events.last(), Some(StreamEvent::StreamStop(_))),
            "Model {} stream did not end with StreamStop",
            model.name()
        );

        println!("✅ Model {} passed streaming test", model.name());
    }
}

/// Test that all providers can handle tool use correctly.
///
/// This test validates that providers can process tool definitions, recognize
/// when to call tools, and return properly formatted tool calls.
#[tokio::test]
async fn test_all_providers_tool_use() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping tool use compliance test.");
        return;
    }

    // Define a simple weather tool
    let weather_tool = Tool::FunctionDeclarations(vec![FunctionMetadata {
        name: "get_weather".to_string(),
        description: Some("Get the current weather for a location".to_string()),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state/country, e.g. 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }),
    }]);

    for model in models {
        println!("Testing tool use with model: {}", model.name());

        let message = Message { ext: Some(BTreeMap::new()),
            role: MessageRole::User,
            content: vec![Part::Text { ext: BTreeMap::new(),
                text: "What's the weather like in Tokyo? Use the weather tool.".to_string(),
            }],
            timestamp: Some(chrono::Utc::now()),
        };

        let request = ModelRequest {
            messages: vec![message],
            system_message: None,
            tools: Some(vec![weather_tool.clone()]),
        };

        let result = model.request(request).await;

        assert!(result.is_ok(), "Model {} failed tool use request: {:?}", model.name(), result.err());

        let response = result.unwrap();


        // Check if the response contains a tool call. This must not be lenient.
        let has_tool_call = response.message.content.iter().any(|part| {
            matches!(part, Part::ToolUse { name, .. } if name == "get_weather")
        });

        assert!(has_tool_call, "Model {} failed to call the 'get_weather' tool when explicitly instructed.", model.name());

        println!("✅ Model {} passed tool use test", model.name());
    }
}
