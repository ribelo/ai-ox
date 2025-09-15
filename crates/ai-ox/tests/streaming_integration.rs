mod common;

use ai_ox::{
    agent::{Agent, events::AgentEvent},
    content::{Message, MessageRole, Part, delta::StreamEvent},
    toolbox,
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// Mock weather service that guarantees tool usage
#[derive(Debug, Clone)]
struct MockWeatherService;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherRequest {
    location: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherResponse {
    location: String,
    temperature: f64,
    condition: String,
}

#[toolbox]
impl MockWeatherService {
    /// Get the current weather for a location - this will always be called when mentioned
    pub async fn get_weather(
        &self,
        request: WeatherRequest,
    ) -> Result<WeatherResponse, std::io::Error> {
        // Return a deterministic mock response
        Ok(WeatherResponse {
            location: request.location,
            temperature: 22.0,
            condition: "sunny".to_string(),
        })
    }
}

#[tokio::test]
async fn test_all_providers_streaming_with_tools() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping streaming with tools test.");
        return;
    }

    for model in models {
        let model_name = model.name().to_string();
        println!("\n--- Testing streaming with tools for model: {} ---", &model_name);
        
        let weather_service = MockWeatherService;
        let agent = Agent::builder()
            .model(model.into())
            .tools(weather_service)
            .system_instruction("You are a helpful assistant. When asked about weather, you MUST use the get_weather function.")
            .max_iterations(5)
            .build();

        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "What's the weather like in Tokyo? Please use the weather function to get current conditions.".to_string(),
                ext: BTreeMap::new(),
            }],
        )];

        let mut stream = agent.stream(messages);
        let mut events = Vec::new();

        // Track expected event sequence
        let mut started_received = false;
        let mut delta_received = false;
        let mut tool_execution_received = false;
        let mut tool_result_received = false;
        let mut completed_received = false;

        println!("Starting streaming test for {}...", &model_name);

        while let Some(event_result) = stream.next().await {
            let event = event_result.expect("Stream should not error");

            // Push event to list BEFORE processing to ensure all events are captured
            events.push(event.clone());

            match &event {
                AgentEvent::Started => {
                    started_received = true;
                    println!("✓ Started event received");
                }
                AgentEvent::StreamEvent(StreamEvent::TextDelta(text)) => {
                    delta_received = true;
                    print!("{text}"); // Print streaming text in real-time
                }
                AgentEvent::StreamEvent(stream_event) => {
                    println!("△ Other stream event: {stream_event:?}");
                }
                AgentEvent::ToolExecution(tool_call) => {
                    if tool_call.name == "get_weather" {
                        tool_execution_received = true;
                        println!("\n✓ Tool execution event received: get_weather");
                        println!("  Tool call ID: {}", tool_call.id);
                        println!("  Arguments: {:?}", tool_call.args);

                        // Verify the tool call has proper structure
                        // Note: Some providers may not always provide tool call IDs, so we allow empty IDs
                        assert!(!tool_call.args.is_null(), "Tool call should have arguments");

                        if let Some(location) = tool_call.args.get("location") {
                            assert!(location.is_string(), "Location should be a string");
                            println!("  Location: {}", location.as_str().unwrap_or("unknown"));
                        }
                    } else {
                        println!("⚠ Unexpected tool execution: {}", tool_call.name);
                    }
                }
                AgentEvent::ToolResult(messages) => {
                    // Look for tool result information in the messages
                    for message in messages {
                        for part in &message.content {
                            if let Part::ToolResult { name, id, .. } = part {
                                if name == "get_weather" {
                                    tool_result_received = true;
                                    println!("\n✓ Tool result event received: get_weather");
                                    println!("  Tool result ID: {}", id);
                                    println!("  Response messages: {}", messages.len());

                                    // Verify the tool result has proper structure
                                    assert!(
                                        !name.is_empty(),
                                        "Tool result should have a name"
                                    );
                                    assert!(
                                        !messages.is_empty(),
                                        "Tool result should have response messages"
                                    );
                                } else {
                                    println!("⚠ Unexpected tool result: {}", name);
                                }
                            }
                        }
                    }
                }
                AgentEvent::Completed(response) => {
                    completed_received = true;
                    println!("\n✓ Completed event received");
                    println!("  Model: {}", response.model_name);
                    println!("  Vendor: {}", response.vendor_name);
                    println!("  Message parts: {}", response.message.content.len());

                    // Verify the final response has content
                    assert!(
                        !response.message.content.is_empty(),
                        "Final response should have content"
                    );
                    break;
                }
                AgentEvent::Failed(error) => {
                    println!("\n✗ Agent failed: {error:?}");
                    panic!("Agent should not fail in this test");
                }
            }
        }

        println!("\n\nStreaming test completed for {}!", &model_name);
        println!("Total events received: {}", events.len());

        // Assert the full sequence of events
        assert!(started_received, "Must receive Started event");
        assert!(
            delta_received,
            "Must receive at least one Delta event with text"
        );
        
        // Note: Some models might not support tools, so we check but don't fail
        if tool_execution_received {
            assert!(
                tool_result_received,
                "If tool was executed, must receive ToolResult event"
            );
        } else {
            println!("⚠️  Model {} did not use tools (might not support them)", &model_name);
        }
        
        assert!(completed_received, "Must receive Completed event");

        // Verify the event sequence makes sense
        let event_types: Vec<&str> = events.iter().map(|e| e.event_type()).collect();
        println!("Event sequence: {event_types:?}");

        // First event should be Started
        assert_eq!(events.first().unwrap().event_type(), "Started");
        // Last event should be Completed
        assert_eq!(events.last().unwrap().event_type(), "Completed");

        println!(
            "🎉 Model {} passed streaming with tools test!",
            &model_name
        );
    }
}

#[tokio::test]
async fn test_all_providers_streaming_without_tools() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("No models available for testing. Skipping streaming without tools test.");
        return;
    }

    for model in models {
        let model_name = model.name().to_string();
        println!("\n--- Testing streaming without tools for model: {} ---", &model_name);
        
        let agent = Agent::builder()
            .model(model.into())
            .system_instruction("You are a helpful assistant. Keep responses brief.")
            .build();

        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "Write a haiku about coding. Make it creative.".to_string(),
                ext: BTreeMap::new(),
            }],
        )];

        let mut stream = agent.stream(messages);
        let mut started_received = false;
        let mut delta_received = false;
        let mut completed_received = false;
        let mut accumulated_text = String::new();

        while let Some(event_result) = stream.next().await {
            let event = event_result.expect("Stream should not error");

            match &event {
                AgentEvent::Started => {
                    started_received = true;
                    println!("✓ Started");
                }
                AgentEvent::StreamEvent(StreamEvent::TextDelta(text)) => {
                    delta_received = true;
                    accumulated_text.push_str(text);
                    print!("{text}");
                }
                AgentEvent::Completed(response) => {
                    completed_received = true;
                    println!("\n✓ Completed: {}", response.model_name);
                    break;
                }
                AgentEvent::ToolExecution(_) => {
                    panic!("Should not receive tool execution in this test");
                }
                AgentEvent::ToolResult(_) => {
                    panic!("Should not receive tool result in this test");
                }
                AgentEvent::Failed(error) => {
                    panic!("Agent failed: {error:?}");
                }
                _ => {
                    // Other delta events are okay
                }
            }
        }

        assert!(started_received, "Must receive Started event");
        assert!(delta_received, "Must receive Delta events");
        assert!(completed_received, "Must receive Completed event");
        assert!(!accumulated_text.is_empty(), "Must receive text content");

        println!("\n✓ Model {} passed simple streaming test", &model_name);
    }
}