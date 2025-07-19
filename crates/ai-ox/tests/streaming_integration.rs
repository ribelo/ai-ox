mod common;

#[cfg(feature = "openrouter")]
use ai_ox::model::openrouter::OpenRouterModel;
use ai_ox::{
    agent::{Agent, events::AgentEvent},
    content::{Message, MessageRole, Part, delta::StreamEvent},
    model::gemini::GeminiModel,
    toolbox,
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

/// Setup function that returns a vector of (provider_name, agent) pairs
/// for all available providers based on environment variables
async fn setup_agents() -> Vec<(String, Agent)> {
    let mut agents = Vec::new();

    // Try to setup Gemini (only if feature is enabled)
    #[cfg(feature = "gemini")]
    {
        if let Ok(api_key) =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        {
            let model = GeminiModel::builder()
                .api_key(api_key)
                .model("gemini-1.5-flash".to_string())
                .build();

            let weather_service = MockWeatherService;
            let agent = Agent::model(model)
                .tools(weather_service)
                .system_instruction("You are a helpful assistant. When asked about weather, you MUST use the get_weather function.")
                .max_iterations(5)
                .build();

            agents.push(("Gemini".to_string(), agent));
        }
    }

    // Try to setup OpenRouter (only if feature is enabled)
    #[cfg(feature = "openrouter")]
    {
        if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
            let model = OpenRouterModel::builder().model("google/gemini-2.5-flash")
                .api_key(api_key)
                .build();

            let weather_service = MockWeatherService;
            let agent = Agent::model(model)
                .tools(weather_service)
                .system_instruction("You are a helpful assistant. When asked about weather, you MUST use the get_weather function.")
                .max_iterations(5)
                .build();

            agents.push(("OpenRouter".to_string(), agent));
        }
    }

    agents
}

/// Setup function for agents without tools
async fn setup_agents_without_tools() -> Vec<(String, Agent)> {
    let mut agents = Vec::new();

    // Try to setup Gemini
    if let Ok(api_key) =
        std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
    {
        let model = GeminiModel::builder()
            .api_key(api_key)
            .model("gemini-2.5-flash")
            .build();

        let agent = Agent::model(model)
            .system_instruction("You are a helpful assistant. Keep responses brief.")
            .build();

        agents.push(("Gemini".to_string(), agent));
    }

    // Try to setup OpenRouter (only if feature is enabled)
    #[cfg(feature = "openrouter")]
    {
        if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
            let model = OpenRouterModel::builder()
                .api_key(api_key)
                .model("google/gemini-2.5-flash")
                .build();

            let agent = Agent::model(model)
                .system_instruction("You are a helpful assistant. Keep responses brief.")
                .build();

            agents.push(("OpenRouter".to_string(), agent));
        }
    }

    agents
}

#[tokio::test]
#[ignore = "Requires API keys and makes actual API calls"]
async fn test_streaming_with_tools() {
    let agents = setup_agents().await;

    if agents.is_empty() {
        println!("Skipping streaming with tools test - no API keys available");
        return;
    }

    for (provider_name, agent) in agents {
        println!("\n--- Testing provider: {provider_name} ---");
        println!("Testing complete streaming multi-turn conversation with tools");

        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "What's the weather like in Tokyo? Please use the weather function to get current conditions.".to_string(),
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

        println!("Starting streaming test...");

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
                AgentEvent::ToolResult(tool_result) => {
                    if tool_result.name == "get_weather" {
                        tool_result_received = true;
                        println!("\n✓ Tool result event received: get_weather");
                        println!("  Tool result ID: {}", tool_result.id);
                        println!("  Response messages: {}", tool_result.response.len());

                        // Verify the tool result has proper structure
                        assert!(
                            !tool_result.name.is_empty(),
                            "Tool result should have a name"
                        );
                        assert!(
                            !tool_result.response.is_empty(),
                            "Tool result should have response messages"
                        );
                    } else {
                        println!("⚠ Unexpected tool result: {}", tool_result.name);
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

        println!("\n\nStreaming test completed for {provider_name}!");
        println!("Total events received: {}", events.len());

        // Assert the full sequence of events
        assert!(started_received, "Must receive Started event");
        assert!(
            delta_received,
            "Must receive at least one Delta event with text"
        );
        assert!(
            tool_execution_received,
            "Must receive ToolExecution event for get_weather"
        );
        assert!(
            tool_result_received,
            "Must receive ToolResult event for get_weather"
        );
        assert!(completed_received, "Must receive Completed event");

        // Verify the event sequence makes sense
        let event_types: Vec<&str> = events.iter().map(|e| e.event_type()).collect();
        println!("Event sequence: {event_types:?}");

        // First event should be Started
        assert_eq!(events.first().unwrap().event_type(), "Started");
        // Last event should be Completed
        assert_eq!(events.last().unwrap().event_type(), "Completed");

        // Should have tool execution and result events
        let tool_execution_count = event_types
            .iter()
            .filter(|&&t| t == "ToolExecution")
            .count();
        let tool_result_count = event_types.iter().filter(|&&t| t == "ToolResult").count();
        assert!(
            tool_execution_count > 0,
            "Should have at least one tool execution"
        );
        assert!(
            tool_result_count > 0,
            "Should have at least one tool result"
        );
        assert_eq!(
            tool_execution_count, tool_result_count,
            "Tool executions and results should match"
        );

        println!(
            "🎉 All assertions passed for {provider_name}! The streaming multi-turn conversation works correctly."
        );
    }
}

#[tokio::test]
#[ignore = "Requires API keys and makes actual API calls"]
async fn test_streaming_without_tools() {
    let agents = setup_agents_without_tools().await;

    if agents.is_empty() {
        println!("Skipping streaming without tools test - no API keys available");
        return;
    }

    for (provider_name, agent) in agents {
        println!("\n--- Testing provider: {provider_name} ---");
        println!("Testing streaming without tools (simple conversation)");

        let messages = vec![Message::new(
            MessageRole::User,
            vec![Part::Text {
                text: "Write a haiku about coding. Make it creative.".to_string(),
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

        println!("\n✓ Simple streaming conversation works correctly for {provider_name}");
    }
}
