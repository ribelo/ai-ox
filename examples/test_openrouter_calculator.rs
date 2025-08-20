use ai_ox::{
    agent::{Agent, events::AgentEvent},
    content::{Message, MessageRole, Part},
    model::openrouter::OpenRouterModel,
    toolbox,
};
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Simple calculator tool
#[derive(Debug, Clone)]
struct Calculator;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AddRequest {
    a: i32,
    b: i32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AddResponse {
    result: i32,
    operation: String,
}

#[toolbox]
impl Calculator {
    /// Add two numbers together
    pub fn add(&self, request: AddRequest) -> Result<AddResponse, std::io::Error> {
        println!("Calculator: Adding {} + {}", request.a, request.b);
        Ok(AddResponse {
            result: request.a + request.b,
            operation: format!("{} + {} = {}", request.a, request.b, request.a + request.b),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing OpenRouter with calculator tool (should reproduce 400 error)...");
    
    let openrouter_api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY environment variable must be set");
    
    // Create OpenRouter model - use Claude which definitely supports tool calling
    let model = OpenRouterModel::builder()
        .api_key(openrouter_api_key)
        .model("anthropic/claude-3.5-sonnet")
        .build();
    
    // Create agent with calculator tool  
    let calculator = Calculator;
    let agent = Agent::builder()
        .model(std::sync::Arc::new(model))
        .tools(calculator)
        .system_instruction("You are a helpful assistant with access to a calculator. When asked to do math, use the add function.")
        .max_iterations(3)
        .build();
    
    // Ask the agent to calculate 2 + 2
    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Please calculate 2 + 2 using the calculator tool".to_string(),
        }],
    )];
    
    println!("Sending request: Calculate 2 + 2");
    
    let mut stream = agent.stream(messages);
    let mut tool_executed = false;
    let mut got_400_error = false;
    
    while let Some(event_result) = stream.next().await {
        match event_result {
            Ok(event) => {
                match &event {
                    AgentEvent::Started => {
                        println!("✓ Agent started");
                    }
                    AgentEvent::StreamEvent(stream_event) => {
                        println!("△ Stream event: {:?}", stream_event);
                    }
                    AgentEvent::ToolExecution(tool_call) => {
                        tool_executed = true;
                        println!("✓ Tool executed: {} with args {:?}", tool_call.name, tool_call.args);
                    }
                    AgentEvent::ToolResult(tool_result) => {
                        println!("✓ Tool result: {} -> {:?}", tool_result.name, tool_result.response);
                        println!("=== This is where the 400 error should occur after this event ===");
                    }
                    AgentEvent::Completed(response) => {
                        println!("✓ Completed successfully!");
                        println!("Response: {:?}", response.message);
                        break;
                    }
                    AgentEvent::Failed(error) => {
                        println!("✗ Agent failed: {}", error);
                        if error.contains("400") && error.contains("Provider returned error") {
                            got_400_error = true;
                            println!("🎯 REPRODUCED: OpenRouter 400 error when processing tool results!");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                println!("✗ Stream error: {}", e);
                if e.to_string().contains("400") {
                    got_400_error = true;
                    println!("🎯 REPRODUCED: OpenRouter 400 error!");
                }
                break;
            }
        }
    }
    
    if tool_executed && got_400_error {
        println!("\n=== REPRODUCTION SUCCESSFUL ===");
        println!("1. Tool was executed successfully");  
        println!("2. 400 error occurred when sending tool result back to OpenRouter");
        println!("This confirms the issue is in the tool result message conversion!");
        return Err("OpenRouter 400 error reproduced as expected".into());
    } else if tool_executed && !got_400_error {
        println!("\n=== TEST PASSED ===");
        println!("Tool was executed and no 400 error occurred!");
        println!("This means the fix is working correctly!");
    } else {
        println!("\n=== INCONCLUSIVE ===");
        println!("Tool was not executed, so we couldn't test the tool result conversion.");
    }
    
    Ok(())
}