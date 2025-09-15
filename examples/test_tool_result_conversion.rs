use ai_ox::{
    agent::Agent,
    model::{ModelRequest, Message, Part, MessageRole},
    openrouter::OpenrouterModel,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing tool result conversion fix...");
    
    // Create a sample tool result message that mimics the complex structure from Agronauts
    let tool_result_content = json!([
        {
            "content": [
                {
                    "call_id": "tool_0_knowledge_search",
                    "content": [
                        {
                            "content_id": 0,
                            "document_id": 2,
                            "score": 0.4306826078078052,
                            "tags": [
                                {"id": 5, "name": "Centrum Doradztwa Rolniczego"},
                                {"id": 1, "name": "nawozy azotowe"}
                            ],
                            "text": "http://www.normatywy.cdr.gov.pl Normatywy Produkcji Rolniczej Centrum Doradztwa Rolniczego w Brwinowie Tabela 10 Ilość CaO potrzebna do neutralizacji zakwaszającego działania nawozów azotowych"
                        },
                        {
                            "content_id": 1,
                            "document_id": 3,
                            "score": 0.375433,
                            "tags": [
                                {"id": 7, "name": "gleby"},
                                {"id": 9, "name": "nawozy"}
                            ],
                            "text": "Tabela 11 Orientacyjne dawki wapna (t/ha) potrzebne do doprowadzenia odczynu gleby do obojętnego (pH około 6,5)"
                        }
                    ],
                    "name": "knowledge_search",
                    "type": "toolResult"
                }
            ],
            "role": "assistant",
            "timestamp": "2025-08-20T13:41:13.146545593Z"
        }
    ]);

    // Create the tool result message that would cause the 400 error
    let tool_result_message = Message {
        role: MessageRole::User,
        content: vec![Part::ToolResult {
            id: "tool_0_knowledge_search".to_string(),
            name: "knowledge_search".to_string(),
            content: vec![Part::Text { text: serde_json::to_string(&tool_result_content).unwrap() }],
        }],
        timestamp: chrono::Utc::now(),
    };

    // Create the model and agent for testing
    let model = OpenrouterModel::new("anthropic/claude-3-haiku")
        .with_api_key(std::env::var("OPENROUTER_API_KEY")?);
    let mut agent = Agent::new(model);
    
    // Create a request with the tool result message
    let request = ModelRequest {
        messages: vec![
            Message {
                role: MessageRole::User,
                content: vec![Part::Text {
                    text: "Tell me about corn fertilizers".to_string(),
                }],
                timestamp: chrono::Utc::now(),
            },
            Message {
                role: MessageRole::Assistant,
                content: vec![Part::ToolUse {
                    id: "tool_0_knowledge_search".to_string(),
                    name: "knowledge_search".to_string(),
                    args: json!({"query": "corn fertilizers"}),
                }],
                timestamp: chrono::Utc::now(),
            },
            tool_result_message,
        ],
        system_message: None,
        tools: Some(vec![]),
    };

    println!("Sending request with complex tool result to OpenRouter...");
    
    // This should work without the 400 error if our fix is correct
    match agent.model.request(request).await {
        Ok(response) => {
            println!("SUCCESS: Tool result processed without 400 error!");
            println!("Response: {:?}", response);
            
            println!("\n=== TEST PASSED ===");
            println!("The improved tool result conversion successfully handles complex nested JSON structures.");
        }
        Err(e) => {
            println!("ERROR: {}", e);
            let error_str = e.to_string();
            if error_str.contains("400") && error_str.contains("Provider returned error") {
                println!("\n=== TEST FAILED ===");
                println!("The 400 error still occurs when processing tool results.");
                println!("The tool result conversion fix needs further improvement.");
                return Err(e.into());
            } else {
                println!("Different error occurred (not the 400 error we're fixing): {}", e);
                return Err(e.into());
            }
        }
    }
    
    Ok(())
}