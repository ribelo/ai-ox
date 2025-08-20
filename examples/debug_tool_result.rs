use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::Part;
use ai_ox::model::openrouter::conversion::convert_message_to_openrouter;
use serde_json::json;

fn main() {
    println!("Testing OpenRouter tool result conversion...");
    
    // Create a message with tool result (similar to what Agronauts produces)
    let tool_result_content = json!([{
        "content_id": 0,
        "document_id": 2,
        "score": 0.4306826078078052,
        "tags": [
            {"id": 5, "name": "Centrum Doradztwa Rolniczego"},
            {"id": 1, "name": "nawozy azotowe"}
        ],
        "text": "Nitrogen fertilizers are essential for corn production. Apply 150-200 kg N/ha in split applications."
    }]);

    let tool_result_message = Message {
        role: MessageRole::User,
        content: vec![Part::ToolResult {
            call_id: "call_123".to_string(),
            name: "knowledge_search".to_string(),
            content: tool_result_content,
        }],
        timestamp: chrono::Utc::now(),
    };

    // Convert to OpenRouter format
    let openrouter_messages = convert_message_to_openrouter(tool_result_message);

    println!("Number of messages: {}", openrouter_messages.len());
    println!("Converted messages:");
    for (i, msg) in openrouter_messages.iter().enumerate() {
        println!("  Message {}: {:#?}", i, msg);
    }

    if let Some(first_msg) = openrouter_messages.first() {
        match first_msg {
            openrouter_ox::message::Message::Tool(tool_msg) => {
                println!("Tool message details:");
                println!("  tool_call_id: {}", tool_msg.tool_call_id);
                println!("  content: {}", tool_msg.content);
                
                // Try to parse the content as JSON
                match serde_json::from_str::<serde_json::Value>(&tool_msg.content) {
                    Ok(parsed) => println!("  parsed_content: {:#?}", parsed),
                    Err(e) => println!("  parsing error: {}", e),
                }
            }
            other => {
                println!("Unexpected message type: {:#?}", other);
            }
        }
    }
}