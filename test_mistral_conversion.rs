use ai_ox::content::part::Part;
use ai_ox::content::message::{Message, MessageRole};
use ai_ox::model::mistral::conversion::convert_message_to_mistral;
use chrono::Utc;
use serde_json::json;

fn main() {
    // Test basic text message
    let msg = Message {
        role: MessageRole::User,
        content: vec![Part::text("Hello world")],
        timestamp: Utc::now(),
    };
    
    match convert_message_to_mistral(msg) {
        Ok(messages) => println!("✓ Basic text conversion successful: {} messages", messages.len()),
        Err(e) => println!("✗ Basic text conversion failed: {}", e),
    }
    
    // Test tool result message
    let tool_msg = Message {
        role: MessageRole::User,
        content: vec![
            Part::text("Here is the result:"),
            Part::tool_result(
                "call_123",
                "test_tool",
                vec![Part::text(serde_json::to_string(&json!({"result": "success"})).unwrap())]
            )
        ],
        timestamp: Utc::now(),
    };
    
    match convert_message_to_mistral(tool_msg) {
        Ok(messages) => println!("✓ Tool result conversion successful: {} messages", messages.len()),
        Err(e) => println!("✗ Tool result conversion failed: {}", e),
    }
    
    println!("All tests completed!");
}
