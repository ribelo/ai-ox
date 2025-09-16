use ai_ox::{
    content::{
        message::{Message, MessageRole},
        part::Part,
    },
    model::{mistral::MistralModel, request::ModelRequest},
};
use std::collections::BTreeMap;

#[tokio::main]
async fn main() {
    // Create the same message as in the failing test
    let message = Message {
        ext: Some(BTreeMap::new()),
        role: MessageRole::User,
        content: vec![Part::Text {
            ext: BTreeMap::new(),
            text: "Why is the sky blue? Please respond in exactly one sentence.".to_string(),
        }],
        timestamp: Some(chrono::Utc::now()),
    };

    let request = ModelRequest {
        messages: vec![message],
        system_message: None,
        tools: None,
    };

    // Create Mistral model (this will fail with dummy key but we can see the JSON)
    match MistralModel::new("mistral-small-latest".to_string()).await {
        Ok(model) => {
            println!("Model created successfully");

            // Try to convert the request to see the JSON
            let mistral_request = ai_ox::model::mistral::conversion::convert_request_to_mistral(
                request,
                "mistral-small-latest".to_string(),
                None,
                Some(mistral_ox::tool::ToolChoice::Auto),
            );

            match mistral_request {
                Ok(req) => {
                    println!("Request conversion successful");
                    // Print the JSON to see if it's truncated
                    match serde_json::to_string_pretty(&req) {
                        Ok(json) => println!("JSON Request:\n{}", json),
                        Err(e) => println!("Failed to serialize JSON: {}", e),
                    }
                }
                Err(e) => println!("Request conversion failed: {:?}", e),
            }
        }
        Err(e) => {
            println!("Model creation failed (expected with dummy key): {:?}", e);

            // Still try to convert the request to see the JSON
            let mistral_request = ai_ox::model::mistral::conversion::convert_request_to_mistral(
                request,
                "mistral-small-latest".to_string(),
                None,
                Some(mistral_ox::tool::ToolChoice::Auto),
            );

            match mistral_request {
                Ok(req) => {
                    println!("Request conversion successful");
                    // Print the JSON to see if it's truncated
                    match serde_json::to_string_pretty(&req) {
                        Ok(json) => println!("JSON Request:\n{}", json),
                        Err(e) => println!("Failed to serialize JSON: {}", e),
                    }
                }
                Err(e) => println!("Request conversion failed: {:?}", e),
            }
        }
    }
}