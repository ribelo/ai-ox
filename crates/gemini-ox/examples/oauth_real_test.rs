use gemini_ox::{
    Gemini, 
    Model,
    request::GenerateContentRequest,
    content::{Content, Part, Role, Text}
};
use futures_util::StreamExt;
use std::fs;
use serde_json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the OAuth token from Gemini CLI
    let oauth_creds = fs::read_to_string("/home/ribelo/.gemini/oauth_creds.json")?;
    let creds: serde_json::Value = serde_json::from_str(&oauth_creds)?;
    let access_token = creds["access_token"].as_str().unwrap();
    
    println!("=== OAuth Real Response Test ===\n");
    
    // Create Gemini client with OAuth token and project ID
    let gemini = Gemini::with_oauth_token_and_project(access_token, "pioneering-trilogy-xq6tl");
    
    // Test 1: Simple question
    println!("🧠 Test 1: Simple Question");
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("What is the capital of France?"))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    match request.send(&gemini).await {
        Ok(response) => {
            if let Some(candidate) = response.candidates.first() {
                let content = &candidate.content;
                for part in &content.parts {
                    if let Some(text) = part.as_text() {
                        println!("🤖 Response: {}", text);
                    }
                }
            }
            println!("📊 Usage: {:?}\n", response.usage_metadata);
        }
        Err(e) => {
            println!("❌ Error: {}", e);
            return Err(e.into());
        }
    }
    
    // Test 2: Streaming response
    println!("🌊 Test 2: Streaming Response");
    let streaming_request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Tell me a short joke about programming."))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    let mut stream = streaming_request.stream(&gemini);
    let mut full_response = String::new();
    
    print!("🤖 Streaming: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(response) => {
                if let Some(candidate) = response.candidates.first() {
                    let content = &candidate.content;
                    for part in &content.parts {
                        if let Some(text) = part.as_text() {
                            print!("{}", &**text);
                            full_response.push_str(&**text);
                        }
                    }
                }
            }
            Err(e) => {
                println!("\n❌ Streaming error: {}", e);
                return Err(e.into());
            }
        }
    }
    
    println!("\n");
    println!("✅ SUCCESS! OAuth is working with real API responses!");
    println!("📝 Full streaming response: {}", full_response);
    
    Ok(())
}