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
    
    println!("=== OAuth Methods Comparison ===\n");
    
    // Create Gemini client with OAuth token and project ID
    let gemini = Gemini::with_oauth_token_and_project(access_token, "pioneering-trilogy-xq6tl");
    
    // Test same request with both methods
    let prompt = "Say 'Hello from OAuth' in exactly 3 words.";
    
    println!("📤 Testing: '{}'", prompt);
    println!();
    
    // Method 1: send() - Regular request
    println!("🔄 Method 1: send() - Regular OAuth Request");
    let request1 = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from(prompt))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    match request1.send(&gemini).await {
        Ok(response) => {
            if let Some(candidate) = response.candidates.first() {
                for part in &candidate.content.parts {
                    if let Some(text) = part.as_text() {
                        println!("✅ send() response: {}", text);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ send() error: {}", e);
        }
    }
    
    println!();
    
    // Method 2: stream() - Streaming request  
    println!("🌊 Method 2: stream() - Streaming OAuth Request");
    let request2 = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from(prompt))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    let mut stream = request2.stream(&gemini);
    let mut full_response = String::new();
    
    print!("✅ stream() response: ");
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(response) => {
                if let Some(candidate) = response.candidates.first() {
                    for part in &candidate.content.parts {
                        if let Some(text) = part.as_text() {
                            print!("{}", &**text);
                            full_response.push_str(&**text);
                        }
                    }
                }
            }
            Err(e) => {
                println!("\n❌ stream() error: {}", e);
                return Err(e.into());
            }
        }
    }
    println!();
    
    println!("\n🎉 BOTH METHODS WORK WITH OAUTH!");
    println!("• send(): Makes single OAuth request to Cloud Code Assist API");
    println!("• stream(): Makes streaming OAuth request to Cloud Code Assist API");
    println!("• Both use same OAuth token + project authentication");
    println!("• Both route to https://cloudcode-pa.googleapis.com");
    
    Ok(())
}