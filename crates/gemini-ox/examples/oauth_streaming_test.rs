use futures_util::StreamExt;
use gemini_ox::{
    Gemini, Model,
    content::{Content, Part, Role, Text},
    request::GenerateContentRequest,
};
use serde_json;
use std::fs;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the OAuth token from Gemini CLI
    let oauth_creds = fs::read_to_string("/home/ribelo/.gemini/oauth_creds.json")?;
    let creds: serde_json::Value = serde_json::from_str(&oauth_creds)?;
    let access_token = creds["access_token"].as_str().unwrap();

    println!("🔑 Using OAuth token with Cloud Code Assist API (streaming)...");

    // Create Gemini client with OAuth token and project ID we got from setup
    let gemini = Gemini::with_oauth_token_and_project(access_token, "pioneering-trilogy-xq6tl");

    println!("📝 Making a STREAMING generate content request with OAuth...");

    // Create a simple request
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Hello! Please respond with 'OAuth Cloud Code Assist STREAMING test successful' if you can read this."))])
                .role(Role::User)
                .build()
        ])
        .build();

    // Make the streaming request - this should work better with Cloud Code Assist
    let mut stream = request.stream(&gemini);
    let mut full_response = String::new();

    println!("📡 Streaming response:");
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

    println!("\n🎉 SUCCESS! OAuth streaming with Cloud Code Assist API worked!");
    println!("🤖 Full response: {}", full_response);

    Ok(())
}
