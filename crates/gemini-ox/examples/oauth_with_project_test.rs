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

    println!("🔑 Using OAuth token with Cloud Code Assist API and project ID...");

    // Create Gemini client with OAuth token and the project ID we got from setup
    let gemini = Gemini::with_oauth_token_and_project(access_token, "pioneering-trilogy-xq6tl");

    println!("📝 Making a generate content request with OAuth + project ID...");

    // Create a simple request
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Hello! Please respond with 'OAuth Cloud Code Assist with project test successful' if you can read this."))])
                .role(Role::User)
                .build()
        ])
        .build();

    // Make the request
    match request.send(&gemini).await {
        Ok(response) => {
            println!("✅ Success! OAuth with Cloud Code Assist API + project ID worked!");
            if let Some(candidate) = response.candidates.first() {
                let content = &candidate.content;
                for part in &content.parts {
                    if let Some(text) = part.as_text() {
                        println!("🤖 Response: {}", text);
                    }
                }
            }
            println!("📊 Usage: {:?}", response.usage_metadata);
        }
        Err(e) => {
            println!("❌ OAuth request failed: {}", e);
            println!("ℹ️ Error suggests we need proper project setup");
            return Err(e.into());
        }
    }

    Ok(())
}
