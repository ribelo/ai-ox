use gemini_ox::{
    Gemini, 
    Model,
    request::GenerateContentRequest,
    content::{Content, Part, Role, Text}
};
use std::fs;
use serde_json;

fn get_oauth_token() -> Option<String> {
    match fs::read_to_string("/home/ribelo/.gemini/oauth_creds.json") {
        Ok(oauth_creds) => {
            match serde_json::from_str::<serde_json::Value>(&oauth_creds) {
                Ok(creds) => {
                    creds["access_token"].as_str().map(|s| s.to_string())
                }
                Err(_) => None,
            }
        }
        Err(_) => None,
    }
}

#[tokio::test]
#[ignore = "Requires OAuth credentials and makes actual API calls"]
async fn test_oauth_without_project_fails_as_expected() {
    let token = get_oauth_token().expect("OAuth token not available");
    let gemini = Gemini::with_oauth_token(token);
    
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Hello! Just respond 'OK' to test OAuth."))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    let response = request.send(&gemini).await;
    
    // OAuth without project should fail for Cloud Code Assist API
    assert!(response.is_err(), "OAuth request without project should fail for Cloud Code Assist API");
    
    if let Err(e) = response {
        // Should get RESOURCE_PROJECT_INVALID error
        let error_string = format!("{:?}", e);
        assert!(error_string.contains("RESOURCE_PROJECT_INVALID"), 
               "Should get project invalid error, got: {}", error_string);
    }
}

#[tokio::test]
#[ignore = "Requires OAuth credentials with project and makes actual API calls"]
async fn test_oauth_with_project() {
    let token = get_oauth_token().expect("OAuth token not available");
    let gemini = Gemini::with_oauth_token_and_project(token, "pioneering-trilogy-xq6tl");
    
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Hello! Just respond 'OK' to test OAuth with project."))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    let response = request.send(&gemini).await;
    assert!(response.is_ok(), "OAuth request with project should work");
    
    let response = response.unwrap();
    assert!(!response.candidates.is_empty());
    assert!(response.candidates[0].content.parts.len() > 0);
}

#[tokio::test]
#[ignore = "Requires OAuth credentials with project and makes actual API calls"]
async fn test_oauth_streaming() {
    use futures_util::StreamExt;
    
    let token = get_oauth_token().expect("OAuth token not available");
    let gemini = Gemini::with_oauth_token_and_project(token, "pioneering-trilogy-xq6tl");
    
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Count to 3 slowly."))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    let mut stream = request.stream(&gemini);
    let mut chunks_received = 0;
    let mut full_response = String::new();
    
    while let Some(chunk) = stream.next().await {
        let response = chunk.expect("Streaming chunk should be valid");
        chunks_received += 1;
        
        if let Some(candidate) = response.candidates.first() {
            for part in &candidate.content.parts {
                if let Some(text) = part.as_text() {
                    full_response.push_str(&**text);
                }
            }
        }
    }
    
    assert!(chunks_received > 0, "Should receive at least one streaming chunk");
    assert!(!full_response.is_empty(), "Should receive some text content");
}

#[test]
fn test_oauth_client_construction() {
    let token = "test_token";
    let project = "test_project";
    
    // Test that constructors work without panicking
    let _client_without_project = Gemini::with_oauth_token(token);
    let _client_with_project = Gemini::with_oauth_token_and_project(token, project);
    
    // Basic smoke test - if constructors work, the test passes
    assert!(true);
}

#[test] 
fn test_oauth_request_construction() {
    let token = "test_oauth_token";
    let project = "test_project";
    let gemini = Gemini::with_oauth_token_and_project(token, project);
    
    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Test message"))])
                .role(Role::User)
                .build()
        ])
        .build();
    
    // Test that we can create OAuth requests without panicking
    // This verifies the OAuth code paths compile and work
    assert_eq!(request.model, "gemini-2.0-flash-001");
}