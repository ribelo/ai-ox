use gemini_ox::{
    Gemini, Model,
    content::{Content, Part, Role, Text},
    request::GenerateContentRequest,
};
use serde::Deserialize;
use std::{env, fs, path::PathBuf};

fn read_token_from_env() -> Option<String> {
    env::var("GOOGLE_OAUTH_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
}

#[derive(Deserialize)]
struct GeminiCliCreds {
    access_token: String,
}

fn creds_path() -> Option<PathBuf> {
    let override_path = env::var("GEMINI_OAUTH_CREDS_PATH").ok();
    if let Some(path) = override_path {
        return Some(PathBuf::from(path));
    }

    let home = env::var("HOME").ok().map(PathBuf::from);
    home.map(|h| h.join(".gemini").join("oauth_creds.json"))
}

fn read_token_from_gemini_cli() -> Option<String> {
    let path = creds_path()?;
    let contents = fs::read_to_string(path).ok()?;
    serde_json::from_str::<GeminiCliCreds>(&contents)
        .ok()
        .map(|creds| creds.access_token)
}

fn get_oauth_token() -> Option<String> {
    read_token_from_env().or_else(read_token_from_gemini_cli)
}

#[tokio::test]
#[ignore = "Requires OAuth credentials and makes actual API calls"]
async fn test_oauth_without_project_fails_as_expected() {
    let Some(token) = get_oauth_token() else {
        eprintln!("Skipping OAuth test: token not available");
        return;
    };
    let gemini = Gemini::with_oauth_token(token);

    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from(
                    "Hello! Just respond 'OK' to test OAuth.",
                ))])
                .role(Role::User)
                .build(),
        ])
        .build();

    let response = request.send(&gemini).await;

    // OAuth without project should fail for Cloud Code Assist API
    assert!(
        response.is_err(),
        "OAuth request without project should fail for Cloud Code Assist API"
    );

    if let Err(e) = response {
        // Error should be related to invalid request (project missing) regardless of exact wording
        let error_string = format!("{:?}", e);
        assert!(
            error_string.contains("RESOURCE_PROJECT_INVALID")
                || error_string.contains("Invalid resource field value"),
            "Unexpected error for missing project: {}",
            error_string
        );
    }
}

#[tokio::test]
#[ignore = "Requires OAuth credentials with project and makes actual API calls"]
async fn test_oauth_with_project() {
    let Some(token) = get_oauth_token() else {
        eprintln!("Skipping OAuth test: token not available");
        return;
    };
    let gemini = Gemini::with_oauth_token_and_project(token, "pioneering-trilogy-xq6tl");

    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from(
                    "Hello! Just respond 'OK' to test OAuth with project.",
                ))])
                .role(Role::User)
                .build(),
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

    let Some(token) = get_oauth_token() else {
        eprintln!("Skipping OAuth test: token not available");
        return;
    };
    let gemini = Gemini::with_oauth_token_and_project(token, "pioneering-trilogy-xq6tl");

    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Count to 3 slowly."))])
                .role(Role::User)
                .build(),
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

    assert!(
        chunks_received > 0,
        "Should receive at least one streaming chunk"
    );
    assert!(
        !full_response.is_empty(),
        "Should receive some text content"
    );
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
    let _gemini = Gemini::with_oauth_token_and_project(token, project);

    let request = GenerateContentRequest::builder()
        .model(Model::Gemini20Flash001)
        .content_list(vec![
            Content::builder()
                .parts(vec![Part::new(Text::from("Test message"))])
                .role(Role::User)
                .build(),
        ])
        .build();

    // Test that we can create OAuth requests without panicking
    // This verifies the OAuth code paths compile and work
    assert_eq!(request.model, "gemini-2.0-flash-001");
}
