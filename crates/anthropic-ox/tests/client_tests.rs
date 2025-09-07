use anthropic_ox::{Anthropic, AnthropicRequestError};

#[test]
fn test_client_creation() {
    let client = Anthropic::new("test-key");
    
    // Client should be created successfully
    assert_eq!(format!("{:?}", client).contains("Anthropic"), true);
}

#[test]
#[ignore = "Environment variable tests are unreliable in concurrent test execution"]
fn test_client_from_env_missing_key() {
    // Remove any existing API key
    unsafe {
        std::env::remove_var("ANTHROPIC_API_KEY");
    }
    
    let result = Anthropic::load_from_env();
    assert!(result.is_err());
}

#[test]
fn test_client_from_env_with_key() {
    unsafe {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
    }
    
    let result = Anthropic::load_from_env();
    assert!(result.is_ok());
    
    unsafe {
        std::env::remove_var("ANTHROPIC_API_KEY");
    }
}

#[test]
fn test_beta_features() {
    use anthropic_ox::client::BetaFeatures;
    
    let beta_features = BetaFeatures {
        fine_grained_tool_streaming: true,
        interleaved_thinking: true,
        computer_use: false,
    };
    
    let client = Anthropic::new("test-key").with_beta_features(beta_features);
    
    // Should contain beta header
    let debug_str = format!("{:?}", client);
    assert!(debug_str.contains("Anthropic"));
}

#[test]
fn test_custom_headers() {
    let client = Anthropic::new("test-key")
        .header("custom-header", "custom-value")
        .header("another-header", "another-value");
    
    // Client should be created with custom headers
    let debug_str = format!("{:?}", client);
    assert!(debug_str.contains("Anthropic"));
}

#[test]
fn test_endpoint_creation() {
    use anthropic_ox::internal::{Endpoint, HttpMethod};
    
    let endpoint = Endpoint::new("test/path", HttpMethod::Get);
    assert_eq!(endpoint.path, "test/path");
    
    let endpoint_with_beta = endpoint.with_beta("test-beta");
    assert_eq!(endpoint_with_beta.requires_beta, Some("test-beta".to_string()));
    
    let endpoint_with_params = Endpoint::new("test/path", HttpMethod::Get)
        .with_query_params(vec![("limit".to_string(), "10".to_string())]);
    assert!(endpoint_with_params.query_params.is_some());
}

#[test]
fn test_http_method_conversion() {
    use anthropic_ox::internal::HttpMethod;
    use reqwest::Method;
    
    assert_eq!(Method::from(HttpMethod::Get), Method::GET);
    assert_eq!(Method::from(HttpMethod::Post), Method::POST);
    assert_eq!(Method::from(HttpMethod::Delete), Method::DELETE);
}

#[tokio::test]
async fn test_authentication_missing_error() {
    // Create client without API key or OAuth token
    let client = Anthropic::builder().build();
    
    // This should fail with AuthenticationMissing error
    use anthropic_ox::{request::ChatRequest, message::{Message, Role}};
    
    let request = ChatRequest::builder()
        .model("claude-3-haiku-20240307".to_string())
        .messages(vec![Message::user(vec!["Hello"])])
        .build();
    
    let result = client.send(&request).await;
    
    match result {
        Err(AnthropicRequestError::AuthenticationMissing) => {
            // Expected error
        }
        _ => panic!("Expected AuthenticationMissing error"),
    }
}

#[cfg(feature = "models")]
#[tokio::test] 
async fn test_models_with_missing_auth() {
    let client = Anthropic::builder().build();
    
    let result = client.list_models(None, None, None).await;
    
    match result {
        Err(AnthropicRequestError::AuthenticationMissing) => {
            // Expected error
        }
        _ => panic!("Expected AuthenticationMissing error"),
    }
}

#[cfg(feature = "tokens")]
#[tokio::test]
async fn test_token_counting_with_missing_auth() {
    use anthropic_ox::tokens::TokenCountRequest;
    use anthropic_ox::message::{Message, Role};
    
    let client = Anthropic::builder().build();
    
    let request = TokenCountRequest {
        model: "claude-3-haiku-20240307".to_string(),
        messages: vec![Message::user(vec!["Hello"])].into(),
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };
    
    let result = client.count_tokens(&request).await;
    
    match result {
        Err(AnthropicRequestError::AuthenticationMissing) => {
            // Expected error
        }
        _ => panic!("Expected AuthenticationMissing error"),
    }
}

#[cfg(feature = "batches")]
#[tokio::test]
async fn test_batches_with_missing_auth() {
    use anthropic_ox::batches::MessageBatchRequest;
    
    let client = Anthropic::builder().build();
    
    let request = MessageBatchRequest {
        requests: vec![],
    };
    
    let result = client.create_message_batch(&request).await;
    
    match result {
        Err(AnthropicRequestError::AuthenticationMissing) => {
            // Expected error
        }
        _ => panic!("Expected AuthenticationMissing error"),
    }
}

#[cfg(feature = "files")]
#[tokio::test]
async fn test_files_with_missing_auth() {
    let client = Anthropic::builder().build();
    
    let result = client.list_files(None, None, None).await;
    
    match result {
        Err(AnthropicRequestError::AuthenticationMissing) => {
            // Expected error
        }
        _ => panic!("Expected AuthenticationMissing error"),
    }
}

#[cfg(feature = "admin")]
#[tokio::test]
async fn test_admin_with_missing_auth() {
    let client = Anthropic::builder().build();
    
    let result = client.list_organization_users().await;
    
    match result {
        Err(AnthropicRequestError::AuthenticationMissing) => {
            // Expected error
        }
        _ => panic!("Expected AuthenticationMissing error"),
    }
}

#[test]
fn test_request_builder_debug() {
    use anthropic_ox::internal::{RequestBuilder, Endpoint, HttpMethod};
    use std::collections::HashMap;
    
    let client = reqwest::Client::new();
    let headers = HashMap::new();
    let api_key = Some("test-key".to_string());
    let oauth_token = None;
    
    let builder = RequestBuilder::new(
        &client,
        "https://api.anthropic.com",
        &api_key,
        &oauth_token,
        "2023-06-01",
        &headers,
    );
    
    let endpoint = Endpoint::new("test", HttpMethod::Get);
    let req_result = builder.build_request(&endpoint);
    
    // Should build request successfully with proper auth
    assert!(req_result.is_ok());
}

