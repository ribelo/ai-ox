use gemini_ox::{Gemini, GeminiRequestError};
use std::env;

fn get_api_key() -> Option<String> {
    env::var("GOOGLE_AI_API_KEY").ok()
}

fn setup_client() -> Option<Gemini> {
    get_api_key().map(|key| Gemini::new(key))
}

#[tokio::test]
#[ignore = "requires GOOGLE_AI_API_KEY"]
async fn test_list_models_succeeds() {
    let client = setup_client().expect("GOOGLE_AI_API_KEY not set");

    let response = client.list_models(None, None).await;

    assert!(response.is_ok(), "list_models should succeed");
    let models_response = response.unwrap();
    assert!(
        !models_response.models.is_empty(),
        "models list should not be empty"
    );
}

#[tokio::test]
#[ignore = "requires GOOGLE_AI_API_KEY"]
async fn test_get_model_succeeds() {
    let client = setup_client().expect("GOOGLE_AI_API_KEY not set");

    let response = client.get_model("gemini-1.5-flash-latest").await;

    assert!(response.is_ok(), "get_model should succeed for valid model");
    let model = response.unwrap();
    assert!(
        model.name.contains("gemini-1.5-flash"),
        "model name should contain gemini-1.5-flash"
    );
}

#[tokio::test]
#[ignore = "requires GOOGLE_AI_API_KEY"]
async fn test_get_model_fails_on_unknown_model() {
    let client = setup_client().expect("GOOGLE_AI_API_KEY not set");

    let response = client.get_model("non-existent-model-12345").await;

    assert!(
        response.is_err(),
        "get_model should fail for non-existent model"
    );
    match response.unwrap_err() {
        GeminiRequestError::InvalidRequestError { .. } => {
            // This is expected
        }
        GeminiRequestError::UnexpectedResponse(_) => {
            // This is also acceptable for 404 responses
        }
        other => panic!("Unexpected error type: {:?}", other),
    }
}

#[tokio::test]
#[ignore = "requires GOOGLE_AI_API_KEY"]
async fn test_list_models_with_pagination() {
    let client = setup_client().expect("GOOGLE_AI_API_KEY not set");

    let response = client.list_models(Some(5), None).await;

    assert!(
        response.is_ok(),
        "list_models with page_size should succeed"
    );
    let models_response = response.unwrap();
    assert!(
        !models_response.models.is_empty(),
        "models list should not be empty"
    );
    assert!(
        models_response.models.len() <= 5,
        "should respect page_size limit"
    );
}
