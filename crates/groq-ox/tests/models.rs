use groq_ox::Groq;

#[tokio::test]
async fn test_list_models() {
    // Skip test if no API key is available
    let client = match Groq::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("Skipping test: GROQ_API_KEY not found");
            return;
        }
    };

    let response = client.list_models().await.expect("Failed to list models");

    // Basic assertions
    assert_eq!(response.object, "list");
    assert!(!response.data.is_empty(), "Should have at least one model");

    // Check that we have expected model types
    let has_chat_model = response.data.iter().any(|m| m.supports_chat());
    let _has_audio_model = response
        .data
        .iter()
        .any(|m| m.is_speech_to_text() || m.is_text_to_speech());

    assert!(has_chat_model, "Should have at least one chat model");
    // Note: Audio models might not always be available, so we don't assert on them

    // Verify model structure
    let first_model = &response.data[0];
    assert!(!first_model.id.is_empty(), "Model ID should not be empty");
    assert!(
        !first_model.owned_by.is_empty(),
        "Model owner should not be empty"
    );
    assert!(
        first_model.context_window > 0,
        "Context window should be positive"
    );

    println!("✅ Found {} models", response.data.len());
    for model in response.data.iter().take(3) {
        println!("   - {} ({})", model.id, model.family());
    }
}

#[tokio::test]
async fn test_get_specific_model() {
    // Skip test if no API key is available
    let client = match Groq::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("Skipping test: GROQ_API_KEY not found");
            return;
        }
    };

    // First get the list of models to find a valid model ID
    let models_response = client.list_models().await.expect("Failed to list models");
    let first_model = &models_response.data[0];

    // Now get the specific model
    let model = client
        .get_model(&first_model.id)
        .await
        .expect("Failed to get specific model");

    // Verify the response
    assert_eq!(model.id, first_model.id);
    assert_eq!(model.object, "model");
    assert!(!model.owned_by.is_empty());
    assert!(model.context_window > 0);
    assert!(model.created > 0);

    println!("✅ Retrieved model: {}", model.id);
    println!("   Context window: {}", model.context_window);
    println!("   Active: {}", model.active);
}

#[tokio::test]
async fn test_model_helper_methods() {
    // Test the helper methods on ModelInfo without making API calls
    use groq_ox::ModelInfo;

    // Create a test model info
    let llama_model = ModelInfo {
        id: "llama-3.3-70b-versatile".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "groq".to_string(),
        active: true,
        context_window: 131072,
        details: serde_json::Value::Null,
    };

    assert!(llama_model.supports_chat());
    assert!(!llama_model.is_speech_to_text());
    assert!(!llama_model.is_text_to_speech());
    assert_eq!(llama_model.family(), "llama");
    assert_eq!(llama_model.size_description(), Some("70B parameters"));

    // Test whisper model
    let whisper_model = ModelInfo {
        id: "whisper-large-v3".to_string(),
        object: "model".to_string(),
        created: 1234567890,
        owned_by: "groq".to_string(),
        active: true,
        context_window: 0,
        details: serde_json::Value::Null,
    };

    assert!(!whisper_model.supports_chat());
    assert!(whisper_model.is_speech_to_text());
    assert!(!whisper_model.is_text_to_speech());
    assert_eq!(whisper_model.family(), "whisper");

    println!("✅ Model helper methods work correctly");
}
