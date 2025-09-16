use openrouter_ox::OpenRouter;

#[tokio::test]
#[ignore] // Requires actual API key to test
async fn test_list_models() {
    let client = OpenRouter::load_from_env().expect("OPENROUTER_API_KEY must be set");

    let models = client.list_models().await;

    match models {
        Ok(models_response) => {
            assert_eq!(models_response.object, "list");
            assert!(!models_response.data.is_empty());

            // Check first model has required fields
            let first_model = &models_response.data[0];
            assert!(!first_model.id.is_empty());
            assert!(!first_model.owned_by.is_empty());
            assert!(first_model.context_length > 0);
        }
        Err(e) => panic!("Failed to list models: {:?}", e),
    }
}

#[tokio::test]
#[ignore] // Requires actual API key to test
async fn test_get_key_status() {
    let client = OpenRouter::load_from_env().expect("OPENROUTER_API_KEY must be set");

    let key_status = client.get_key_status().await;

    match key_status {
        Ok(status) => {
            // Should have basic key status info
            assert!(status.data.usage >= 0.0);
            assert!(status.data.rate_limit.requests > 0);
            assert!(!status.data.rate_limit.interval.is_empty());
        }
        Err(e) => panic!("Failed to get key status: {:?}", e),
    }
}

#[test]
fn test_response_types_serialization() {
    use openrouter_ox::ModelInfo;
    use serde_json;

    // Test that our response types can be properly serialized/deserialized
    let model_info = ModelInfo {
        id: "test-model".to_string(),
        object: "model".to_string(),
        created: Some(1234567890),
        owned_by: "test-provider".to_string(),
        name: Some("Test Model".to_string()),
        description: Some("A test model".to_string()),
        pricing: openrouter_ox::ModelPricing {
            prompt: "0.001".to_string(),
            completion: "0.002".to_string(),
            image: None,
            request: None,
        },
        context_length: 4096,
        architecture: openrouter_ox::ModelArchitecture {
            modality: "text".to_string(),
            tokenizer: Some("tiktoken".to_string()),
            instruct_type: Some("chat".to_string()),
        },
        top_provider: openrouter_ox::ModelProvider {
            max_completion_tokens: Some(2048),
            is_moderated: Some(false),
        },
        per_request_limits: None,
    };

    // Test serialization roundtrip
    let json = serde_json::to_string(&model_info).expect("Should serialize");
    let _deserialized: ModelInfo = serde_json::from_str(&json).expect("Should deserialize");
}
