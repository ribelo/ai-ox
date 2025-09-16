use gemini_ox::generate_content::response::GenerateContentResponse;
use serde_json;

#[test]
fn test_response_with_only_usage_metadata() {
    // This is the exact response format that was failing
    let json_response = r#"{
        "modelVersion": "gemini-2.5-flash",
        "responseId": "3E_JaLLtI_TT_uMP1pHHmAU",
        "usageMetadata": {
            "promptTokenCount": 181,
            "promptTokensDetails": [
                {
                    "modality": "TEXT",
                    "tokenCount": 181
                }
            ],
            "totalTokenCount": 181
        }
    }"#;

    // This should now parse successfully
    let result: Result<GenerateContentResponse, _> = serde_json::from_str(json_response);

    match result {
        Ok(response) => {
            println!("✅ Successfully parsed response with only usageMetadata");
            println!("Response: {:?}", response);
            assert!(response.candidates.is_empty());
            assert!(response.usage_metadata.is_some());
            assert_eq!(response.usage_metadata.as_ref().unwrap().prompt_token_count, 181);
        }
        Err(e) => {
            panic!("❌ Failed to parse response: {}", e);
        }
    }
}

#[test]
fn test_response_with_candidates_and_usage_metadata() {
    // Test that responses with both candidates and usageMetadata still work
    let json_response = r#"{
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello, world!"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30
        }
    }"#;

    let result: Result<GenerateContentResponse, _> = serde_json::from_str(json_response);

    match result {
        Ok(response) => {
            println!("✅ Successfully parsed response with candidates and usageMetadata");
            assert!(!response.candidates.is_empty());
            assert!(response.usage_metadata.is_some());
        }
        Err(e) => {
            panic!("❌ Failed to parse response: {}", e);
        }
    }
}