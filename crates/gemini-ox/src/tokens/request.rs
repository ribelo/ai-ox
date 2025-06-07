use bon::Builder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::response::AuthTokenResponse;
use crate::{BASE_URL, Gemini, GeminiRequestError, parse_error_response};

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveConnectConfigOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio_transcription: Option<bool>,
    #[serde(flatten)]
    pub additional_config: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LiveEphemeralParametersOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<LiveConnectConfigOptions>,
}

#[derive(Debug, Builder)]
pub struct CreateAuthTokenOperation {
    pub uses: Option<i32>,
    pub expire_time: Option<String>,
    pub live_constrained_parameters: Option<LiveEphemeralParametersOptions>,
    pub lock_additional_fields: Option<Vec<String>>,
    pub(crate) gemini: Gemini,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiCreateAuthTokenPayload<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    uses: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expire_time: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bidi_generate_content_setup: Option<&'a LiveEphemeralParametersOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_mask: Option<String>,
}

impl CreateAuthTokenOperation {
    pub async fn send(&self) -> Result<AuthTokenResponse, GeminiRequestError> {
        let api_key = &self.gemini.api_key;
        let url = format!("{}/auth_tokens?key={}", BASE_URL, api_key);

        let final_field_mask = build_final_field_mask(
            self.live_constrained_parameters.as_ref(),
            self.lock_additional_fields.as_ref(),
        );

        let payload = ApiCreateAuthTokenPayload {
            uses: self.uses,
            expire_time: self.expire_time.as_deref(),
            bidi_generate_content_setup: self.live_constrained_parameters.as_ref(),
            field_mask: final_field_mask,
        };

        let response = self.gemini.client.post(&url).json(&payload).send().await?;

        if response.status().is_success() {
            response
                .json::<AuthTokenResponse>()
                .await
                .map_err(GeminiRequestError::from)
        } else {
            let status = response.status();
            let bytes = response.bytes().await?;
            Err(parse_error_response(status, bytes))
        }
    }
}

fn generate_base_field_mask_for_live_ephemeral(params: &LiveEphemeralParametersOptions) -> String {
    let mut fields = Vec::new();

    if params.model.is_some() {
        fields.push("model".to_string());
    }

    if let Some(config) = &params.config {
        if config.system_instruction.is_some() {
            fields.push("config.systemInstruction".to_string());
        }
        if config.output_audio_transcription.is_some() {
            fields.push("config.outputAudioTranscription".to_string());
        }
        for key in config.additional_config.keys() {
            fields.push(format!("config.{}", key));
        }
    }

    fields.join(",")
}

fn build_final_field_mask(
    live_params_opt: Option<&LiveEphemeralParametersOptions>,
    lock_additional_opt: Option<&Vec<String>>,
) -> Option<String> {
    match lock_additional_opt {
        None => None,
        Some(additional_fields) => {
            let mut all_fields = Vec::new();

            // Add base fields from live_constrained_parameters
            if let Some(params) = live_params_opt {
                let base_mask = generate_base_field_mask_for_live_ephemeral(params);
                if !base_mask.is_empty() {
                    all_fields.extend(base_mask.split(',').map(|s| s.to_string()));
                }
            }

            // Add additional fields from lock_additional_fields
            for field in additional_fields {
                // Check if field should be prefixed with "config."
                if is_config_field(field) {
                    all_fields.push(format!("config.{}", field));
                } else {
                    all_fields.push(field.clone());
                }
            }

            if all_fields.is_empty() {
                None
            } else {
                Some(all_fields.join(","))
            }
        }
    }
}

fn is_config_field(field: &str) -> bool {
    // Fields that belong to LiveConnectConfigOptions
    matches!(field, "systemInstruction" | "outputAudioTranscription") || 
    // If it's not a known top-level field, assume it's a config field
    !matches!(field, "model")
}

impl Gemini {
    pub fn create_auth_token(
        &self,
    ) -> CreateAuthTokenOperationBuilder<create_auth_token_operation_builder::SetGemini> {
        CreateAuthTokenOperation::builder().gemini(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_generate_base_field_mask_for_live_ephemeral() {
        let params = LiveEphemeralParametersOptions {
            model: Some("gemini-1.5-flash".to_string()),
            config: Some(LiveConnectConfigOptions {
                system_instruction: Some("You are helpful".to_string()),
                output_audio_transcription: Some(true),
                additional_config: {
                    let mut map = HashMap::new();
                    map.insert("customField".to_string(), json!("value"));
                    map
                },
            }),
        };

        let mask = generate_base_field_mask_for_live_ephemeral(&params);
        let fields: Vec<&str> = mask.split(',').collect();

        assert!(fields.contains(&"model"));
        assert!(fields.contains(&"config.systemInstruction"));
        assert!(fields.contains(&"config.outputAudioTranscription"));
        assert!(fields.contains(&"config.customField"));
    }

    #[test]
    fn test_build_final_field_mask_none() {
        let result = build_final_field_mask(None, None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_build_final_field_mask_empty_additional() {
        let params = LiveEphemeralParametersOptions {
            model: Some("gemini-1.5-flash".to_string()),
            config: None,
        };

        let result = build_final_field_mask(Some(&params), Some(&vec![]));
        assert_eq!(result, Some("model".to_string()));
    }

    #[test]
    fn test_build_final_field_mask_with_additional() {
        let params = LiveEphemeralParametersOptions {
            model: Some("gemini-1.5-flash".to_string()),
            config: None,
        };

        let additional = vec!["systemInstruction".to_string()];
        let result = build_final_field_mask(Some(&params), Some(&additional));

        assert!(result.is_some());
        let mask = result.unwrap();
        assert!(mask.contains("model"));
        assert!(mask.contains("config.systemInstruction"));
    }

    #[test]
    fn test_is_config_field() {
        assert!(is_config_field("systemInstruction"));
        assert!(is_config_field("outputAudioTranscription"));
        assert!(is_config_field("customField"));
        assert!(!is_config_field("model"));
    }

    #[tokio::test]
    async fn test_create_auth_token_operation_builder() {
        let gemini = Gemini::new("test-api-key");

        let _operation = gemini
            .create_auth_token()
            .uses(10)
            .expire_time("2024-12-31T23:59:59Z".to_string())
            .build();

        // This test just verifies the builder compiles and can be constructed
    }

    #[tokio::test]
    async fn test_send_operation_payload_structure() {
        // Test that the payload is structured correctly
        let params = LiveEphemeralParametersOptions {
            model: Some("gemini-1.5-flash".to_string()),
            config: Some(LiveConnectConfigOptions {
                system_instruction: Some("You are helpful".to_string()),
                output_audio_transcription: None,
                additional_config: HashMap::new(),
            }),
        };

        let final_field_mask = build_final_field_mask(Some(&params), Some(&vec![]));

        let payload = ApiCreateAuthTokenPayload {
            uses: Some(10),
            expire_time: Some("2024-12-31T23:59:59Z"),
            bidi_generate_content_setup: Some(&params),
            field_mask: final_field_mask,
        };

        let serialized = serde_json::to_value(&payload).unwrap();

        // Verify the structure matches expected API format
        assert_eq!(serialized["uses"], 10);
        assert_eq!(serialized["expireTime"], "2024-12-31T23:59:59Z");
        assert_eq!(serialized["fieldMask"], "model,config.systemInstruction");
        assert_eq!(
            serialized["bidiGenerateContentSetup"]["model"],
            "gemini-1.5-flash"
        );
        assert_eq!(
            serialized["bidiGenerateContentSetup"]["config"]["systemInstruction"],
            "You are helpful"
        );
    }

    #[test]
    fn test_error_response_parsing() {
        // Test that error responses are parsed correctly
        let error_json = json!({
            "error": {
                "code": 400,
                "message": "Invalid API key",
                "status": "INVALID_ARGUMENT"
            }
        });

        let error_bytes = serde_json::to_vec(&error_json).unwrap();
        let status = reqwest::StatusCode::BAD_REQUEST;
        let error = parse_error_response(status, error_bytes.into());

        match error {
            GeminiRequestError::InvalidRequestError { message, .. } => {
                assert_eq!(message, "Invalid API key");
            }
            _ => panic!("Expected InvalidRequestError"),
        }
    }

    #[test]
    fn test_api_payload_serialization() {
        let params = LiveEphemeralParametersOptions {
            model: Some("gemini-1.5-flash".to_string()),
            config: Some(LiveConnectConfigOptions {
                system_instruction: Some("You are helpful".to_string()),
                output_audio_transcription: Some(true),
                additional_config: HashMap::new(),
            }),
        };

        let payload = ApiCreateAuthTokenPayload {
            uses: Some(10),
            expire_time: Some("2024-12-31T23:59:59Z"),
            bidi_generate_content_setup: Some(&params),
            field_mask: Some("model,config.systemInstruction".to_string()),
        };

        let serialized = serde_json::to_value(&payload).unwrap();

        assert_eq!(serialized["uses"], 10);
        assert_eq!(serialized["expireTime"], "2024-12-31T23:59:59Z");
        assert_eq!(serialized["fieldMask"], "model,config.systemInstruction");
        assert_eq!(
            serialized["bidiGenerateContentSetup"]["model"],
            "gemini-1.5-flash"
        );
        assert_eq!(
            serialized["bidiGenerateContentSetup"]["config"]["systemInstruction"],
            "You are helpful"
        );
    }
}
