use serde_json::Value;

use crate::usage::{Modality, Usage};
use gemini_ox::generate_content::usage::UsageMetadata;

impl From<gemini_ox::generate_content::usage::Modality> for Modality {
    fn from(gemini_modality: gemini_ox::generate_content::usage::Modality) -> Self {
        match gemini_modality {
            gemini_ox::generate_content::usage::Modality::ModalityUnspecified => {
                Modality::Other("unspecified".to_string())
            }
            gemini_ox::generate_content::usage::Modality::Text => Modality::Text,
            gemini_ox::generate_content::usage::Modality::Image => Modality::Image,
            gemini_ox::generate_content::usage::Modality::Video => Modality::Video,
            gemini_ox::generate_content::usage::Modality::Audio => Modality::Audio,
            gemini_ox::generate_content::usage::Modality::Document => Modality::Document,
        }
    }
}

impl From<UsageMetadata> for Usage {
    fn from(gemini_usage: UsageMetadata) -> Self {
        use std::collections::HashMap;

        // Convert modality details to Value for storage in details
        let mut details_map = serde_json::Map::new();

        if let Some(prompt_details) = &gemini_usage.prompt_tokens_details {
            details_map.insert(
                "prompt_tokens_details".to_string(),
                serde_json::to_value(prompt_details).unwrap_or(Value::Null),
            );
        }

        if let Some(cache_details) = &gemini_usage.cache_tokens_details {
            details_map.insert(
                "cache_tokens_details".to_string(),
                serde_json::to_value(cache_details).unwrap_or(Value::Null),
            );
        }

        if let Some(candidates_details) = &gemini_usage.candidates_tokens_details {
            details_map.insert(
                "candidates_tokens_details".to_string(),
                serde_json::to_value(candidates_details).unwrap_or(Value::Null),
            );
        }

        if let Some(tool_details) = &gemini_usage.tool_use_prompt_tokens_details {
            details_map.insert(
                "tool_use_prompt_tokens_details".to_string(),
                serde_json::to_value(tool_details).unwrap_or(Value::Null),
            );
        }

        // Create input tokens map - put all prompt tokens as Text for simplicity
        let mut input_tokens_by_modality = HashMap::new();
        if gemini_usage.prompt_token_count > 0 {
            input_tokens_by_modality.insert(Modality::Text, gemini_usage.prompt_token_count as u64);
        }

        // Create output tokens map - put all candidate tokens as Text for simplicity
        let mut output_tokens_by_modality = HashMap::new();
        if let Some(candidates_count) = gemini_usage.candidates_token_count {
            if candidates_count > 0 {
                output_tokens_by_modality.insert(Modality::Text, candidates_count as u64);
            }
        }

        // Create cache tokens map if present
        let mut cache_tokens_by_modality = HashMap::new();
        if let Some(cache_count) = gemini_usage.cached_content_token_count {
            if cache_count > 0 {
                cache_tokens_by_modality.insert(Modality::Text, cache_count as u64);
            }
        }

        // Create tool tokens map if present
        let mut tool_tokens_by_modality = HashMap::new();
        if let Some(tool_count) = gemini_usage.tool_use_prompt_token_count {
            if tool_count > 0 {
                tool_tokens_by_modality.insert(Modality::Text, tool_count as u64);
            }
        }

        Usage {
            requests: 1, // Each UsageMetadata represents one request
            input_tokens_by_modality,
            output_tokens_by_modality,
            cache_tokens_by_modality,
            tool_tokens_by_modality,
            cache_read_tokens: gemini_usage.cached_content_token_count.map(|x| x as u64),
            cache_creation_tokens: None, // Gemini doesn't distinguish cache creation vs read
            thoughts_tokens: gemini_usage.thoughts_token_count.map(|x| x as u64),
            details: if details_map.is_empty() {
                None
            } else {
                Some(Value::Object(details_map))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_conversion() {
        let gemini_text = gemini_ox::generate_content::usage::Modality::Text;
        let ai_ox_text = Modality::from(gemini_text);
        assert_eq!(ai_ox_text, Modality::Text);

        let gemini_unspecified = gemini_ox::generate_content::usage::Modality::ModalityUnspecified;
        let ai_ox_unspecified = Modality::from(gemini_unspecified);
        assert_eq!(
            ai_ox_unspecified,
            Modality::Other("unspecified".to_string())
        );
    }

    #[test]
    fn test_usage_conversion_basic() {
        let gemini_usage = UsageMetadata {
            prompt_token_count: 100,
            cached_content_token_count: Some(50),
            candidates_token_count: Some(75),
            tool_use_prompt_token_count: None,
            thoughts_token_count: None,
            total_token_count: 175,
            prompt_tokens_details: None,
            cache_tokens_details: None,
            candidates_tokens_details: None,
            tool_use_prompt_tokens_details: None,
        };

        let ai_ox_usage = Usage::from(gemini_usage);

        assert_eq!(ai_ox_usage.requests, 1);
        assert_eq!(ai_ox_usage.input_tokens(), 100);
        assert_eq!(ai_ox_usage.output_tokens(), 75);
        assert_eq!(ai_ox_usage.cache_read_tokens, Some(50));
        assert_eq!(ai_ox_usage.cache_creation_tokens, None);
        assert_eq!(ai_ox_usage.total_tokens(), 175);

        // Check that tokens are stored in modality maps
        assert_eq!(
            *ai_ox_usage
                .input_tokens_by_modality
                .get(&Modality::Text)
                .unwrap(),
            100
        );
        assert_eq!(
            *ai_ox_usage
                .output_tokens_by_modality
                .get(&Modality::Text)
                .unwrap(),
            75
        );
        assert_eq!(
            *ai_ox_usage
                .cache_tokens_by_modality
                .get(&Modality::Text)
                .unwrap(),
            50
        );
    }

    #[test]
    fn test_usage_conversion_with_details() {
        use gemini_ox::generate_content::usage::ModalityTokenCount;

        let modality_details = vec![
            ModalityTokenCount {
                modality: gemini_ox::generate_content::usage::Modality::Text,
                token_count: 80,
            },
            ModalityTokenCount {
                modality: gemini_ox::generate_content::usage::Modality::Image,
                token_count: 20,
            },
        ];

        let gemini_usage = UsageMetadata {
            prompt_token_count: 100,
            cached_content_token_count: None,
            candidates_token_count: Some(75),
            tool_use_prompt_token_count: None,
            thoughts_token_count: None,
            total_token_count: 175,
            prompt_tokens_details: Some(modality_details.clone()),
            cache_tokens_details: None,
            candidates_tokens_details: None,
            tool_use_prompt_tokens_details: None,
        };

        let ai_ox_usage = Usage::from(gemini_usage);

        assert_eq!(ai_ox_usage.requests, 1);
        assert_eq!(ai_ox_usage.input_tokens(), 100);
        assert_eq!(ai_ox_usage.output_tokens(), 75);
        assert!(ai_ox_usage.details.is_some());

        if let Some(Value::Object(details)) = &ai_ox_usage.details {
            assert!(details.contains_key("prompt_tokens_details"));
        }
    }
}
