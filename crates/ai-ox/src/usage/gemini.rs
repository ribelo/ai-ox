use gemini_ox::generate_content::usage::UsageMetadata;

use super::{Modality, Usage};

impl From<UsageMetadata> for Usage {
    fn from(meta: UsageMetadata) -> Self {
        let mut input_tokens_by_modality = std::collections::HashMap::new();
        // The prompt_token_count in gemini is a u32, not an option.
        input_tokens_by_modality.insert(Modality::Text, meta.prompt_token_count as u64);

        let mut output_tokens_by_modality = std::collections::HashMap::new();
        if let Some(tokens) = meta.candidates_token_count {
            output_tokens_by_modality.insert(Modality::Text, tokens as u64);
        }

        let mut cache_tokens_by_modality = std::collections::HashMap::new();
        if let Some(tokens) = meta.cached_content_token_count {
            cache_tokens_by_modality.insert(Modality::Text, tokens as u64);
        }

        let mut tool_tokens_by_modality = std::collections::HashMap::new();
        if let Some(tokens) = meta.tool_use_prompt_token_count {
            tool_tokens_by_modality.insert(Modality::Text, tokens as u64);
        }

        Self {
            requests: 1, // One metadata object corresponds to one request.
            input_tokens_by_modality,
            output_tokens_by_modality,
            cache_tokens_by_modality,
            tool_tokens_by_modality,
            // Gemini lumps all cached tokens together. We'll put it in `cache_read_tokens`
            // as it represents tokens read from the cache during a request.
            cache_read_tokens: meta.cached_content_token_count.map(|t| t as u64),
            // Gemini API doesn't provide a separate cache creation token count.
            cache_creation_tokens: None,
            thoughts_tokens: meta.thoughts_token_count.map(|t| t as u64),
            // We are not mapping the detailed modality breakdowns for now to keep it simple.
            details: None,
        }
    }
}
