use serde::{Deserialize, Serialize};

use ai_ox_common::usage::TokenUsage;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    #[serde(flatten)]
    pub tokens: TokenUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time: Option<f64>,
}

impl Usage {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            tokens: TokenUsage::with_prompt_completion(prompt_tokens, completion_tokens),
            prompt_time: None,
            completion_time: None,
            total_time: None,
        }
    }

    pub fn with_timing(
        prompt_tokens: u64,
        completion_tokens: u64,
        prompt_time: f64,
        completion_time: f64,
    ) -> Self {
        Self {
            tokens: TokenUsage::with_prompt_completion(prompt_tokens, completion_tokens),
            prompt_time: Some(prompt_time),
            completion_time: Some(completion_time),
            total_time: Some(prompt_time + completion_time),
        }
    }

    pub fn prompt_tokens(&self) -> u64 {
        self.tokens.prompt_tokens()
    }

    pub fn completion_tokens(&self) -> u64 {
        self.tokens.completion_tokens()
    }

    pub fn total_tokens(&self) -> u64 {
        self.tokens.total_tokens()
    }
}
