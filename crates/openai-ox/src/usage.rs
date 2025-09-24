use serde::{Deserialize, Serialize};

use ai_ox_common::usage::TokenUsage;

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(flatten)]
    pub tokens: TokenUsage,

    /// Detailed token usage breakdown (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,

    /// Detailed completion token usage (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Detailed prompt token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Tokens used for cached content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,

    /// Tokens used for audio processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
}

/// Detailed completion token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Tokens used for reasoning (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,

    /// Tokens used for audio generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
}

impl Usage {
    /// Create a new usage instance
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            tokens: TokenUsage::with_prompt_completion(prompt_tokens, completion_tokens),
            prompt_tokens_details: None,
            completion_tokens_details: None,
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

    /// Calculate the total cost based on token pricing
    pub fn calculate_cost(&self, prompt_price_per_1k: f64, completion_price_per_1k: f64) -> f64 {
        let prompt_cost = (self.tokens.prompt_tokens() as f64 / 1000.0) * prompt_price_per_1k;
        let completion_cost =
            (self.tokens.completion_tokens() as f64 / 1000.0) * completion_price_per_1k;
        prompt_cost + completion_cost
    }

    /// Get the ratio of completion tokens to prompt tokens
    pub fn completion_ratio(&self) -> f64 {
        if self.tokens.prompt_tokens() == 0 {
            0.0
        } else {
            self.tokens.completion_tokens() as f64 / self.tokens.prompt_tokens() as f64
        }
    }

    /// Check if this usage represents a cached response
    pub fn is_cached(&self) -> bool {
        self.prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .unwrap_or(0)
            > 0
    }
}

impl std::ops::Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            tokens: self.tokens + other.tokens,
            prompt_tokens_details: None, // Could be implemented to merge details
            completion_tokens_details: None,
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.tokens += other.tokens;
        // Details are lost in accumulation for simplicity
        self.prompt_tokens_details = None;
        self.completion_tokens_details = None;
    }
}
