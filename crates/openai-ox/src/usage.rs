use serde::{Deserialize, Serialize};

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Number of tokens in the completion
    pub completion_tokens: u32,

    /// Total number of tokens used
    pub total_tokens: u32,

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
    pub cached_tokens: Option<u32>,

    /// Tokens used for audio processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// Detailed completion token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Tokens used for reasoning (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,

    /// Tokens used for audio generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

impl Usage {
    /// Create a new usage instance
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    /// Calculate the total cost based on token pricing
    pub fn calculate_cost(&self, prompt_price_per_1k: f64, completion_price_per_1k: f64) -> f64 {
        let prompt_cost = (self.prompt_tokens as f64 / 1000.0) * prompt_price_per_1k;
        let completion_cost = (self.completion_tokens as f64 / 1000.0) * completion_price_per_1k;
        prompt_cost + completion_cost
    }

    /// Get the ratio of completion tokens to prompt tokens
    pub fn completion_ratio(&self) -> f64 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            self.completion_tokens as f64 / self.prompt_tokens as f64
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
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            prompt_tokens_details: None, // Could be implemented to merge details
            completion_tokens_details: None,
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        // Details are lost in accumulation for simplicity
        self.prompt_tokens_details = None;
        self.completion_tokens_details = None;
    }
}
