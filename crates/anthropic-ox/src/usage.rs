use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
}

impl Usage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_tokens(&self) -> u32 {
        self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0)
    }

    pub fn total_input_tokens(&self) -> u32 {
        self.input_tokens.unwrap_or(0)
            + self.cache_creation_input_tokens.unwrap_or(0)
            + self.cache_read_input_tokens.unwrap_or(0)
    }

    /// Convenience method for compatibility with other providers
    /// Returns the same value as `input_tokens` - prompt tokens in Anthropic terms
    pub fn prompt_tokens(&self) -> u32 {
        self.input_tokens.unwrap_or(0)
    }

    /// Convenience method for compatibility with other providers  
    /// Returns the same value as `output_tokens` - completion tokens in Anthropic terms
    pub fn completion_tokens(&self) -> u32 {
        self.output_tokens.unwrap_or(0)
    }
}
