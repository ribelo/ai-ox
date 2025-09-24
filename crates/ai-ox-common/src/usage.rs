use serde::{Deserialize, Serialize};

/// Normalised token usage information shared across providers.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_tokens: Option<u64>,
}

impl TokenUsage {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_prompt_completion(prompt: u64, completion: u64) -> Self {
        Self {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            total_tokens: Some(prompt + completion),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn prompt_tokens(&self) -> u64 {
        self.prompt_tokens.unwrap_or(0)
    }

    #[must_use]
    pub fn completion_tokens(&self) -> u64 {
        self.completion_tokens.unwrap_or(0)
    }

    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
            .or_else(|| {
                match (
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.thoughts_tokens,
                ) {
                    (Some(p), Some(c), Some(t)) => Some(p + c + t),
                    (Some(p), Some(c), None) => Some(p + c),
                    _ => None,
                }
            })
            .unwrap_or(0)
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        self.add_assign(&other);
        self
    }

    pub fn add_assign(&mut self, other: &Self) {
        self.prompt_tokens = add_option(self.prompt_tokens, other.prompt_tokens);
        self.completion_tokens = add_option(self.completion_tokens, other.completion_tokens);
        self.total_tokens = add_option(self.total_tokens, other.total_tokens);
        self.cache_creation_tokens =
            add_option(self.cache_creation_tokens, other.cache_creation_tokens);
        self.cache_read_tokens = add_option(self.cache_read_tokens, other.cache_read_tokens);
        self.reasoning_tokens = add_option(self.reasoning_tokens, other.reasoning_tokens);
        self.tool_prompt_tokens = add_option(self.tool_prompt_tokens, other.tool_prompt_tokens);
        self.thoughts_tokens = add_option(self.thoughts_tokens, other.thoughts_tokens);
    }
}

fn add_option(lhs: Option<u64>, rhs: Option<u64>) -> Option<u64> {
    match (lhs, rhs) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        lhs.add_assign(&rhs);
        lhs
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}
