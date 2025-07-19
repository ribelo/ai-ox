#[cfg(feature = "gemini")]
pub mod gemini;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::ops::{Add, AddAssign};

/// Represents different content modalities for token counting.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Plain text content
    Text,
    /// Image content
    Image,
    /// Video content
    Video,
    /// Audio content
    Audio,
    /// Document content (e.g., PDF)
    Document,
    /// Other/unspecified modality
    Other(String),
}

/// Helper function to merge modality token maps
fn add_modality_maps(lhs: &mut HashMap<Modality, u64>, rhs: &HashMap<Modality, u64>) {
    for (modality, count) in rhs {
        *lhs.entry(modality.clone()).or_default() += count;
    }
}

/// Simple, clean usage tracking for AI model interactions.
///
/// Following the principle: store only essential data, calculate the rest.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of requests made to the model
    pub requests: u64,
    /// Input tokens broken down by modality
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub input_tokens_by_modality: HashMap<Modality, u64>,
    /// Output tokens broken down by modality
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub output_tokens_by_modality: HashMap<Modality, u64>,
    /// Cache tokens broken down by modality
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub cache_tokens_by_modality: HashMap<Modality, u64>,
    /// Tool usage tokens broken down by modality
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub tool_tokens_by_modality: HashMap<Modality, u64>,
    /// Tokens from cached content (when caching is used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Tokens from cache creation (when content is first cached)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
    /// Thoughts tokens (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_tokens: Option<u64>,
    /// Additional provider-specific usage details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl Usage {
    /// Creates a new Usage instance with all values set to zero.
    pub fn new() -> Self {
        Self {
            requests: 0,
            input_tokens_by_modality: HashMap::new(),
            output_tokens_by_modality: HashMap::new(),
            cache_tokens_by_modality: HashMap::new(),
            tool_tokens_by_modality: HashMap::new(),
            cache_read_tokens: None,
            cache_creation_tokens: None,
            thoughts_tokens: None,
            details: None,
        }
    }

    /// Calculate total input tokens across all modalities.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens_by_modality.values().sum()
    }

    /// Calculate total output tokens across all modalities.
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens_by_modality.values().sum()
    }

    /// Calculate total cache tokens across all modalities.
    pub fn cache_tokens(&self) -> u64 {
        self.cache_tokens_by_modality.values().sum()
    }

    /// Calculate total tool tokens across all modalities.
    pub fn tool_tokens(&self) -> u64 {
        self.tool_tokens_by_modality.values().sum()
    }

    /// Calculate total tokens (input + output + thoughts).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens() + self.output_tokens() + self.thoughts_tokens.unwrap_or(0)
    }

    /// Returns the effective input tokens (excluding cache creation cost).
    /// This represents the actual computational cost for input processing.
    pub fn effective_input_tokens(&self) -> u64 {
        let cache_creation = self.cache_creation_tokens.unwrap_or(0);
        // Cache creation tokens are typically billed differently
        self.input_tokens().saturating_sub(cache_creation)
    }

    /// Returns the total cache-related tokens.
    pub fn total_cache_tokens(&self) -> u64 {
        self.cache_read_tokens.unwrap_or(0) + self.cache_creation_tokens.unwrap_or(0)
    }
}

impl Add for Usage {
    type Output = Usage;

    fn add(mut self, other: Usage) -> Usage {
        self.requests += other.requests;
        add_modality_maps(
            &mut self.input_tokens_by_modality,
            &other.input_tokens_by_modality,
        );
        add_modality_maps(
            &mut self.output_tokens_by_modality,
            &other.output_tokens_by_modality,
        );
        add_modality_maps(
            &mut self.cache_tokens_by_modality,
            &other.cache_tokens_by_modality,
        );
        add_modality_maps(
            &mut self.tool_tokens_by_modality,
            &other.tool_tokens_by_modality,
        );
        self.cache_read_tokens = add_optional_u64(self.cache_read_tokens, other.cache_read_tokens);
        self.cache_creation_tokens =
            add_optional_u64(self.cache_creation_tokens, other.cache_creation_tokens);
        self.thoughts_tokens = add_optional_u64(self.thoughts_tokens, other.thoughts_tokens);
        self.details = merge_details(self.details, other.details);
        self
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.requests += rhs.requests;
        add_modality_maps(
            &mut self.input_tokens_by_modality,
            &rhs.input_tokens_by_modality,
        );
        add_modality_maps(
            &mut self.output_tokens_by_modality,
            &rhs.output_tokens_by_modality,
        );
        add_modality_maps(
            &mut self.cache_tokens_by_modality,
            &rhs.cache_tokens_by_modality,
        );
        add_modality_maps(
            &mut self.tool_tokens_by_modality,
            &rhs.tool_tokens_by_modality,
        );
        self.cache_read_tokens = add_optional_u64(self.cache_read_tokens, rhs.cache_read_tokens);
        self.cache_creation_tokens =
            add_optional_u64(self.cache_creation_tokens, rhs.cache_creation_tokens);
        self.thoughts_tokens = add_optional_u64(self.thoughts_tokens, rhs.thoughts_tokens);
        self.details = merge_details(self.details.take(), rhs.details);
    }
}

// Helper functions

fn add_optional_u64(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn merge_details(a: Option<Value>, b: Option<Value>) -> Option<Value> {
    match (a, b) {
        (Some(Value::Object(mut a_map)), Some(Value::Object(b_map))) => {
            for (key, value) in b_map {
                a_map.insert(key, value);
            }
            Some(Value::Object(a_map))
        }
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
        // If not both objects, just take the second one
        (_, Some(b)) => Some(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_new() {
        let usage = Usage::new();
        assert_eq!(usage.requests, 0);
        assert_eq!(usage.input_tokens(), 0);
        assert_eq!(usage.output_tokens(), 0);
        assert_eq!(usage.total_tokens(), 0);
        assert!(usage.input_tokens_by_modality.is_empty());
        assert!(usage.output_tokens_by_modality.is_empty());
    }

    #[test]
    fn test_usage_with_modality_tokens() {
        let mut usage = Usage::new();
        usage.requests = 1;
        usage.input_tokens_by_modality.insert(Modality::Text, 100);
        usage.input_tokens_by_modality.insert(Modality::Image, 50);
        usage.output_tokens_by_modality.insert(Modality::Text, 30);
        usage.cache_read_tokens = Some(20);
        usage.cache_creation_tokens = Some(10);
        usage.thoughts_tokens = Some(5);

        assert_eq!(usage.requests, 1);
        assert_eq!(usage.input_tokens(), 150); // 100 + 50
        assert_eq!(usage.output_tokens(), 30);
        assert_eq!(usage.total_tokens(), 185); // 150 + 30 + 5
        assert_eq!(usage.total_cache_tokens(), 30); // 20 + 10
        assert_eq!(usage.effective_input_tokens(), 140); // 150 - 10
    }

    #[test]
    fn test_add_modality_maps() {
        let mut map1 = HashMap::new();
        map1.insert(Modality::Text, 100);
        map1.insert(Modality::Image, 50);

        let mut map2 = HashMap::new();
        map2.insert(Modality::Text, 25); // Should merge
        map2.insert(Modality::Audio, 75); // Should add new

        add_modality_maps(&mut map1, &map2);

        assert_eq!(*map1.get(&Modality::Text).unwrap(), 125); // 100 + 25
        assert_eq!(*map1.get(&Modality::Image).unwrap(), 50);
        assert_eq!(*map1.get(&Modality::Audio).unwrap(), 75);
        assert_eq!(map1.len(), 3);
    }

    #[test]
    fn test_usage_add_assign() {
        let mut usage1 = Usage::new();
        usage1.requests = 1;
        usage1.input_tokens_by_modality.insert(Modality::Text, 100);
        usage1.output_tokens_by_modality.insert(Modality::Text, 50);
        usage1.cache_read_tokens = Some(10);

        let mut usage2 = Usage::new();
        usage2.requests = 2;
        usage2.input_tokens_by_modality.insert(Modality::Text, 25); // Merge with existing
        usage2.input_tokens_by_modality.insert(Modality::Image, 30); // New modality
        usage2.output_tokens_by_modality.insert(Modality::Image, 15);
        usage2.cache_creation_tokens = Some(5);
        usage2.thoughts_tokens = Some(8);

        usage1 += usage2;

        assert_eq!(usage1.requests, 3);
        assert_eq!(usage1.input_tokens(), 155); // 100 + 25 + 30
        assert_eq!(usage1.output_tokens(), 65); // 50 + 15
        assert_eq!(usage1.total_tokens(), 228); // 155 + 65 + 8
        assert_eq!(
            *usage1
                .input_tokens_by_modality
                .get(&Modality::Text)
                .unwrap(),
            125
        ); // 100 + 25
        assert_eq!(
            *usage1
                .input_tokens_by_modality
                .get(&Modality::Image)
                .unwrap(),
            30
        );
        assert_eq!(usage1.cache_read_tokens, Some(10));
        assert_eq!(usage1.cache_creation_tokens, Some(5));
        assert_eq!(usage1.thoughts_tokens, Some(8));
    }

    #[test]
    fn test_usage_add() {
        let mut usage1 = Usage::new();
        usage1.requests = 1;
        usage1.input_tokens_by_modality.insert(Modality::Text, 80);
        usage1.output_tokens_by_modality.insert(Modality::Text, 40);
        usage1.cache_read_tokens = Some(15);

        let mut usage2 = Usage::new();
        usage2.requests = 1;
        usage2.input_tokens_by_modality.insert(Modality::Text, 20);
        usage2.input_tokens_by_modality.insert(Modality::Video, 60);
        usage2.output_tokens_by_modality.insert(Modality::Video, 25);
        usage2.cache_creation_tokens = Some(12);

        let result = usage1 + usage2;

        assert_eq!(result.requests, 2);
        assert_eq!(result.input_tokens(), 160); // 80 + 20 + 60
        assert_eq!(result.output_tokens(), 65); // 40 + 25
        assert_eq!(result.total_tokens(), 225); // 160 + 65
        assert_eq!(
            *result
                .input_tokens_by_modality
                .get(&Modality::Text)
                .unwrap(),
            100
        ); // 80 + 20
        assert_eq!(
            *result
                .input_tokens_by_modality
                .get(&Modality::Video)
                .unwrap(),
            60
        );
        assert_eq!(result.cache_read_tokens, Some(15));
        assert_eq!(result.cache_creation_tokens, Some(12));
    }

    #[test]
    fn test_usage_calculations_empty() {
        let usage = Usage::new();
        assert_eq!(usage.input_tokens(), 0);
        assert_eq!(usage.output_tokens(), 0);
        assert_eq!(usage.cache_tokens(), 0);
        assert_eq!(usage.tool_tokens(), 0);
        assert_eq!(usage.total_tokens(), 0);
        assert_eq!(usage.effective_input_tokens(), 0);
        assert_eq!(usage.total_cache_tokens(), 0);
    }
}
