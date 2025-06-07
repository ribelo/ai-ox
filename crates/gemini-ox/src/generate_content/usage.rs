use serde::{Deserialize, Serialize};

/// Enum representing the different types of content modalities.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Modality {
    /// Unspecified modality.
    #[default]
    ModalityUnspecified,
    /// Plain text.
    Text,
    /// Image.
    Image,
    /// Video.
    Video,
    /// Audio.
    Audio,
    /// Document, e.g. PDF.
    Document,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModalityTokenCount {
    /// The modality for which the token count is provided.
    pub modality: Modality,
    /// The number of tokens counted for the specified modality.
    pub token_count: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Number of tokens in the prompt.
    pub prompt_token_count: u32,
    /// Number of tokens in the cached part of the prompt (the cached content).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>,
    /// Total number of tokens across all the generated response candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<u32>,
    /// Output only. Number of tokens present in tool-use prompt(s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_prompt_token_count: Option<u32>,
    /// Output only. Number of tokens of thoughts for thinking models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
    /// Total token count for the generation request (prompt + response candidates).
    pub total_token_count: u32,
    /// Output only. List of modalities that were processed in the request input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// Output only. List of modalities of the cached content in the request input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// Output only. List of modalities that were returned in the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates_tokens_details: Option<Vec<ModalityTokenCount>>,
    /// Output only. List of modalities that were processed for tool-use request inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
}

// Helper function to add Option<u32> values
fn add_optional_u32(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

// Helper function to add Option<Vec<T>> values
fn add_optional_vec<T: Clone>(a: Option<Vec<T>>, b: Option<Vec<T>>) -> Option<Vec<T>> {
    match (a, b) {
        (Some(mut x), Some(y)) => {
            x.extend(y);
            Some(x)
        }
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

// Helper function to add_assign Option<Vec<T>> values
fn add_assign_optional_vec<T: Clone>(a: &mut Option<Vec<T>>, b: Option<Vec<T>>) {
    match b {
        None => {
            // If b is None, do nothing to a.
        }
        Some(vec_b) => {
            // b is Some. Now check the state of a.
            match a {
                Some(vec_a) => {
                    // a is Some, b is Some. Extend vec_a.
                    vec_a.extend(vec_b);
                }
                None => {
                    // a is None, b is Some. Assign vec_b to a.
                    *a = Some(vec_b);
                }
            }
        }
    }
}

impl std::ops::Add for UsageMetadata {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            prompt_token_count: self.prompt_token_count + other.prompt_token_count,
            cached_content_token_count: add_optional_u32(
                self.cached_content_token_count,
                other.cached_content_token_count,
            ),
            candidates_token_count: add_optional_u32(
                self.candidates_token_count,
                other.candidates_token_count,
            ),
            tool_use_prompt_token_count: add_optional_u32(
                self.tool_use_prompt_token_count,
                other.tool_use_prompt_token_count,
            ),
            thoughts_token_count: add_optional_u32(
                self.thoughts_token_count,
                other.thoughts_token_count,
            ),
            total_token_count: self.total_token_count + other.total_token_count,
            prompt_tokens_details: add_optional_vec(
                self.prompt_tokens_details,
                other.prompt_tokens_details,
            ),
            cache_tokens_details: add_optional_vec(
                self.cache_tokens_details,
                other.cache_tokens_details,
            ),
            candidates_tokens_details: add_optional_vec(
                self.candidates_tokens_details,
                other.candidates_tokens_details,
            ),
            tool_use_prompt_tokens_details: add_optional_vec(
                self.tool_use_prompt_tokens_details,
                other.tool_use_prompt_tokens_details,
            ),
        }
    }
}

impl std::ops::AddAssign for UsageMetadata {
    fn add_assign(&mut self, other: Self) {
        self.prompt_token_count += other.prompt_token_count;
        self.cached_content_token_count = add_optional_u32(
            self.cached_content_token_count,
            other.cached_content_token_count,
        );
        self.candidates_token_count =
            add_optional_u32(self.candidates_token_count, other.candidates_token_count);
        self.tool_use_prompt_token_count = add_optional_u32(
            self.tool_use_prompt_token_count,
            other.tool_use_prompt_token_count,
        );
        self.thoughts_token_count =
            add_optional_u32(self.thoughts_token_count, other.thoughts_token_count);
        self.total_token_count += other.total_token_count;

        add_assign_optional_vec(&mut self.prompt_tokens_details, other.prompt_tokens_details);
        add_assign_optional_vec(&mut self.cache_tokens_details, other.cache_tokens_details);
        add_assign_optional_vec(
            &mut self.candidates_tokens_details,
            other.candidates_tokens_details,
        );
        add_assign_optional_vec(
            &mut self.tool_use_prompt_tokens_details,
            other.tool_use_prompt_tokens_details,
        );
    }
}

impl std::iter::Sum for UsageMetadata {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, x| acc + x)
    }
}
