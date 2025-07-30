use serde::{Deserialize, Serialize};

/// Represents a Groq model with its capabilities and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The unique identifier for the model
    pub id: String,
    /// The object type (typically "model")
    pub object: String,
    /// Unix timestamp of when the model was created
    pub created: u64,
    /// The organization that owns the model
    pub owned_by: String,
    /// Whether the model is currently active and available
    pub active: bool,
    /// The context window size (maximum tokens) for this model
    pub context_window: u32,
    /// Additional details about the model (may vary by model)
    #[serde(flatten)]
    pub details: serde_json::Value,
}

/// Response from the list models API endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// The object type (typically "list")
    pub object: String,
    /// List of available models
    pub data: Vec<ModelInfo>,
}

impl ModelInfo {
    /// Check if this model supports chat completions
    pub fn supports_chat(&self) -> bool {
        // Most Groq models support chat, except for audio models
        !self.id.contains("whisper") && !self.id.contains("tts")
    }

    /// Check if this model is for speech-to-text
    pub fn is_speech_to_text(&self) -> bool {
        self.id.contains("whisper")
    }

    /// Check if this model is for text-to-speech
    pub fn is_text_to_speech(&self) -> bool {
        self.id.contains("tts")
    }

    /// Get the model family (e.g., "llama", "mixtral", "gemma", "whisper")
    pub fn family(&self) -> &str {
        if self.id.starts_with("llama") {
            "llama"
        } else if self.id.starts_with("mixtral") {
            "mixtral"
        } else if self.id.starts_with("gemma") {
            "gemma"
        } else if self.id.contains("whisper") {
            "whisper"
        } else if self.id.contains("tts") {
            "tts"
        } else {
            "unknown"
        }
    }

    /// Get a human-readable size description based on the model ID
    pub fn size_description(&self) -> Option<&str> {
        if self.id.contains("405b") {
            Some("405B parameters")
        } else if self.id.contains("70b") {
            Some("70B parameters")
        } else if self.id.contains("90b") {
            Some("90B parameters")
        } else if self.id.contains("11b") {
            Some("11B parameters")
        } else if self.id.contains("9b") {
            Some("9B parameters")
        } else if self.id.contains("8b") {
            Some("8B parameters")
        } else if self.id.contains("7b") {
            Some("7B parameters")
        } else if self.id.contains("3b") {
            Some("3B parameters")
        } else if self.id.contains("1b") {
            Some("1B parameters")
        } else if self.id.contains("8x7b") {
            Some("8x7B parameters (Mixture of Experts)")
        } else {
            None
        }
    }
}