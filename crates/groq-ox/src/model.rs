use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Groq model enumeration
#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Eq, Hash)]
pub enum Model {
    // Llama models
    #[strum(to_string = "llama-3.3-70b-versatile")]
    Llama3_3_70bVersatile,
    #[strum(to_string = "llama-3.3-70b-specdec")]
    Llama3_3_70bSpecdec,
    #[strum(to_string = "llama-3.2-90b-text-preview")]
    Llama3_2_90bTextPreview,
    #[strum(to_string = "llama-3.2-11b-text-preview")]
    Llama3_2_11bTextPreview,
    #[strum(to_string = "llama-3.2-3b-preview")]
    Llama3_2_3bPreview,
    #[strum(to_string = "llama-3.2-1b-preview")]
    Llama3_2_1bPreview,
    #[strum(to_string = "llama-3.1-405b-reasoning")]
    Llama3_1_405bReasoning,
    #[strum(to_string = "llama-3.1-70b-versatile")]
    Llama3_1_70bVersatile,
    #[strum(to_string = "llama-3.1-8b-instant")]
    Llama3_1_8bInstant,
    #[strum(to_string = "llama3-70b-8192")]
    Llama3_70b8192,
    #[strum(to_string = "llama3-8b-8192")]
    Llama3_8b8192,

    // Mixtral models
    #[strum(to_string = "mixtral-8x7b-32768")]
    Mixtral8x7b32768,

    // Gemma models
    #[strum(to_string = "gemma2-9b-it")]
    Gemma2_9bIt,
    #[strum(to_string = "gemma-7b-it")]
    Gemma7bIt,

    // Whisper for speech-to-text
    #[strum(to_string = "whisper-large-v3")]
    WhisperLargeV3,
    #[strum(to_string = "whisper-large-v3-turbo")]
    WhisperLargeV3Turbo,
    #[strum(to_string = "distil-whisper-large-v3-en")]
    DistilWhisperLargeV3En,

    // TTS models
    #[strum(to_string = "tts-1")]
    Tts1,
    #[strum(to_string = "tts-1-hd")]
    Tts1Hd,
}

impl Model {
    /// Check if this model supports tool/function calling
    pub fn supports_tools(&self) -> bool {
        match self {
            // Chat models support tools
            Model::Llama3_3_70bVersatile
            | Model::Llama3_3_70bSpecdec
            | Model::Llama3_2_90bTextPreview
            | Model::Llama3_2_11bTextPreview
            | Model::Llama3_2_3bPreview
            | Model::Llama3_2_1bPreview
            | Model::Llama3_1_405bReasoning
            | Model::Llama3_1_70bVersatile
            | Model::Llama3_1_8bInstant
            | Model::Llama3_70b8192
            | Model::Llama3_8b8192
            | Model::Mixtral8x7b32768
            | Model::Gemma2_9bIt
            | Model::Gemma7bIt => true,

            // Audio models don't support tools
            Model::WhisperLargeV3
            | Model::WhisperLargeV3Turbo
            | Model::DistilWhisperLargeV3En
            | Model::Tts1
            | Model::Tts1Hd => false,
        }
    }

    /// Check if this model is for speech-to-text
    pub fn is_speech_to_text(&self) -> bool {
        matches!(
            self,
            Model::WhisperLargeV3 | Model::WhisperLargeV3Turbo | Model::DistilWhisperLargeV3En
        )
    }

    /// Check if this model is for text-to-speech
    pub fn is_text_to_speech(&self) -> bool {
        matches!(self, Model::Tts1 | Model::Tts1Hd)
    }

    /// Check if this model is for chat completion
    pub fn is_chat_model(&self) -> bool {
        !self.is_speech_to_text() && !self.is_text_to_speech()
    }

    /// Get the context window size for this model
    pub fn context_window(&self) -> Option<u32> {
        match self {
            Model::Llama3_3_70bVersatile => Some(131072),
            Model::Llama3_3_70bSpecdec => Some(8192),
            Model::Llama3_2_90bTextPreview => Some(131072),
            Model::Llama3_2_11bTextPreview => Some(131072),
            Model::Llama3_2_3bPreview => Some(131072),
            Model::Llama3_2_1bPreview => Some(131072),
            Model::Llama3_1_405bReasoning => Some(131072),
            Model::Llama3_1_70bVersatile => Some(131072),
            Model::Llama3_1_8bInstant => Some(131072),
            Model::Llama3_70b8192 => Some(8192),
            Model::Llama3_8b8192 => Some(8192),
            Model::Mixtral8x7b32768 => Some(32768),
            Model::Gemma2_9bIt => Some(8192),
            Model::Gemma7bIt => Some(8192),
            // Audio models don't have context windows in the traditional sense
            _ => None,
        }
    }
}
