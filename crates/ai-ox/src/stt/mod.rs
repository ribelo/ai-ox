pub mod error;
pub mod request;
pub mod response;
pub mod streaming;
pub mod builder;
pub mod providers;

pub use error::SttError;
pub use request::{TranscriptionRequest, AudioSource, AudioFormat, OutputFormat, TimestampGranularity};
pub use response::{TranscriptionResponse, Alternative, Segment, Word, SttUsage};

#[cfg(feature = "groq")]
pub use builder::groq_stt;

#[cfg(feature = "mistral")]
pub use builder::mistral_stt;

#[cfg(feature = "gemini")]
pub use builder::gemini_stt;

use futures_util::future::BoxFuture;
use std::sync::Arc;

/// Provider information containing provider name and model identifier
#[derive(Debug, Clone)]
pub struct SttInfo<'a> {
    pub provider: &'a str,
    pub model: &'a str,
}

impl<'a> SttInfo<'a> {
    pub fn new(provider: &'a str, model: &'a str) -> Self {
        Self { provider, model }
    }
}

impl<'a> std::fmt::Display for SttInfo<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

/// Model information for STT providers
#[derive(Debug, Clone)]
pub struct SttModel {
    /// Model identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model description
    pub description: Option<String>,
    /// Supported audio formats
    pub supported_formats: Vec<AudioFormat>,
    /// Maximum audio duration
    pub max_duration: Option<std::time::Duration>,
    /// Whether model supports streaming
    pub supports_streaming: bool,
    /// Supported languages (ISO-639-1 codes)
    pub supported_languages: Vec<String>,
}

/// The primary trait for Speech-to-Text providers
///
/// This trait provides a standardized interface for transcribing audio across
/// different STT providers. It supports both single-shot and streaming transcription.
pub trait SpeechToText: Send + Sync + 'static + std::fmt::Debug {
    /// Returns the provider information containing provider and model identifier
    ///
    /// # Returns
    ///
    /// A `SttInfo` containing the provider and model identifier
    fn info(&self) -> SttInfo<'_>;

    /// Returns the model name
    ///
    /// # Returns
    ///
    /// A string slice containing the model name
    fn model(&self) -> &str;

    /// Transcribes a single audio file or input
    ///
    /// # Arguments
    ///
    /// * `request` - A TranscriptionRequest containing audio and configuration
    ///
    /// # Returns
    ///
    /// A `Result` containing either the `TranscriptionResponse` or an `SttError`
    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, SttError>>;


    /// Lists available models/engines for this provider
    ///
    /// # Returns
    ///
    /// A `Result` containing either a list of `SttModel` or an `SttError`
    fn available_models(&self) -> BoxFuture<'_, Result<Vec<SttModel>, SttError>>;

    /// Checks if this provider supports a given audio format
    fn supports_format(&self, format: AudioFormat) -> bool {
        // Default implementation - providers can override
        matches!(format, AudioFormat::Mp3 | AudioFormat::Wav | AudioFormat::Flac)
    }

}

impl<T: SpeechToText + ?Sized> SpeechToText for Arc<T> {
    fn info(&self) -> SttInfo<'_> {
        self.as_ref().info()
    }

    fn model(&self) -> &str {
        self.as_ref().model()
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, SttError>> {
        self.as_ref().transcribe(request)
    }


    fn available_models(&self) -> BoxFuture<'_, Result<Vec<SttModel>, SttError>> {
        self.as_ref().available_models()
    }

    fn supports_format(&self, format: AudioFormat) -> bool {
        self.as_ref().supports_format(format)
    }
}