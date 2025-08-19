use futures_util::{future::BoxFuture, FutureExt};
use base64::Engine;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use crate::stt::{
    SpeechToText, SttInfo, SttModel, SttError,
    TranscriptionRequest, TranscriptionResponse, 
    AudioSource, AudioFormat, OutputFormat, TimestampGranularity,
    Segment, Word, SttUsage,
};

use groq_ox::audio::transcription::{TranscriptionFormat, TimestampGranularity as GroqTimestampGranularity};

/// Groq STT provider implementation
#[derive(Debug, Clone)]
pub struct GroqStt {
    client: groq_ox::Groq,
    model: String,
}

/// Builder for GroqStt
pub struct GroqSttBuilder {
    model: Option<String>,
    api_key: Option<String>,
}

impl GroqSttBuilder {
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn api_key_from_env(mut self, env_var: &str) -> Result<Self, SttError> {
        let key = std::env::var(env_var)
            .map_err(|_| SttError::MissingApiKey)?;
        self.api_key = Some(key);
        Ok(self)
    }

    pub fn build(self) -> Result<Arc<dyn SpeechToText>, SttError> {
        let api_key = self.api_key.ok_or(SttError::MissingApiKey)?;
        let model = self.model.unwrap_or_else(|| "whisper-large-v3".to_string());
        
        let client = groq_ox::Groq::new(&api_key);
        Ok(Arc::new(GroqStt::new(client, model)))
    }
}

impl Default for GroqSttBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GroqStt {
    /// Create a new Groq STT provider
    pub fn new(client: groq_ox::Groq, model: String) -> Self {
        Self { client, model }
    }

    /// Create a builder for GroqStt
    pub fn builder() -> GroqSttBuilder {
        GroqSttBuilder::new()
    }

    /// Convert unified audio source to Groq format
    fn convert_audio_source(&self, source: AudioSource) -> Result<Vec<u8>, SttError> {
        match source {
            AudioSource::Bytes { data, .. } => Ok(data),
            AudioSource::File(path) => {
                std::fs::read(&path).map_err(|e| {
                    SttError::InvalidConfig(format!("Failed to read file {:?}: {}", path, e))
                })
            }
            AudioSource::Base64 { data, .. } => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| SttError::InvalidAudioData(format!("Invalid base64: {}", e)))
            }
            AudioSource::Url(_) => {
                Err(SttError::InvalidConfig("URL audio sources not supported by Groq".to_string()))
            }
            AudioSource::RecordingId(_) => {
                Err(SttError::InvalidConfig("Recording ID sources not supported by Groq".to_string()))
            }
        }
    }

    /// Convert unified request to Groq format
    fn convert_request(&self, request: TranscriptionRequest) -> Result<groq_ox::audio::TranscriptionRequest, SttError> {
        let audio_data = self.convert_audio_source(request.audio)?;

        // Convert output format
        let response_format = match request.output_format {
            OutputFormat::Simple => Some(TranscriptionFormat::Text),
            OutputFormat::Detailed => Some(TranscriptionFormat::Json),
            OutputFormat::Verbose => Some(TranscriptionFormat::VerboseJson),
        };

        // Convert timestamps
        let timestamp_granularities = match request.timestamps {
            TimestampGranularity::None => None,
            TimestampGranularity::Segment => Some(vec![GroqTimestampGranularity::Segment]),
            TimestampGranularity::Word => Some(vec![
                GroqTimestampGranularity::Segment,
                GroqTimestampGranularity::Word,
            ]),
        };

        // Create the basic request structure
        let groq_request = groq_ox::audio::TranscriptionRequest {
            file: audio_data,
            model: self.model.clone(),
            language: request.language,
            prompt: request.prompt,
            response_format,
            temperature: if request.temperature > 0.0 { Some(request.temperature) } else { None },
            timestamp_granularities,
        };

        Ok(groq_request)
    }

    /// Convert Groq response to unified format
    fn convert_response(&self, groq_response: groq_ox::audio::TranscriptionResponse) -> TranscriptionResponse {
        let mut response = TranscriptionResponse::simple(
            groq_response.text,
            "groq".to_string(),
            self.model.clone(),
        );

        response.language = groq_response.language;
        response.duration = groq_response.duration.map(Duration::from_secs_f32);

        // Convert segments if available
        if let Some(groq_segments) = groq_response.segments {
            response.segments = groq_segments
                .into_iter()
                .map(|seg| {
                    Segment::new(
                        seg.text,
                        Duration::from_secs_f32(seg.start),
                        Duration::from_secs_f32(seg.end),
                    )
                    .with_id(seg.id as u32)
                    // Use a reasonable default confidence since Groq doesn't provide actual confidence scores
                    .with_confidence(0.95)
                })
                .collect();
        }

        // Convert words if available
        if let Some(groq_words) = groq_response.words {
            response.words = groq_words
                .into_iter()
                .map(|word| {
                    Word::new(
                        word.word,
                        Duration::from_secs_f32(word.start),
                        Duration::from_secs_f32(word.end),
                    )
                })
                .collect();
        }

        // Set usage information
        if let Some(duration) = response.duration {
            response.usage = SttUsage::new(duration);
            response.usage.characters_transcribed = response.text.len() as u32;
        }

        response
    }
}

/// Available Groq STT models with metadata
static GROQ_MODELS: LazyLock<Vec<SttModel>> = LazyLock::new(|| {
    vec![
        SttModel {
            id: "whisper-large-v3".to_string(),
            name: "Whisper Large V3".to_string(),
            description: Some("Most accurate Whisper model".to_string()),
            supported_formats: vec![
                AudioFormat::Mp3,
                AudioFormat::Wav,
                AudioFormat::Flac,
                AudioFormat::M4a,
                AudioFormat::WebM,
            ],
            max_duration: Some(Duration::from_secs(25 * 60)), // 25 minutes
            supports_streaming: false,
            supported_languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()], // Simplified
        },
        SttModel {
            id: "whisper-large-v3-turbo".to_string(),
            name: "Whisper Large V3 Turbo".to_string(),
            description: Some("Faster Whisper model with good accuracy".to_string()),
            supported_formats: vec![
                AudioFormat::Mp3,
                AudioFormat::Wav,
                AudioFormat::Flac,
                AudioFormat::M4a,
                AudioFormat::WebM,
            ],
            max_duration: Some(Duration::from_secs(25 * 60)),
            supports_streaming: false,
            supported_languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
        },
        SttModel {
            id: "distil-whisper-large-v3-en".to_string(),
            name: "Distil-Whisper Large V3 English".to_string(),
            description: Some("English-only optimized Whisper model".to_string()),
            supported_formats: vec![
                AudioFormat::Mp3,
                AudioFormat::Wav,
                AudioFormat::Flac,
                AudioFormat::M4a,
                AudioFormat::WebM,
            ],
            max_duration: Some(Duration::from_secs(25 * 60)),
            supports_streaming: false,
            supported_languages: vec!["en".to_string()],
        },
    ]
});

impl SpeechToText for GroqStt {
    fn info(&self) -> SttInfo<'_> {
        SttInfo::new("groq", &self.model)
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, SttError>> {
        async move {
            let groq_request = self.convert_request(request)?;
            let groq_response = self.client.transcribe(&groq_request).await?;
            Ok(self.convert_response(groq_response))
        }
        .boxed()
    }


    fn available_models(&self) -> BoxFuture<'_, Result<Vec<SttModel>, SttError>> {
        async move {
            // Could potentially call the Groq models API here
            // For now, return static list
            Ok(GROQ_MODELS.clone())
        }
        .boxed()
    }

    fn supports_format(&self, format: AudioFormat) -> bool {
        matches!(
            format,
            AudioFormat::Mp3
                | AudioFormat::Wav
                | AudioFormat::Flac
                | AudioFormat::M4a
                | AudioFormat::WebM
        )
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_source_conversion() {
        let client = groq_ox::Groq::new("test-key");
        let groq_stt = GroqStt::new(client, "whisper-large-v3".to_string());

        // Test bytes conversion
        let audio_data = vec![1, 2, 3, 4];
        let source = AudioSource::from_bytes(audio_data.clone(), AudioFormat::Mp3);
        let result = groq_stt.convert_audio_source(source).unwrap();
        assert_eq!(result, audio_data);

        // Test base64 conversion
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&audio_data);
        let source = AudioSource::from_base64(base64_data, AudioFormat::Mp3);
        let result = groq_stt.convert_audio_source(source).unwrap();
        assert_eq!(result, audio_data);
    }

    #[test]
    fn test_format_support() {
        let client = groq_ox::Groq::new("test-key");
        let groq_stt = GroqStt::new(client, "whisper-large-v3".to_string());

        assert!(groq_stt.supports_format(AudioFormat::Mp3));
        assert!(groq_stt.supports_format(AudioFormat::Wav));
        assert!(groq_stt.supports_format(AudioFormat::Flac));
        assert!(!groq_stt.supports_format(AudioFormat::Ogg));
    }

    // Note: supports_streaming() method was removed from trait
    // Streaming is not supported by current providers

    #[test]
    fn test_model_info() {
        let client = groq_ox::Groq::new("test-key");
        let groq_stt = GroqStt::new(client, "whisper-large-v3".to_string());

        let info = groq_stt.info();
        assert_eq!(info.provider, "groq");
        assert_eq!(info.model, "whisper-large-v3");
        assert_eq!(groq_stt.model(), "whisper-large-v3");
    }
}