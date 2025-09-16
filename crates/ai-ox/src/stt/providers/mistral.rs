use base64::Engine;
use futures_util::{FutureExt, future::BoxFuture};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use crate::stt::{
    AudioFormat, AudioSource, OutputFormat, Segment, SpeechToText, SttError, SttInfo, SttModel,
    SttUsage, TimestampGranularity, TranscriptionRequest, TranscriptionResponse, Word,
};

use mistral_ox::audio::{TimestampGranularity as MistralTimestampGranularity, TranscriptionFormat};

/// Mistral STT provider implementation using Voxtral models
#[derive(Debug, Clone)]
pub struct MistralStt {
    client: mistral_ox::Mistral,
    model: String,
}

/// Builder for MistralStt
pub struct MistralSttBuilder {
    model: Option<String>,
    api_key: Option<String>,
}

impl MistralSttBuilder {
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
        let key = std::env::var(env_var).map_err(|_| SttError::MissingApiKey)?;
        self.api_key = Some(key);
        Ok(self)
    }

    pub fn build(self) -> Result<Arc<dyn SpeechToText>, SttError> {
        let api_key = self.api_key.ok_or(SttError::MissingApiKey)?;
        let model = self
            .model
            .unwrap_or_else(|| "voxtral-large-24-05".to_string());

        let client = mistral_ox::Mistral::new(&api_key);
        Ok(Arc::new(MistralStt::new(client, model)))
    }
}

impl Default for MistralSttBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MistralStt {
    /// Create a new Mistral STT provider
    pub fn new(client: mistral_ox::Mistral, model: String) -> Self {
        Self { client, model }
    }

    /// Create a builder for MistralStt
    pub fn builder() -> MistralSttBuilder {
        MistralSttBuilder::new()
    }

    /// Convert unified audio source to Mistral format (bytes)
    fn convert_audio_source(&self, source: AudioSource) -> Result<Vec<u8>, SttError> {
        match source {
            AudioSource::Bytes { data, .. } => Ok(data),
            AudioSource::File(path) => std::fs::read(&path).map_err(|e| {
                SttError::InvalidConfig(format!("Failed to read file {:?}: {}", path, e))
            }),
            AudioSource::Base64 { data, .. } => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|e| SttError::InvalidAudioData(format!("Invalid base64: {}", e)))
            }
            AudioSource::Url(_) => Err(SttError::InvalidConfig(
                "URL audio sources not supported by Mistral".to_string(),
            )),
            AudioSource::RecordingId(_) => Err(SttError::InvalidConfig(
                "Recording ID sources not supported by Mistral".to_string(),
            )),
        }
    }

    /// Convert unified request to Mistral format
    fn convert_request(
        &self,
        request: TranscriptionRequest,
    ) -> Result<mistral_ox::audio::TranscriptionRequest, SttError> {
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
            TimestampGranularity::Segment => Some(vec![MistralTimestampGranularity::Segment]),
            TimestampGranularity::Word => Some(vec![
                MistralTimestampGranularity::Segment,
                MistralTimestampGranularity::Word,
            ]),
        };

        // Create the request manually since the builder pattern is complex for optional fields
        let mistral_request = mistral_ox::audio::TranscriptionRequest {
            file: audio_data,
            model: self.model.clone(),
            language: request.language,
            prompt: request.prompt,
            response_format,
            temperature: if request.temperature > 0.0 {
                Some(request.temperature)
            } else {
                None
            },
            timestamp_granularities,
        };

        Ok(mistral_request)
    }

    /// Convert Mistral response to unified format
    fn convert_response(
        &self,
        mistral_response: mistral_ox::audio::TranscriptionResponse,
    ) -> TranscriptionResponse {
        let mut response = TranscriptionResponse::simple(
            mistral_response.text,
            "mistral".to_string(),
            self.model.clone(),
        );

        response.language = mistral_response.language;
        response.duration = mistral_response.duration.map(Duration::from_secs_f32);

        // Convert segments if available
        if let Some(mistral_segments) = mistral_response.segments {
            response.segments = mistral_segments
                .into_iter()
                .map(|seg| {
                    Segment::new(
                        seg.text,
                        Duration::from_secs_f32(seg.start),
                        Duration::from_secs_f32(seg.end),
                    )
                    .with_id(seg.id as u32)
                    // Use logprob-based confidence approximation
                    .with_confidence((seg.avg_logprob.exp() * 0.95).max(0.1).min(1.0))
                })
                .collect();
        }

        // Convert words if available
        if let Some(mistral_words) = mistral_response.words {
            response.words = mistral_words
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

/// Available Mistral STT models with metadata
static MISTRAL_MODELS: LazyLock<Vec<SttModel>> = LazyLock::new(|| {
    vec![SttModel {
        id: "voxtral-large-24-05".to_string(),
        name: "Voxtral Large 24.05".to_string(),
        description: Some("Mistral's flagship multilingual audio transcription model".to_string()),
        supported_formats: vec![
            AudioFormat::Mp3,
            AudioFormat::Wav,
            AudioFormat::Flac,
            AudioFormat::M4a,
            AudioFormat::WebM,
            AudioFormat::Ogg,
        ],
        max_duration: Some(Duration::from_secs(30 * 60)), // 30 minutes
        supports_streaming: false,
        supported_languages: vec![
            "en".to_string(),
            "fr".to_string(),
            "es".to_string(),
            "de".to_string(),
            "it".to_string(),
            "nl".to_string(),
            "pt".to_string(),
            "ru".to_string(),
            "zh".to_string(),
            "ja".to_string(),
            "ar".to_string(),
        ],
    }]
});

impl SpeechToText for MistralStt {
    fn info(&self) -> SttInfo<'_> {
        SttInfo::new("mistral", &self.model)
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'_, Result<TranscriptionResponse, SttError>> {
        async move {
            let mistral_request = self.convert_request(request)?;
            let mistral_response = self
                .client
                .transcribe(&mistral_request)
                .await
                .map_err(SttError::from)?;
            Ok(self.convert_response(mistral_response))
        }
        .boxed()
    }

    fn available_models(&self) -> BoxFuture<'_, Result<Vec<SttModel>, SttError>> {
        async move {
            // Could potentially call the Mistral models API here
            // For now, return static list
            Ok(MISTRAL_MODELS.clone())
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
                | AudioFormat::Ogg
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_source_conversion() {
        let client = mistral_ox::Mistral::new("test-key");
        let mistral_stt = MistralStt::new(client, "voxtral-large-24-05".to_string());

        // Test bytes conversion
        let audio_data = vec![1, 2, 3, 4];
        let source = AudioSource::from_bytes(audio_data.clone(), AudioFormat::Mp3);
        let result = mistral_stt.convert_audio_source(source).unwrap();
        assert_eq!(result, audio_data);

        // Test base64 conversion
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&audio_data);
        let source = AudioSource::from_base64(base64_data, AudioFormat::Mp3);
        let result = mistral_stt.convert_audio_source(source).unwrap();
        assert_eq!(result, audio_data);
    }

    #[test]
    fn test_format_support() {
        let client = mistral_ox::Mistral::new("test-key");
        let mistral_stt = MistralStt::new(client, "voxtral-large-24-05".to_string());

        assert!(mistral_stt.supports_format(AudioFormat::Mp3));
        assert!(mistral_stt.supports_format(AudioFormat::Wav));
        assert!(mistral_stt.supports_format(AudioFormat::Flac));
        assert!(mistral_stt.supports_format(AudioFormat::Ogg));
    }

    #[test]
    fn test_model_info() {
        let client = mistral_ox::Mistral::new("test-key");
        let mistral_stt = MistralStt::new(client, "voxtral-large-24-05".to_string());

        let info = mistral_stt.info();
        assert_eq!(info.provider, "mistral");
        assert_eq!(info.model, "voxtral-large-24-05");
        assert_eq!(mistral_stt.model(), "voxtral-large-24-05");
    }
}
