use crate::GroqRequestError;
use bon::Builder;
use serde::{Deserialize, Serialize};

/// Request for text-to-speech synthesis
#[derive(Debug, Clone, Serialize, Builder)]
pub struct SpeechRequest {
    /// The model to use for speech synthesis
    #[builder(into)]
    pub model: String,

    /// The text to convert to speech
    #[builder(into)]
    pub input: String,

    /// The voice to use for synthesis
    #[builder(into)]
    pub voice: String,

    /// The format of the output audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<AudioFormat>,

    /// The speed of the generated audio (0.25 to 4.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

/// Audio output format
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

/// Response from text-to-speech synthesis (raw audio bytes)
pub struct SpeechResponse {
    /// The generated audio data
    pub audio: Vec<u8>,
    /// The content type of the audio
    pub content_type: String,
}

impl crate::Groq {
    /// Generate speech from text using Groq's TTS models
    pub async fn speech(
        &self,
        request: &SpeechRequest,
    ) -> Result<SpeechResponse, GroqRequestError> {
        let url = format!("{}/openai/v1/audio/speech", self.base_url);

        let res = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await?;

        if res.status().is_success() {
            let content_type = res
                .headers()
                .get("content-type")
                .and_then(|ct| ct.to_str().ok())
                .unwrap_or("audio/mpeg")
                .to_string();

            let audio = res.bytes().await?.to_vec();

            Ok(SpeechResponse {
                audio,
                content_type,
            })
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(crate::error::parse_error_response(status, bytes))
        }
    }
}
