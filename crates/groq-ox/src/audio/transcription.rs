use crate::GroqRequestError;
use bon::Builder;
use serde::{Deserialize, Serialize};

/// Request for audio transcription using Whisper models
#[derive(Debug, Clone, Serialize, Builder)]
pub struct TranscriptionRequest {
    /// The audio file to transcribe
    #[serde(skip)]
    pub file: Vec<u8>,

    /// The model to use for transcription (whisper-large-v3, etc.)
    #[builder(into)]
    pub model: String,

    /// The language of the input audio (optional, ISO-639-1 format)
    #[builder(into)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// The prompt to guide the model's style or continue a previous audio segment
    #[builder(into)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// The format of the transcript output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<TranscriptionFormat>,

    /// The sampling temperature (0 makes output more deterministic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// The timestamp granularities to populate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
}

/// The format of the transcript output
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, strum::Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TranscriptionFormat {
    Json,
    Text,
    VerboseJson,
}

/// The level of timestamp detail
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, strum::Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum TimestampGranularity {
    Word,
    Segment,
}

/// Response from audio transcription
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text
    pub text: String,

    /// The language of the input audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// The duration of the input audio
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f32>,

    /// Segments of the transcribed text (verbose_json format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,

    /// Words with timestamps (verbose_json format with word granularity)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionWord>>,
}

/// A segment of transcribed text
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionSegment {
    /// Unique identifier of the segment
    pub id: i32,

    /// Start time of the segment in seconds
    pub start: f32,

    /// End time of the segment in seconds
    pub end: f32,

    /// Text content of the segment
    pub text: String,

    /// Temperature parameter used for this segment
    pub temperature: f32,

    /// Average logprob of the segment
    pub avg_logprob: f32,

    /// Compression ratio of the segment
    pub compression_ratio: f32,

    /// Probability of no speech in the segment
    pub no_speech_prob: f32,
}

/// A word with timing information
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionWord {
    /// The text content of the word
    pub word: String,

    /// Start time of the word in seconds
    pub start: f32,

    /// End time of the word in seconds
    pub end: f32,
}

impl crate::Groq {
    /// Transcribe audio using Groq's Whisper models
    pub async fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResponse, GroqRequestError> {
        // Validate that timestamp_granularities requires verbose_json format
        if request.timestamp_granularities.is_some()
            && request.response_format != Some(TranscriptionFormat::VerboseJson)
        {
            return Err(GroqRequestError::InvalidRequest {
                code: None,
                message: "`timestamp_granularities` requires `response_format = verbose_json`"
                    .into(),
                details: Some(serde_json::json!({"type": "validation_error"})),
            });
        }

        let url = format!("{}/openai/v1/audio/transcriptions", self.base_url);

        // Create multipart form
        let mut form = reqwest::multipart::Form::new().text("model", request.model.clone());

        if let Some(language) = &request.language {
            form = form.text("language", language.clone());
        }

        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }

        if let Some(format) = &request.response_format {
            form = form.text("response_format", format.to_string());
        }

        if let Some(temp) = request.temperature {
            form = form.text("temperature", temp.to_string());
        }

        if let Some(granularities) = &request.timestamp_granularities {
            for g in granularities {
                form = form.text("timestamp_granularities[]", g.to_string());
            }
        }

        // Add the audio file
        let part = reqwest::multipart::Part::bytes(request.file.clone()).file_name("audio.mp3");
        form = form.part("file", part);

        let res = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?;

        if res.status().is_success() {
            Ok(res.json::<TranscriptionResponse>().await?)
        } else {
            let status = res.status();
            let bytes = res.bytes().await?;
            Err(crate::error::parse_error_response(status, bytes))
        }
    }
}
