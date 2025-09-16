use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::MistralRequestError;

/// Request for audio transcription
#[derive(Debug, Clone, Serialize, Builder)]
pub struct TranscriptionRequest {
    /// The audio file to transcribe
    #[serde(skip)]
    pub file: Vec<u8>,

    /// The model to use for transcription
    #[builder(into)]
    pub model: String,

    /// The language of the input audio (optional)
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionFormat {
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

/// The level of timestamp detail
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

    /// Segments of the transcribed text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,

    /// Words with timestamps
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


