use bon::Builder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Request for audio transcription
#[derive(Debug, Clone, Builder)]
pub struct TranscriptionRequest {
    /// Audio source - flexible input
    pub audio: AudioSource,
    
    /// Target language (ISO-639-1) - None for auto-detect
    #[builder(into)]
    pub language: Option<String>,
    
    /// Prompt to guide transcription style
    #[builder(into)]
    pub prompt: Option<String>,
    
    /// Desired output detail level
    #[builder(default = OutputFormat::Simple)]
    pub output_format: OutputFormat,
    
    /// Temperature for transcription creativity (0.0 - 1.0)
    #[builder(default = 0.0)]
    pub temperature: f32,
    
    /// Whether to include timestamps
    #[builder(default = TimestampGranularity::None)]
    pub timestamps: TimestampGranularity,
    
    /// Provider-specific options
    #[builder(default)]
    pub vendor_options: HashMap<String, serde_json::Value>,
}

/// Flexible audio input sources
#[derive(Debug, Clone)]
pub enum AudioSource {
    /// Raw audio bytes with format info
    Bytes {
        data: Vec<u8>,
        format: AudioFormat,
        /// Optional filename for provider context
        filename: Option<String>,
    },
    /// URL to audio file
    Url(String),
    /// File path
    File(PathBuf),
    /// For providers that support recording references
    RecordingId(String),
    /// Base64 encoded audio data
    Base64 {
        data: String,
        format: AudioFormat,
    },
}

impl Default for AudioSource {
    fn default() -> Self {
        Self::Bytes {
            data: Vec::new(),
            format: AudioFormat::Unknown,
            filename: None,
        }
    }
}

impl AudioSource {
    /// Create from raw bytes
    pub fn from_bytes(data: Vec<u8>, format: AudioFormat) -> Self {
        Self::Bytes { data, format, filename: None }
    }

    /// Create from bytes with filename
    pub fn from_bytes_with_name(data: Vec<u8>, format: AudioFormat, filename: String) -> Self {
        Self::Bytes { 
            data, 
            format, 
            filename: Some(filename) 
        }
    }

    /// Create from file path
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        Self::File(path.into())
    }

    /// Create from URL
    pub fn from_url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    /// Create from base64 encoded data
    pub fn from_base64(data: impl Into<String>, format: AudioFormat) -> Self {
        Self::Base64 { 
            data: data.into(), 
            format 
        }
    }

    /// Get the estimated format of the audio
    pub fn format(&self) -> Option<AudioFormat> {
        match self {
            AudioSource::Bytes { format, .. } => Some(*format),
            AudioSource::Base64 { format, .. } => Some(*format),
            AudioSource::File(path) => {
                // Try to detect from file extension
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .and_then(|ext| match ext.to_lowercase().as_str() {
                        "mp3" => Some(AudioFormat::Mp3),
                        "wav" => Some(AudioFormat::Wav),
                        "flac" => Some(AudioFormat::Flac),
                        "ogg" => Some(AudioFormat::Ogg),
                        "webm" => Some(AudioFormat::WebM),
                        "m4a" => Some(AudioFormat::M4a),
                        _ => Some(AudioFormat::Unknown),
                    })
            }
            AudioSource::Url(_) | AudioSource::RecordingId(_) => None,
        }
    }
}

/// Audio format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Ogg,
    WebM,
    M4a,
    Aac,
    /// Unknown format - let provider detect
    Unknown,
}

impl AudioFormat {
    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Flac => "audio/flac", 
            AudioFormat::Ogg => "audio/ogg",
            AudioFormat::WebM => "audio/webm",
            AudioFormat::M4a => "audio/mp4",
            AudioFormat::Aac => "audio/aac",
            AudioFormat::Unknown => "application/octet-stream",
        }
    }

    /// Get common file extensions for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            AudioFormat::Mp3 => &["mp3"],
            AudioFormat::Wav => &["wav"],
            AudioFormat::Flac => &["flac"],
            AudioFormat::Ogg => &["ogg"],
            AudioFormat::WebM => &["webm"],
            AudioFormat::M4a => &["m4a", "mp4"],
            AudioFormat::Aac => &["aac"],
            AudioFormat::Unknown => &[],
        }
    }
}

/// Output format options
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// Just the transcribed text
    #[default]
    Simple,
    /// Include confidence scores and alternatives
    Detailed,
    /// Everything including word timings and metadata
    Verbose,
}

/// Timestamp granularity options
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampGranularity {
    /// No timestamps
    #[default]
    None,
    /// Segment-level timestamps
    Segment,
    /// Word-level timestamps
    Word,
}

// Note: Use TranscriptionRequest::builder() directly for creating instances
// Example: TranscriptionRequest::builder().audio(audio_source).build()