use bon::Builder;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::response::{Segment, Word};

/// Request for streaming audio transcription
#[derive(Debug, Clone, Builder)]
pub struct StreamingTranscriptionRequest {
    /// Audio stream configuration
    pub audio_config: AudioStreamConfig,

    /// Target language (ISO-639-1) - None for auto-detect
    #[builder(into)]
    pub language: Option<String>,

    /// Prompt to guide transcription style
    #[builder(into)]
    pub prompt: Option<String>,

    /// Temperature for transcription creativity (0.0 - 1.0)
    #[builder(default = 0.0)]
    pub temperature: f32,

    /// How often to emit interim results
    #[builder(default = Duration::from_millis(500))]
    pub interim_results_interval: Duration,

    /// Whether to enable speaker diarization (if supported)
    #[builder(default = false)]
    pub enable_speaker_diarization: bool,

    /// Maximum number of speakers to detect
    #[builder(default = 2)]
    pub max_speakers: u8,

    /// Provider-specific options
    #[builder(default)]
    pub vendor_options: std::collections::HashMap<String, serde_json::Value>,
}

/// Audio stream configuration
#[derive(Debug, Clone)]
pub struct AudioStreamConfig {
    /// Audio format/encoding
    pub format: super::AudioFormat,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u8,
    /// Audio encoding details
    pub encoding: AudioEncoding,
    /// Chunk size in bytes (for buffering)
    pub chunk_size: Option<usize>,
}

impl AudioStreamConfig {
    /// Create a common configuration for 16kHz mono WAV
    pub fn wav_16khz_mono() -> Self {
        Self {
            format: super::AudioFormat::Wav,
            sample_rate: 16000,
            channels: 1,
            encoding: AudioEncoding::Linear16,
            chunk_size: Some(4096),
        }
    }

    /// Create a common configuration for 44.1kHz stereo WAV
    pub fn wav_44khz_stereo() -> Self {
        Self {
            format: super::AudioFormat::Wav,
            sample_rate: 44100,
            channels: 2,
            encoding: AudioEncoding::Linear16,
            chunk_size: Some(8192),
        }
    }

    /// Create configuration for MP3 stream
    pub fn mp3(sample_rate: u32, channels: u8) -> Self {
        Self {
            format: super::AudioFormat::Mp3,
            sample_rate,
            channels,
            encoding: AudioEncoding::Mp3,
            chunk_size: Some(4096),
        }
    }
}

impl Default for AudioStreamConfig {
    fn default() -> Self {
        Self::wav_16khz_mono()
    }
}

/// Audio encoding specifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AudioEncoding {
    /// Uncompressed 16-bit signed little-endian samples
    Linear16,
    /// Uncompressed 32-bit float samples
    Float32,
    /// MP3 compressed audio
    Mp3,
    /// FLAC compressed audio
    Flac,
    /// Opus compressed audio
    Opus,
    /// μ-law encoded audio
    Mulaw,
    /// A-law encoded audio
    Alaw,
}

/// Events emitted during streaming transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TranscriptionEvent {
    /// Partial/interim transcription result
    Interim {
        /// Partial transcription text
        text: String,
        /// Stability score (0.0 - 1.0) - how likely this text is to change
        stability: Option<f32>,
        /// Confidence score for the interim result
        confidence: Option<f32>,
        /// Segment ID this interim result belongs to
        segment_id: u32,
        /// Speaker ID (if diarization is enabled)
        speaker_id: Option<u8>,
    },

    /// Final transcription result for a segment
    Final {
        /// Complete segment with timing information
        segment: Segment,
        /// Segment ID for reference
        segment_id: u32,
        /// Speaker ID (if diarization is enabled)  
        speaker_id: Option<u8>,
        /// Word-level details if available
        words: Vec<Word>,
    },

    /// Speech activity detection
    SpeechStart {
        /// Timestamp when speech started
        #[serde(with = "super::response::duration_secs")]
        timestamp: Duration,
    },

    /// End of speech segment detected
    SpeechEnd {
        /// Timestamp when speech ended
        #[serde(with = "super::response::duration_secs")]
        timestamp: Duration,
    },

    /// End of entire audio stream
    EndOfStream,

    /// Non-fatal warning during processing
    Warning {
        /// Warning message
        message: String,
        /// Warning code (provider-specific)
        code: Option<String>,
    },

    /// Error during streaming transcription
    Error {
        /// Error message
        message: String,
        /// Error code (provider-specific)
        code: Option<String>,
        /// Whether the error is recoverable
        recoverable: bool,
    },
}

impl TranscriptionEvent {
    /// Create an interim result event
    pub fn interim(text: String, segment_id: u32) -> Self {
        Self::Interim {
            text,
            stability: None,
            confidence: None,
            segment_id,
            speaker_id: None,
        }
    }

    /// Create an interim result with stability score
    pub fn interim_with_stability(text: String, segment_id: u32, stability: f32) -> Self {
        Self::Interim {
            text,
            stability: Some(stability),
            confidence: None,
            segment_id,
            speaker_id: None,
        }
    }

    /// Create a final result event
    pub fn final_result(segment: Segment, segment_id: u32) -> Self {
        Self::Final {
            segment,
            segment_id,
            speaker_id: None,
            words: Vec::new(),
        }
    }

    /// Create a final result with words
    pub fn final_with_words(segment: Segment, segment_id: u32, words: Vec<Word>) -> Self {
        Self::Final {
            segment,
            segment_id,
            speaker_id: None,
            words,
        }
    }

    /// Create a speech start event
    pub fn speech_start(timestamp: Duration) -> Self {
        Self::SpeechStart { timestamp }
    }

    /// Create a speech end event
    pub fn speech_end(timestamp: Duration) -> Self {
        Self::SpeechEnd { timestamp }
    }

    /// Create an end of stream event
    pub fn end_of_stream() -> Self {
        Self::EndOfStream
    }

    /// Create a warning event
    pub fn warning(message: String) -> Self {
        Self::Warning {
            message,
            code: None,
        }
    }

    /// Create an error event
    pub fn error(message: String, recoverable: bool) -> Self {
        Self::Error {
            message,
            code: None,
            recoverable,
        }
    }

    /// Check if this event is an error
    pub fn is_error(&self) -> bool {
        matches!(self, TranscriptionEvent::Error { .. })
    }

    /// Check if this event is final
    pub fn is_final(&self) -> bool {
        matches!(self, TranscriptionEvent::Final { .. })
    }

    /// Check if this event is interim
    pub fn is_interim(&self) -> bool {
        matches!(self, TranscriptionEvent::Interim { .. })
    }

    /// Extract text content if available
    pub fn text(&self) -> Option<&str> {
        match self {
            TranscriptionEvent::Interim { text, .. } => Some(text),
            TranscriptionEvent::Final { segment, .. } => Some(&segment.text),
            _ => None,
        }
    }

    /// Get segment ID if applicable
    pub fn segment_id(&self) -> Option<u32> {
        match self {
            TranscriptionEvent::Interim { segment_id, .. } => Some(*segment_id),
            TranscriptionEvent::Final { segment_id, .. } => Some(*segment_id),
            _ => None,
        }
    }
}

// Note: Use StreamingTranscriptionRequest::builder() directly for creating instances
// Example: StreamingTranscriptionRequest::builder().audio_config(config).build()
