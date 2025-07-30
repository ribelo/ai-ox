use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Response from audio transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// Primary transcription text
    pub text: String,
    
    /// Detected or specified language (ISO-639-1 code)
    pub language: Option<String>,
    
    /// Audio duration in seconds
    pub duration: Option<Duration>,
    
    /// Overall confidence score (0.0 - 1.0)
    pub confidence: Option<f32>,
    
    /// Alternative transcriptions with confidence scores
    #[serde(default)]
    pub alternatives: Vec<Alternative>,
    
    /// Time-aligned segments of the transcription
    #[serde(default)]
    pub segments: Vec<Segment>,
    
    /// Word-level details with timestamps
    #[serde(default)]
    pub words: Vec<Word>,
    
    /// Provider name for reference
    pub provider: String,
    
    /// Model used for transcription
    pub model: String,
    
    /// Usage and performance statistics
    pub usage: SttUsage,
    
    /// Additional provider-specific metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl TranscriptionResponse {
    /// Create a simple response with just text
    pub fn simple(text: String, provider: String, model: String) -> Self {
        Self {
            text,
            language: None,
            duration: None,
            confidence: None,
            alternatives: Vec::new(),
            segments: Vec::new(),
            words: Vec::new(),
            provider,
            model,
            usage: SttUsage::default(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Check if response has segment-level timestamps
    pub fn has_segments(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Check if response has word-level timestamps
    pub fn has_words(&self) -> bool {
        !self.words.is_empty()
    }

    /// Get the total number of words transcribed
    pub fn word_count(&self) -> usize {
        if self.has_words() {
            self.words.len()
        } else {
            // Rough estimate from text
            self.text.split_whitespace().count()
        }
    }

    /// Get the average confidence across all words (if available)
    pub fn average_word_confidence(&self) -> Option<f32> {
        if self.words.is_empty() {
            return self.confidence;
        }

        let confidences: Vec<f32> = self.words
            .iter()
            .filter_map(|w| w.confidence)
            .collect();

        if confidences.is_empty() {
            self.confidence
        } else {
            Some(confidences.iter().sum::<f32>() / confidences.len() as f32)
        }
    }
}

/// Alternative transcription with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    /// Alternative transcription text
    pub text: String,
    /// Confidence score for this alternative (0.0 - 1.0)
    pub confidence: f32,
}

impl Alternative {
    pub fn new(text: String, confidence: f32) -> Self {
        Self { text, confidence }
    }
}

/// Time-aligned segment of transcribed text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Segment text content
    pub text: String,
    /// Start time of the segment
    #[serde(with = "duration_secs")]
    pub start: Duration,
    /// End time of the segment
    #[serde(with = "duration_secs")]
    pub end: Duration,
    /// Confidence score for this segment (0.0 - 1.0)
    pub confidence: Option<f32>,
    /// Segment ID for reference
    pub id: Option<u32>,
}

impl Segment {
    pub fn new(text: String, start: Duration, end: Duration) -> Self {
        Self {
            text,
            start,
            end,
            confidence: None,
            id: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_id(mut self, id: u32) -> Self {
        self.id = Some(id);
        self
    }

    /// Get the duration of this segment
    pub fn duration(&self) -> Duration {
        self.end.saturating_sub(self.start)
    }
}

/// Individual word with timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    /// The transcribed word
    pub text: String,
    /// Start time of the word
    #[serde(with = "duration_secs")]
    pub start: Duration,
    /// End time of the word
    #[serde(with = "duration_secs")]
    pub end: Duration,
    /// Confidence score for this word (0.0 - 1.0)
    pub confidence: Option<f32>,
}

impl Word {
    pub fn new(text: String, start: Duration, end: Duration) -> Self {
        Self {
            text,
            start,
            end,
            confidence: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Get the duration of this word
    pub fn duration(&self) -> Duration {
        self.end.saturating_sub(self.start)
    }
}

/// Usage statistics and performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SttUsage {
    /// Duration of the processed audio
    #[serde(with = "duration_secs", default)]
    pub audio_duration: Duration,
    /// Time taken to process the audio
    #[serde(with = "duration_secs_option", default)]
    pub processing_time: Option<Duration>,
    /// Estimated cost in USD (if available)
    pub cost_estimate: Option<f64>,
    /// Number of audio segments processed
    #[serde(default)]
    pub segments_processed: u32,
    /// Total characters transcribed
    #[serde(default)]
    pub characters_transcribed: u32,
}

impl SttUsage {
    pub fn new(audio_duration: Duration) -> Self {
        Self {
            audio_duration,
            processing_time: None,
            cost_estimate: None,
            segments_processed: 0,
            characters_transcribed: 0,
        }
    }

    pub fn with_processing_time(mut self, processing_time: Duration) -> Self {
        self.processing_time = Some(processing_time);
        self
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost_estimate = Some(cost);
        self
    }

    /// Calculate processing speed ratio (audio duration / processing time)
    pub fn speed_ratio(&self) -> Option<f32> {
        self.processing_time.map(|pt| {
            if pt.is_zero() {
                f32::INFINITY
            } else {
                self.audio_duration.as_secs_f32() / pt.as_secs_f32()
            }
        })
    }
}

// Helper modules for Duration serialization
pub(crate) mod duration_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs.max(0.0)))
    }
}

mod duration_secs_option {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_some(&d.as_secs_f64()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt_secs = Option::<f64>::deserialize(deserializer)?;
        Ok(opt_secs.map(|secs| Duration::from_secs_f64(secs.max(0.0))))
    }
}