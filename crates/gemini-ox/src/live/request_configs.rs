use crate::generate_content::content::Content;
use crate::generate_content::{GenerationConfig, SafetySettings};
use crate::tool::Tool;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveConnectConfig {
    #[serde(rename = "setup")]
    pub setup_args: BidiSetupArgs,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<SpeechConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_modalities: Option<Vec<ResponseModality>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proactivity: Option<Proactivity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window_compression: Option<ContextWindowCompression>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum ResponseModality {
    Audio,
    Text,
    // Add other modalities as needed
}

#[derive(Debug, Default, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BidiSetupArgs {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<SafetySettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub realtime_input_config: Option<RealtimeInputConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SpeechConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_config: Option<VoiceConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prebuilt_voice_config: Option<PrebuiltVoiceConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PrebuiltVoiceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RealtimeInputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_chunks: Option<Vec<MediaChunk>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub automatic_activity_detection: Option<AutomaticActivityDetection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activity_handling: Option<ActivityHandling>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_coverage: Option<TurnCoverage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutomaticActivityDetection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_of_speech_sensitivity: Option<SpeechSensitivity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_of_speech_sensitivity: Option<SpeechSensitivity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SpeechSensitivity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActivityHandling {
    #[serde(rename = "ACTIVITY_HANDLING_UNSPECIFIED")]
    Unspecified,
    SendText,
    SendWhenComplete,
    SendAlways,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TurnCoverage {
    #[serde(rename = "TURN_COVERAGE_UNSPECIFIED")]
    Unspecified,
    EntireTurn,
    PerActivity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>, // Base64 encoded data
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Proactivity {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proactive_audio: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextWindowCompression {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<SlidingWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_tokens: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlidingWindow {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_tokens: Option<i64>,
}
