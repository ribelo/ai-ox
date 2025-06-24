use mime::Mime;

/// Standard image MIME types
pub const IMAGE_JPEG: Mime = mime::IMAGE_JPEG;
pub const IMAGE_PNG: Mime = mime::IMAGE_PNG;

/// Custom audio PCM MIME types for live API
/// Note: These are non-standard MIME types specific to the Gemini API
pub const AUDIO_PCM_16KHZ: &str = "audio/pcm;rate=16000";
pub const AUDIO_PCM_24KHZ: &str = "audio/pcm;rate=24000";

/// Helper function to create audio PCM MIME type string with custom sample rate
pub fn audio_pcm_with_rate(sample_rate: u32) -> String {
    format!("audio/pcm;rate={sample_rate}")
}
