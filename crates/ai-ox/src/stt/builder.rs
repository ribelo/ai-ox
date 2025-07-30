use std::sync::Arc;

use super::{SttError, SpeechToText};

/// Convenience function to create a Groq STT provider
#[cfg(feature = "groq")]
pub fn groq_stt(model: &str) -> Result<Arc<dyn SpeechToText>, SttError> {
    super::providers::groq::GroqStt::builder()
        .model(model)
        .api_key_from_env("GROQ_API_KEY")?
        .build()
}

/// Convenience function to create a Groq STT provider with custom API key
#[cfg(feature = "groq")]
pub fn groq_stt_with_key(model: &str, api_key: &str) -> Result<Arc<dyn SpeechToText>, SttError> {
    super::providers::groq::GroqStt::builder()
        .model(model)
        .api_key(api_key)
        .build()
}

/// Convenience function to create a Mistral STT provider
#[cfg(feature = "mistral")]
pub fn mistral_stt(model: &str) -> Result<Arc<dyn SpeechToText>, SttError> {
    super::providers::mistral::MistralStt::builder()
        .model(model)
        .api_key_from_env("MISTRAL_API_KEY")?
        .build()
}

/// Convenience function to create a Mistral STT provider with custom API key
#[cfg(feature = "mistral")]
pub fn mistral_stt_with_key(model: &str, api_key: &str) -> Result<Arc<dyn SpeechToText>, SttError> {
    super::providers::mistral::MistralStt::builder()
        .model(model)
        .api_key(api_key)
        .build()
}

/// Convenience function to create a Gemini STT provider
#[cfg(feature = "gemini")]
pub fn gemini_stt(model: &str) -> Result<Arc<dyn SpeechToText>, SttError> {
    super::providers::gemini::GeminiStt::builder()
        .model(model)
        .api_key_from_env("GEMINI_API_KEY")?
        .build()
}

/// Convenience function to create a Gemini STT provider with custom API key
#[cfg(feature = "gemini")]
pub fn gemini_stt_with_key(model: &str, api_key: &str) -> Result<Arc<dyn SpeechToText>, SttError> {
    super::providers::gemini::GeminiStt::builder()
        .model(model)
        .api_key(api_key)
        .build()
}

/// Auto-detect and create the best available STT provider
/// Auto-detection preference order:
/// 1. Groq (fastest, good quality)
/// 2. Mistral (good balance)  
/// 3. Gemini (feature-rich)
pub fn auto_stt() -> Result<Arc<dyn SpeechToText>, SttError> {
    #[cfg(feature = "groq")]
    if std::env::var("GROQ_API_KEY").is_ok() {
        return groq_stt("whisper-large-v3");
    }

    #[cfg(feature = "mistral")]
    if std::env::var("MISTRAL_API_KEY").is_ok() {
        return mistral_stt("voxtral-large-24-05");
    }

    #[cfg(feature = "gemini")]
    if std::env::var("GEMINI_API_KEY").is_ok() {
        return gemini_stt("gemini-1.5-flash");
    }

    Err(SttError::MissingApiKey)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_builder_convenience() {
        // Test that convenience functions work
        #[cfg(feature = "groq")]
        {
            let result = groq_stt_with_key("whisper-large-v3", "test-key");
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_auto_detection_returns_error_when_no_keys() {
        // When no API keys are available, should return MissingApiKey
        let original_groq = std::env::var("GROQ_API_KEY");
        let original_mistral = std::env::var("MISTRAL_API_KEY");
        let original_gemini = std::env::var("GEMINI_API_KEY");

        // Remove all keys temporarily
        std::env::remove_var("GROQ_API_KEY");
        std::env::remove_var("MISTRAL_API_KEY");
        std::env::remove_var("GEMINI_API_KEY");

        let result = auto_stt();
        assert!(matches!(result, Err(SttError::MissingApiKey)));

        // Restore original values
        if let Ok(key) = original_groq {
            std::env::set_var("GROQ_API_KEY", key);
        }
        if let Ok(key) = original_mistral {
            std::env::set_var("MISTRAL_API_KEY", key);
        }
        if let Ok(key) = original_gemini {
            std::env::set_var("GEMINI_API_KEY", key);
        }
    }
}