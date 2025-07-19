//! `common` is a four-letter word here. This is NOT a dumping ground.
//! It's a focused module for instantiating concrete `Model` implementations
//! for the purpose of compliance testing.
//!
//! The logic here is intentionally strict. If a provider's feature flag is enabled,
//! this module EXPECTS to be able to initialize it. Failure to do so is a panic.
//! This ensures that anyone testing a specific feature has their environment
//! correctly configured. To test a subset of providers, use Cargo's feature flags:
//! `cargo test --no-default-features --features <provider-name>`

use ai_ox::model::Model;

#[cfg(feature = "bedrock")]
use ai_ox::model::bedrock::BedrockModel;

#[cfg(feature = "gemini")]
use ai_ox::model::gemini::GeminiModel;

#[cfg(feature = "openrouter")]
use ai_ox::model::openrouter::OpenRouterModel;

/// Asynchronously collects all available models for integration testing based on
/// enabled feature flags.
///
/// This function is the single point of contact for test suites that need to
/// run against supported `Model` providers.
///
/// It will panic if a feature is enabled but the corresponding environment
/// variables (e.g., API keys) are not correctly set. This is by design.
#[allow(dead_code)]
pub async fn get_available_models() -> Vec<Box<dyn Model>> {
    let mut models: Vec<Box<dyn Model>> = Vec::new();

    #[cfg(feature = "bedrock")]
    {
        let model = BedrockModel::new("anthropic.claude-3-haiku-20240307-v1:0".to_string())
            .await
            .expect("FATAL: `bedrock` feature is enabled, but BedrockModel failed to initialize. Check your AWS credentials and region configuration.");
        models.push(Box::new(model));
        println!("✅ Bedrock `Model` initialized for testing.");
    }

    #[cfg(feature = "gemini")]
    {
        let model = GeminiModel::new("gemini-1.5-flash-latest".to_string())
            .await
            .expect("FATAL: `gemini` feature is enabled, but GeminiModel failed to initialize. Check your GOOGLE_AI_API_KEY.");
        models.push(Box::new(model));
        println!("✅ Gemini `Model` initialized for testing.");
    }

    #[cfg(feature = "openrouter")]
    {
        let model = OpenRouterModel::new("google/gemini-flash-1.5".to_string())
            .await
            .expect("FATAL: `openrouter` feature is enabled, but OpenRouterModel failed to initialize. Check your OPENROUTER_API_KEY.");
        models.push(Box::new(model));
        println!("✅ OpenRouter `Model` initialized for testing.");
    }

    // This is to satisfy the compiler in case no features are enabled.
    #[cfg(not(any(feature = "bedrock", feature = "gemini", feature = "openrouter")))]
    {
        println!("⚠️ No provider features enabled. All compliance tests will be skipped.");
    }

    if models.is_empty() && cfg!(any(feature = "bedrock", feature = "gemini", feature = "openrouter")) {
         // This case should not be reached if the .expect() calls are working correctly,
         // but it's a good safeguard.
        panic!("A provider feature flag was enabled, but no models were initialized. This indicates a logic error in get_available_models.");
    }

    models
}
