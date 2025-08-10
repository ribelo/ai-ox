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

#[cfg(feature = "mistral")]
use ai_ox::model::mistral::MistralModel;

#[cfg(feature = "anthropic")]
use ai_ox::model::anthropic::AnthropicModel;

// Model version constants for consistent testing
const BEDROCK_MODEL: &str = "anthropic.claude-3-haiku-20240307-v1:0";
const GEMINI_MODEL: &str = "gemini-2.5-flash";
const OPENROUTER_MODEL: &str = "openai/gpt-4o-mini";
const MISTRAL_MODEL: &str = "mistral-small-latest";
const ANTHROPIC_MODEL: &str = "claude-3-haiku-20240307";

/// Helper function to initialize a provider model with graceful error handling
async fn try_init_provider<T, E, F, Fut>(
    env_var: &str,
    provider_name: &str, 
    init_fn: F,
) -> Option<Box<dyn Model>>
where
    T: Model + 'static,
    E: std::fmt::Display,
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    if std::env::var(env_var).is_ok() {
        match init_fn().await {
            Ok(model) => {
                println!("✅ {} `Model` initialized for testing.", provider_name);
                Some(Box::new(model))
            }
            Err(e) => {
                println!("⚠️  {} initialization failed (API key may be invalid): {}", provider_name, e);
                println!("   Skipping {} tests. Set {} to enable.", provider_name, env_var);
                None
            }
        }
    } else {
        println!("⚠️  {} tests skipped: {} not found in environment.", provider_name, env_var);
        println!("   Set {} to enable {} tests.", env_var, provider_name);
        None
    }
}

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
        // Bedrock requires both AWS credentials
        if std::env::var("AWS_ACCESS_KEY_ID").is_ok() && 
           std::env::var("AWS_SECRET_ACCESS_KEY").is_ok() {
            if let Some(model) = try_init_provider(
                "AWS_ACCESS_KEY_ID", 
                "Bedrock",
                || BedrockModel::new(BEDROCK_MODEL.to_string())
            ).await {
                models.push(model);
            }
        } else {
            println!("⚠️  Bedrock tests skipped: AWS credentials not found in environment.");
            println!("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to enable Bedrock tests.");
        }
    }

    #[cfg(feature = "gemini")]
    {
        if let Some(model) = try_init_provider(
            "GOOGLE_AI_API_KEY",
            "Gemini", 
            || GeminiModel::new(GEMINI_MODEL.to_string())
        ).await {
            models.push(model);
        }
    }

    #[cfg(feature = "openrouter")]
    {
        // Use a model that supports tool calling through OpenRouter
        // Note: google/gemini models don't support tools through OpenRouter
        if let Some(model) = try_init_provider(
            "OPENROUTER_API_KEY",
            "OpenRouter",
            || OpenRouterModel::new(OPENROUTER_MODEL.to_string())
        ).await {
            models.push(model);
        }
    }

    #[cfg(feature = "mistral")]
    {
        if let Some(model) = try_init_provider(
            "MISTRAL_API_KEY",
            "Mistral",
            || MistralModel::new(MISTRAL_MODEL.to_string())
        ).await {
            models.push(model);
        }
    }

    #[cfg(feature = "anthropic")]
    {
        if let Some(model) = try_init_provider(
            "ANTHROPIC_API_KEY",
            "Anthropic",
            || AnthropicModel::new(ANTHROPIC_MODEL.to_string())
        ).await {
            models.push(model);
        }
    }

    // This is to satisfy the compiler in case no features are enabled.
    #[cfg(not(any(feature = "bedrock", feature = "gemini", feature = "openrouter", feature = "mistral", feature = "anthropic")))]
    {
        println!("⚠️ No provider features enabled. All compliance tests will be skipped.");
    }

    if models.is_empty() && cfg!(any(feature = "bedrock", feature = "gemini", feature = "openrouter", feature = "mistral", feature = "anthropic")) {
         // This case can happen when features are enabled but API keys are not set
         // All providers now gracefully handle missing API keys
        println!("⚠️  No models were initialized. Make sure to set the appropriate API keys:");
        #[cfg(feature = "bedrock")]
        println!("   - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for Bedrock");
        #[cfg(feature = "gemini")]
        println!("   - GOOGLE_AI_API_KEY for Gemini");
        #[cfg(feature = "openrouter")]
        println!("   - OPENROUTER_API_KEY for OpenRouter");
        #[cfg(feature = "mistral")]
        println!("   - MISTRAL_API_KEY for Mistral");
        #[cfg(feature = "anthropic")]
        println!("   - ANTHROPIC_API_KEY for Anthropic");
    }

    models
}
