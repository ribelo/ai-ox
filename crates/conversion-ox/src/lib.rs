//! AI Provider Format Conversions
//!
//! This crate provides conversions between different AI provider API formats,
//! enabling interoperability between Anthropic, OpenRouter, Gemini, and other
//! AI service APIs. 
//!
//! Since Rust's orphan rule prevents implementing `From` traits for external types,
//! this crate provides wrapper traits and re-exports to enable conversions.

#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

/// Error types for conversion failures
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    /// Unable to convert content due to format incompatibility
    #[error("Unable to convert content: {0}")]
    ContentConversion(String),
    /// Missing required data for conversion
    #[error("Missing required data: {0}")]
    MissingData(String),
    /// Unsupported conversion operation
    #[error("Unsupported conversion: {0}")]
    UnsupportedConversion(String),
}

/// Re-export types and traits from provider crates
#[cfg(feature = "anthropic-openrouter")]
pub mod anthropic_openrouter {
    // Re-export the original conversion traits from openrouter-ox
    pub use openrouter_ox::conversion::*;
    pub use openrouter_ox::conversion::streaming::*;
}

#[cfg(feature = "anthropic-gemini")]
pub mod anthropic_gemini;