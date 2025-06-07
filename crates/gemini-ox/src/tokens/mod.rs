//! Async authentication tokens client for the Gemini API.
//!
//! This module provides functionality to create authentication tokens that can be used
//! for bidirectional streaming connections with the Gemini API. The tokens are created
//! with configurable constraints and parameters.
//!
//! # Examples
//!
//! ## Basic token creation
//!
//! ```no_run
//! use gemini_ox::{Gemini, tokens::LiveEphemeralParametersOptions};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::new("your-api-key");
//!
//! let token_response = gemini
//!     .create_auth_token()
//!     .uses(10)
//!     .expire_time("2024-12-31T23:59:59Z".to_string())
//!     .build()
//!     .send()
//!     .await?;
//!
//! println!("Created token: {}", token_response.token);
//! # Ok(())
//! # }
//! ```
//!
//! ## Token with live parameters
//!
//! ```no_run
//! use gemini_ox::{Gemini, tokens::{LiveEphemeralParametersOptions, LiveConnectConfigOptions}};
//! use std::collections::HashMap;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let gemini = Gemini::new("your-api-key");
//!
//! let live_params = LiveEphemeralParametersOptions::builder()
//!     .model("gemini-1.5-flash".to_string())
//!     .config(
//!         LiveConnectConfigOptions::builder()
//!             .system_instruction("You are a helpful assistant".to_string())
//!             .output_audio_transcription(true)
//!             .additional_config(HashMap::new())
//!             .build()
//!     )
//!     .build();
//!
//! let token_response = gemini
//!     .create_auth_token()
//!     .live_constrained_parameters(live_params)
//!     .lock_additional_fields(vec!["model".to_string()])
//!     .build()
//!     .send()
//!     .await?;
//!
//! println!("Created token: {}", token_response.token);
//! # Ok(())
//! # }
//! ```

pub mod request;
pub mod response;

pub use request::{
    CreateAuthTokenOperation, CreateAuthTokenOperationBuilder, LiveConnectConfigOptions,
    LiveConnectConfigOptionsBuilder, LiveEphemeralParametersOptions,
    LiveEphemeralParametersOptionsBuilder,
};
pub use response::AuthTokenResponse;
