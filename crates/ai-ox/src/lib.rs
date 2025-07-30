pub mod agent;
pub mod content;
pub mod errors;
pub mod model;
#[cfg(any(feature = "groq", feature = "mistral", feature = "gemini"))]
pub mod stt;
pub mod tool;
pub mod usage;
pub mod workflow;

// Re-export the toolbox macro
pub use ai_ox_macros::toolbox;

// Re-export commonly used types
pub use errors::GenerateContentError;
pub use model::response::{ModelResponse, StructuredResponse};

// Re-export model implementations based on features
#[cfg(feature = "gemini")]
pub use model::gemini::GeminiModel;

#[cfg(feature = "openrouter")]
pub use model::openrouter::OpenRouterModel;
