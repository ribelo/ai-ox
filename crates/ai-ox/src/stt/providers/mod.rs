#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "mistral")]
pub mod mistral;

// TODO: Implement Gemini STT provider (requires ALSA dependencies)
// #[cfg(feature = "gemini")]
// pub mod gemini;