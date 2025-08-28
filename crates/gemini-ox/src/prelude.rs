//! Crate prelude

pub use crate::{
    request::{
        GenerateContentRequest,
        EmbedContentRequest,
        GenerationConfig,
        SafetySetting,
        HarmCategory,
        HarmBlockThreshold,
    },
    content::{
        Content,
        Part,
        Text,
    },
    tool::{
        Tool,
        FunctionMetadata,
    },
    Gemini,
    GeminiRequestError,
};
