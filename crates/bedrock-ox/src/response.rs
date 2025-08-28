use serde::Deserialize;
use crate::message::Message;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<Message>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamEvent {
    pub delta: StreamDelta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamDelta {
    pub text: String,
}
