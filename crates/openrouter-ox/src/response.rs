use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    message::{AssistantMessage, Content, ContentPart, Message},
    ApiRequestError, ErrorResponse,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    #[serde(alias = "STOP")]
    Stop,
    Limit,
    ContentFilter,
    ToolCalls,
    #[serde(alias = "length", alias = "LENGTH")]
    Length,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub system_fingerprint: Option<String>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        let calls: Vec<ToolCall> = self
            .choices
            .iter()
            .flat_map(|choice| choice.message.tool_calls.clone().unwrap_or_default())
            .collect();
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }
}

impl From<ChatCompletionResponse> for Message {
    fn from(resp: ChatCompletionResponse) -> Self {
        // Convert ChatCompletionResponse into a Message by selecting the first choice's message.
        // If no choices are provided, return an AssistantMessage with empty content.
        if let Some(choice) = resp.choices.into_iter().next() {
            Message::Assistant(choice.message)
        } else {
            Message::Assistant(AssistantMessage {
                content: Content(Vec::new()),
                tool_calls: None,
                name: None,
                refusal: None,
            })
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Choice {
    pub index: usize,
    pub message: AssistantMessage,
    pub logprobs: Option<Value>,
    pub finish_reason: FinishReason,
    pub native_finish_reason: Option<FinishReason>,
}

impl<'de> Deserialize<'de> for Choice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Debug, Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Index,
            Message,
            Logprobs,
            FinishReason,
            NativeFinishReason,
        }

        struct ChoiceVisitor;

        impl<'de> serde::de::Visitor<'de> for ChoiceVisitor {
            type Value = Choice;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Choice")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Choice, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut index = None;
                let mut response_message: Option<ResponseMessage> = None;
                let mut logprobs = None;
                let mut finish_reason = None;
                let mut native_finish_reason = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Index => {
                            index = Some(map.next_value()?);
                        }
                        Field::Message => {
                            response_message = Some(map.next_value()?);
                        }
                        Field::Logprobs => {
                            logprobs = Some(map.next_value()?);
                        }
                        Field::FinishReason => {
                            finish_reason = Some(map.next_value()?);
                        }
                        Field::NativeFinishReason => {
                            native_finish_reason = Some(map.next_value()?);
                        }
                    }
                }
                let index = index.ok_or_else(|| serde::de::Error::missing_field("index"))?;
                let message = response_message
                    .ok_or_else(|| serde::de::Error::missing_field("message"))?
                    .into();
                let finish_reason =
                    finish_reason.ok_or_else(|| serde::de::Error::missing_field("finishReason"))?;
                Ok(Choice {
                    index,
                    message,
                    logprobs,
                    finish_reason,
                    native_finish_reason,
                })
            }
        }

        const FIELDS: &[&str] = &["index", "message", "logprobs", "finishReason"];
        deserializer.deserialize_struct("Choice", FIELDS, ChoiceVisitor)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolCall {
    #[serde(skip_serializing)]
    pub index: Option<usize>,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FunctionCall {
    pub name: Option<String>,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PromptTokensDetails {
    pub cached_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: i32,
    pub accepted_prediction_tokens: i32,
    pub rejected_prediction_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub refusal: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl From<ResponseMessage> for AssistantMessage {
    fn from(resp: ResponseMessage) -> Self {
        AssistantMessage {
            content: Content(match resp.content {
                Some(text) if !text.is_empty() => vec![ContentPart::Text(text.into())],
                _ => vec![],
            }),
            tool_calls: resp.tool_calls,
            name: None,
            refusal: resp.refusal,
        }
    }
}

impl From<ResponseMessage> for Message {
    fn from(resp: ResponseMessage) -> Self {
        Message::Assistant(AssistantMessage::from(resp))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChatCompletionChunk {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub object: String,
    pub created: i64,
    pub choices: Vec<ChunkChoice>,
    pub usage: Option<Usage>,
}

impl ChatCompletionChunk {
    pub fn from_streaming_data(
        lines_str: &str,
    ) -> Vec<Result<ChatCompletionChunk, ApiRequestError>> {
        let mut results = Vec::new();
        for line in lines_str.lines() {
            // The original debug prints are removed for cleaner code.
            // If needed, they could be added back, potentially using a logging framework.
            // println!("lines {}", line.lines().count()); // This would always be 1 here.
            // println!("Processing line: {}", line);
            // println!();
            // println!();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue; // Skip empty lines
            }

            // Some providers might send error JSON directly without the 'data:' prefix
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(trimmed) {
                results.push(Err(ApiRequestError::InvalidRequestError(err)));
                continue;
            }

            if !trimmed.starts_with("data:") {
                // Ignore lines not starting with 'data:' unless it was parsed as an error above
                continue;
            }

            let data = match trimmed.strip_prefix("data:") {
                Some(d) => d.trim(),
                // This case should technically be unreachable due to the starts_with check,
                // but we handle it defensively by skipping the line.
                None => continue,
            };

            if data == "[DONE]" {
                // The [DONE] marker signifies the end of the stream.
                // It doesn't contain chunk data, so we skip it.
                continue;
            }

            // Attempt to parse the data payload as a ChatCompletionChunk
            match serde_json::from_str::<ChatCompletionChunk>(data) {
                Ok(chunk) => results.push(Ok(chunk)),
                Err(_e) => {
                    // Use _e to indicate the variable is intentionally unused
                    // If parsing as a chunk fails, try parsing as an ErrorResponse,
                    // as some APIs might send errors within the 'data:' payload.
                    match serde_json::from_str::<ErrorResponse>(data) {
                        Ok(error_response) => {
                            results.push(Err(ApiRequestError::InvalidRequestError(error_response)));
                        }
                        Err(_) => {
                            // If it fails to parse as both ChatCompletionChunk and ErrorResponse,
                            // the line is considered unparseable in the expected formats.
                            // We skip it, mirroring the behavior of the original function.
                            // Consider adding logging here if unparseable lines need tracking.
                            // eprintln!("Failed to parse stream data line: {}", data);
                            // eprintln!("Chunk Error: {:?}, ErrorResponse Error: {:?}", _e, _);
                        }
                    }
                }
            }
        }
        results
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}
