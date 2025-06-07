use crate::content::{Blob, Content, FunctionCall, FunctionResponse};
use serde::{Deserialize, Serialize};

// Client-to-Server Messages

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum ClientMessage {
    ClientContent(ClientContentPayload),
    RealtimeInput(RealtimeInputPayload),
    ToolResponse(ToolResponsePayload),
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ClientContentPayload {
    pub turns: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_complete: Option<bool>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RealtimeInputPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_chunks: Option<Vec<Blob>>,
}


#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ToolResponsePayload {
    pub function_responses: Vec<FunctionResponse>,
}

// Server-to-Client Messages

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum LiveApiResponseChunk {
    SetupComplete {
        #[serde(rename = "setupComplete")]
        _setup: serde_json::Value,
    },
    ModelTurn {
        #[serde(rename = "serverContent")]
        server_content: ModelTurnContent,
    },
    TurnComplete {
        #[serde(rename = "serverContent")]
        server_content: TurnCompleteContent,
    },
    Interrupted {
        #[serde(rename = "serverContent")]
        server_content: InterruptedContent,
    },
    GenerationComplete {
        #[serde(rename = "serverContent")]
        server_content: GenerationCompleteContent,
    },
    ToolCall {
        #[serde(rename = "toolCall")]
        tool_call: ToolCallPayload,
    },
    ToolCallCancellation {
        #[serde(rename = "toolCallCancellation")]
        tool_call_cancellation: ToolCallCancellationPayload,
    },
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct ModelTurnContent {
    #[serde(rename = "modelTurn")]
    pub model_turn: ModelTurnPayload,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct TurnCompleteContent {
    #[serde(rename = "turnComplete")]
    pub turn_complete: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct InterruptedContent {
    #[serde(rename = "interrupted")]
    pub interrupted: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct GenerationCompleteContent {
    #[serde(rename = "generationComplete")]
    pub generation_complete: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct SetupCompletePayload {}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ServerContentPayload {
    #[serde(rename = "modelTurn")]
    pub model_turn: Option<ModelTurnPayload>,
    #[serde(rename = "turnComplete")]
    pub turn_complete: Option<bool>,
    #[serde(rename = "interrupted")]
    pub interrupted: Option<bool>,
    #[serde(rename = "generationComplete")]
    pub generation_complete: Option<bool>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelTurnPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<ModelTurnPart>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelTurnPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<Blob>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executable_code: Option<ExecutableCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution_result: Option<CodeExecutionResult>,
}


#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableCode {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecutionResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallPayload {
    pub function_calls: Vec<FunctionCall>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallCancellationPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<String>>,
}
