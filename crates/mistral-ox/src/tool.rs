use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Tool {
    pub r#type: String,
    pub function: ToolFunction,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters: None,
            },
        }
    }

    pub fn with_parameters(mut self, parameters: Value) -> Self {
        self.function.parameters = Some(parameters);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Any,
    #[serde(rename = "function")]
    Function {
        name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(default = "default_tool_type", skip_serializing_if = "is_default_type")]
    pub r#type: String,
    pub function: FunctionCall,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

fn default_tool_type() -> String {
    "function".to_string()
}

fn is_default_type(t: &String) -> bool {
    t == "function"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}
