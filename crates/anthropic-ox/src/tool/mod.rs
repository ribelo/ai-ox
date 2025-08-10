use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fmt, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[cfg(feature = "schema")]
    pub input_schema: serde_json::Value,
}

impl Tool {
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            #[cfg(feature = "schema")]
            input_schema: serde_json::json!({}),
        }
    }

    #[cfg(feature = "schema")]
    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

impl ToolUse {
    pub fn new(id: String, name: String, input: serde_json::Value) -> Self {
        Self { id, name, input }
    }
}

impl fmt::Display for ToolUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolUse(id: {}, name: {})", self.id, self.name)
    }
}

#[derive(Debug, Default, Clone)]
pub struct ToolUseBuilder {
    id: String,
    name: String,
    input: String,
}

impl ToolUseBuilder {
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            input: String::new(),
        }
    }

    pub fn push_str(&mut self, s: &str) {
        self.input.push_str(s);
    }

    pub fn build(self) -> Result<ToolUse, serde_json::Error> {
        let input: serde_json::Value = serde_json::from_str(&self.input)?;
        Ok(ToolUse {
            id: self.id,
            name: self.name,
            input,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: Vec<ToolResultContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: crate::message::ImageSource },
}

impl ToolResult {
    pub fn new(tool_use_id: String, content: Vec<ToolResultContent>) -> Self {
        Self {
            tool_use_id,
            content,
            is_error: None,
        }
    }

    pub fn text(tool_use_id: String, text: String) -> Self {
        Self {
            tool_use_id,
            content: vec![ToolResultContent::Text { text }],
            is_error: None,
        }
    }

    pub fn error(tool_use_id: String, error: String) -> Self {
        Self {
            tool_use_id,
            content: vec![ToolResultContent::Text { text: error }],
            is_error: Some(true),
        }
    }
}

impl fmt::Display for ToolResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToolResult(id: {})", self.tool_use_id)
    }
}


/// Trait for tools that can be called by the AI
#[async_trait]
pub trait ToolTrait: Send + Sync + Clone {
    type Input: for<'de> Deserialize<'de> + Send;
    type Output: Serialize + Send;
    type Error: fmt::Display + Send;

    fn name(&self) -> String;
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
}

/// A collection of tools that can be invoked
#[derive(Clone, Default)]
pub struct ToolBox {
    tools: HashMap<String, Arc<dyn ToolInvoker>>,
}

impl std::fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolBox")
            .field("tools", &format!("{} tools", self.tools.len()))
            .finish()
    }
}

impl ToolBox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add<T: ToolTrait + 'static>(&self, _tool: T) {
        // This is a simplified version - in reality you'd need to store the tools
        // and provide a way to invoke them
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ToolInvoker>> {
        self.tools.get(name)
    }

    pub async fn invoke(&self, tool_use: ToolUse) -> ToolResult {
        // Simplified implementation
        ToolResult::text(tool_use.id, "Tool invocation not implemented".to_string())
    }
}

// Helper trait for type erasure
#[async_trait]
trait ToolInvoker: Send + Sync {
    async fn invoke(&self, input: Value) -> Result<Value, String>;
}

impl Serialize for ToolBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as empty array for now
        let tools: Vec<Tool> = vec![];
        tools.serialize(serializer)
    }
}