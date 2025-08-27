use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::{collections::HashMap, fmt, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

/// Represents a tool that can be used by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Tool {
    /// A custom tool defined by the user.
    Custom(CustomTool),
    /// The built-in computer use tool.
    Computer(ComputerTool),
}

/// A custom tool defined by the user.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomTool {
    /// The type of the tool, which is always "custom".
    #[serde(rename = "type", default = "default_tool_type")]
    pub object_type: String,
    /// The name of the tool.
    pub name: String,
    /// A description of the tool.
    pub description: String,
    /// The input schema for the tool.
    pub input_schema: serde_json::Value,
}

fn default_tool_type() -> String {
    "custom".to_string()
}

impl CustomTool {
    pub fn new(name: String, description: String) -> Self {
        Self {
            object_type: "custom".to_string(),
            name,
            description,
            input_schema: serde_json::json!({}),
        }
    }

    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = schema;
        self
    }
}

/// The built-in computer use tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComputerTool {
    /// The type of the tool, e.g., "computer_20250124".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The name of the tool, which is always "computer".
    pub name: String,
    /// The width of the display in pixels.
    pub display_width_px: u32,
    /// The height of the display in pixels.
    pub display_height_px: u32,
    /// The display number for X11 environments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_number: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::message::CacheControl>,
}

impl ToolUse {
    pub fn new(id: String, name: String, input: serde_json::Value) -> Self {
        Self { id, name, input, cache_control: None }
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
            cache_control: None,
        })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: Vec<ToolResultContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::message::CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    Text { text: String },
    Image { source: crate::message::ImageSource },
}


impl<'de> Deserialize<'de> for ToolResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{Error, MapAccess, Visitor};
        use std::fmt;

        struct ToolResultVisitor;

        impl<'de> Visitor<'de> for ToolResultVisitor {
            type Value = ToolResult;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a ToolResult struct")
            }

            fn visit_map<V>(self, mut map: V) -> Result<ToolResult, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut tool_use_id = None;
                let mut content = None;
                let mut is_error = None;
                let mut cache_control = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "tool_use_id" => {
                            tool_use_id = Some(map.next_value()?);
                        }
                        "content" => {
                            content = Some(deserialize_tool_result_content_value(map.next_value()?)?);
                        }
                        "is_error" => {
                            is_error = Some(map.next_value()?);
                        }
                        "cache_control" => {
                            cache_control = Some(map.next_value()?);
                        }
                        _ => {
                            let _: serde_json::Value = map.next_value()?;
                        }
                    }
                }

                let tool_use_id = tool_use_id.ok_or_else(|| Error::missing_field("tool_use_id"))?;
                let content = content.ok_or_else(|| Error::missing_field("content"))?;

                Ok(ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    cache_control,
                })
            }
        }

        deserializer.deserialize_struct("ToolResult", &["tool_use_id", "content", "is_error", "cache_control"], ToolResultVisitor)
    }
}

fn deserialize_tool_result_content_value<E>(value: serde_json::Value) -> Result<Vec<ToolResultContent>, E>
where
    E: serde::de::Error,
{
    match value {
        // Handle string format (Claude Code format)
        serde_json::Value::String(text) => {
            Ok(vec![ToolResultContent::Text { text }])
        }
        // Handle array format (standard Anthropic format)
        serde_json::Value::Array(arr) => {
            arr.into_iter()
                .map(|item| {
                    serde_json::from_value(item).map_err(E::custom)
                })
                .collect()
        }
        _ => Err(E::custom("content must be either a string or an array")),
    }
}

impl ToolResult {
    pub fn new(tool_use_id: String, content: Vec<ToolResultContent>) -> Self {
        Self {
            tool_use_id,
            content,
            is_error: None,
            cache_control: None,
        }
    }

    pub fn text(tool_use_id: String, text: String) -> Self {
        Self {
            tool_use_id,
            content: vec![ToolResultContent::Text { text }],
            is_error: None,
            cache_control: None,
        }
    }

    pub fn error(tool_use_id: String, error: String) -> Self {
        Self {
            tool_use_id,
            content: vec![ToolResultContent::Text { text: error }],
            is_error: Some(true),
            cache_control: None,
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