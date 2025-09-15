use ai_ox::tool::ToolUse;
use ai_ox::content::Part;
use mcp_sdk::types::{CallToolRequest, CallToolResponse};
use serde_json::{Value, json};

use crate::{ConversionConfig, FromMcp, McpConversionError, ToMcp};

// ToolUse conversions
impl ToMcp<CallToolRequest> for ToolUse {
    fn to_mcp(&self) -> Result<CallToolRequest, McpConversionError> {
        let mut arguments = self.args.clone();
        if let Some(obj) = arguments.as_object_mut() {
            obj.insert("x_ai_ox_tool_call_id".to_string(), json!(self.id));
        }

        Ok(CallToolRequest {
            name: self.name.clone(),
            arguments: Some(arguments),
            meta: None,
        })
    }
}

impl FromMcp<CallToolRequest> for ToolUse {
    fn from_mcp(value: CallToolRequest) -> Result<Self, McpConversionError> {
        let mut arguments = value.arguments.unwrap_or_else(|| Value::Object(Default::default()));
        let id = if let Some(obj) = arguments.as_object_mut() {
            obj.remove("x_ai_ox_tool_call_id")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default()
        } else {
            String::new()
        };

        Ok(ToolUse {
            id,
            name: value.name,
            args: arguments,
            ext: None,
        })
    }
}

// ToolResult conversions
impl ToMcp<CallToolResponse> for Part {
    fn to_mcp(&self) -> Result<CallToolResponse, McpConversionError> {
        if let Part::ToolResult { id, name, parts, ext } = self {
            // Create a message with the parts
            let message = ai_ox::content::Message::new(ai_ox::content::MessageRole::Assistant, parts.clone());
            let response = vec![message];

            // Convert Vec<Message> to Vec<ToolResponseContent>
            let mcp_content = response.to_mcp()?;

            // Use meta field with namespaced keys
            let meta = Some(json!({
                "ai_ox": {
                    "call_id": id,
                    "name": name
                }
            }));

            // Propagate is_error flag from ext
            let is_error = ext.get("mcp.is_error")
                .and_then(|v| v.as_bool());

            Ok(CallToolResponse {
                content: mcp_content,
                meta,
                is_error,
            })
        } else {
            Err(McpConversionError::InvalidFormat("Expected ToolResult part".to_string()))
        }
    }
}

impl FromMcp<CallToolResponse> for Part {
    fn from_mcp(value: CallToolResponse) -> Result<Self, McpConversionError> {
        Self::from_mcp_with_config(value, &ConversionConfig::default())
    }

    fn from_mcp_with_config(value: CallToolResponse, config: &ConversionConfig) -> Result<Self, McpConversionError> {
        // Extract metadata from meta field
        let (id, name) = if let Some(meta) = &value.meta {
            if let Some(ai_ox) = meta.get("ai_ox") {
                if let Some(obj) = ai_ox.as_object() {
                    let id_opt = obj.get("call_id").and_then(|v| v.as_str());
                    let name_opt = obj.get("name").and_then(|v| v.as_str());
                    if config.strict {
                        let id = id_opt.ok_or_else(|| McpConversionError::InvalidFormat("Missing call_id in meta".to_string()))?;
                        let name = name_opt.ok_or_else(|| McpConversionError::InvalidFormat("Missing name in meta".to_string()))?;
                        (id.to_string(), name.to_string())
                    } else {
                        let id = id_opt.unwrap_or("").to_string();
                        let name = name_opt.unwrap_or("").to_string();
                        if id_opt.is_none() {
                            eprintln!("Warning: Missing call_id in meta, using empty string");
                        }
                        if name_opt.is_none() {
                            eprintln!("Warning: Missing name in meta, using empty string");
                        }
                        (id, name)
                    }
                } else {
                    if config.strict {
                        return Err(McpConversionError::InvalidFormat("Invalid ai_ox meta structure".to_string()));
                    } else {
                        eprintln!("Warning: Invalid ai_ox meta structure, using empty strings");
                        (String::new(), String::new())
                    }
                }
            } else {
                if config.strict {
                    return Err(McpConversionError::InvalidFormat("Missing ai_ox in meta".to_string()));
                } else {
                    eprintln!("Warning: Missing ai_ox in meta, using empty strings");
                    (String::new(), String::new())
                }
            }
        } else {
            if config.strict {
                return Err(McpConversionError::InvalidFormat("Missing meta field".to_string()));
            } else {
                eprintln!("Warning: Missing meta field, using empty strings");
                (String::new(), String::new())
            }
        };

        // Convert Vec<ToolResponseContent> to Vec<Message>
        let response = Vec::<ai_ox::content::Message>::from_mcp(value.content)?;

        // Propagate is_error flag to ext
        let mut ext = std::collections::BTreeMap::new();
        if value.is_error == Some(true) {
            ext.insert("mcp.is_error".to_string(), serde_json::Value::Bool(true));
        }

        Ok(Part::ToolResult {
            id,
            name,
            parts: response.into_iter().flat_map(|m| m.content).collect(),
            ext,
        })
    }
}

// McpTool conversions (if needed for tool definitions)
// TODO: Implement for FunctionMetadata instead of the enum