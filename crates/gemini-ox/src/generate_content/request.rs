use bon::Builder;
use serde::Serialize;
use serde_json::Value;

use crate::tool::{Tool, config::ToolConfig};

use super::{GenerationConfig, SafetySettings};
use crate::content::Content;

#[derive(Debug, Clone, Serialize, Builder)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    #[builder(field)]
    pub contents: Vec<Content>,
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<SafetySettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
}

impl<S: generate_content_request_builder::State> GenerateContentRequestBuilder<S> {
    pub fn content_list(mut self, contents: impl IntoIterator<Item = impl Into<Content>>) -> Self {
        self.contents = contents.into_iter().map(Into::into).collect();
        self
    }
    pub fn content(mut self, content: impl Into<Content>) -> Self {
        self.contents.push(content.into());
        self
    }
    pub fn tool(mut self, tool: impl Into<Tool>) -> Self {
        self.tools
            .get_or_insert_default()
            .push(serde_json::to_value(tool.into()).unwrap());
        self
    }
    pub fn tools(mut self, tools: impl IntoIterator<Item = impl Into<Tool>>) -> Self {
        self.tools.get_or_insert_default().extend(
            tools
                .into_iter()
                .map(|tool| serde_json::to_value(tool.into()).unwrap()),
        );
        self
    }
}

// impl Gemini {
//     pub fn generate_content(
//         &self,
//     ) -> GenerateContentRequestBuilder<generate_content_request_builder::SetGemini> {
//         GenerateContentRequest::builder().gemini(self.clone())
//     }
// }
