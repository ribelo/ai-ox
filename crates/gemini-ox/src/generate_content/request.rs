use bon::Builder;
use serde::Serialize;

use crate::{
    Gemini,
    tool::{Tool, config::ToolConfig},
};

use super::{GenerationConfig, SafetySettings};
use crate::content::Content;

#[derive(Debug, Serialize, Builder)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    #[builder(field)]
    pub contents: Vec<Content>,
    #[builder(field)]
    tools: Vec<Tool>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<SafetySettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_content: Option<String>,
    #[serde(skip)]
    pub(crate) gemini: Gemini,
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
        self.tools.push(tool.into());
        self
    }
    pub fn tools(mut self, tools: impl IntoIterator<Item = impl Into<Tool>>) -> Self {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }
}

impl Gemini {
    pub fn generate_content(
        &self,
    ) -> GenerateContentRequestBuilder<generate_content_request_builder::SetGemini> {
        GenerateContentRequest::builder().gemini(self.clone())
    }
}
