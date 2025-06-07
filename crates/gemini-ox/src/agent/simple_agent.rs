use async_trait::async_trait;
use bon::Builder;
use std::pin::Pin;
use tokio_stream::Stream;

use super::error::AgentError;
use super::events::AgentEvent;
use super::executor::AgentExecutor;
use super::traits::{Agent, AgentCore};
use crate::{
    message::Messages, response::ChatCompletionResponse, tool::Tool, tool::ToolBox,
    ApiRequestError, OpenRouter,
};

#[derive(Debug, Clone, Builder)]
pub struct SimpleAgent {
    #[builder(field)]
    pub tools: Option<ToolBox>,
    #[builder(into)]
    pub description: Option<String>,
    #[builder(into)]
    pub instructions: Option<String>,
    #[builder(into)]
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_iterations: Option<usize>,
}

impl<S: simple_agent_builder::State> SimpleAgentBuilder<S> {
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        match &mut self.tools {
            Some(toolbox) => {
                toolbox.add(tool);
            }
            None => {
                let toolbox = ToolBox::builder();
                self.tools = Some(toolbox.tool(tool).build());
            }
        }
        self
    }
}

#[async_trait]
impl AgentCore for SimpleAgent {
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn stop_sequences(&self) -> Option<&Vec<String>> {
        self.stop_sequences.as_ref()
    }

    fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    fn tools(&self) -> Option<&ToolBox> {
        self.tools.as_ref()
    }

    fn max_iterations(&self) -> usize {
        self.max_iterations.unwrap_or(12)
    }

    async fn once(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, ApiRequestError> {
        executor.execute_once(self, messages).await
    }

    fn stream_once(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Pin<
        Box<
            dyn Stream<Item = Result<crate::response::ChatCompletionChunk, ApiRequestError>> + Send,
        >,
    > {
        executor.stream_once(self, messages)
    }
}
