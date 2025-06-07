use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use tokio_stream::Stream;

use super::error::AgentError;
use super::executor::AgentExecutor;
use super::traits::{AgentCore, TypedAgent};
use crate::{
    message::Messages, response::ChatCompletionResponse, tool::ToolBox, ApiRequestError, OpenRouter,
};

#[derive(Debug, Clone, Builder)]
pub struct SimpleTypedAgent<T: JsonSchema + Serialize + DeserializeOwned> {
    #[builder(into)]
    pub name: Option<String>,
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
    pub tools: Option<ToolBox>,
    pub max_iterations: Option<usize>,
    #[builder(default)]
    pub output: PhantomData<T>,
}

#[async_trait]
impl<T> AgentCore for SimpleTypedAgent<T>
where
    T: Clone + JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn agent_name(&self) -> &str {
        self.name
            .as_deref()
            .unwrap_or_else(|| std::any::type_name::<Self>().split("::").last().unwrap())
    }

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

// Implement TypedAgent for SimpleTypedAgent
#[async_trait]
impl<T> TypedAgent for SimpleTypedAgent<T>
where
    T: Clone + JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    type Output = T;

    async fn once_typed(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError> {
        executor.execute_once_typed(self, messages).await
    }

    async fn run_typed(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError> {
        executor.execute_run_typed(self, messages).await
    }
}
