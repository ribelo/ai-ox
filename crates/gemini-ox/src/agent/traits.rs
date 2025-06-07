// use std::future::Future;
// use std::pin::Pin;

// use async_stream::try_stream;
// use futures_util::{Stream, StreamExt};
// use schemars::JsonSchema;
// use serde::de::DeserializeOwned;
// use serde::{Deserialize, Serialize};
// use serde_json::Value;

// use crate::generate_content::content::{self, Content};
// use crate::generate_content::part::FunctionCall;
// use crate::generate_content::response::GenerateContentResponse;
// use crate::generate_content::usage::UsageMetadata;
// use crate::tool::config::ToolConfig;
// use crate::tool::{FunctionCallError, ToolBox};
// use crate::{Gemini, GeminiRequestError, GenerationConfig, Model};

// use super::error::AgentError;
// use super::events::AgentEvent;

// pub trait Agent: Clone + Send + Sync + 'static {
//     fn name(&self) -> &str {
//         std::any::type_name::<Self>().split("::").last().unwrap()
//     }
//     fn description(&self) -> Option<&str> {
//         None
//     }
//     fn instructions(&self) -> Option<Content>;
//     fn model(&self) -> Model;
//     fn max_tokens(&self) -> Option<u32> {
//         None
//     }
//     fn stop_sequences(&self) -> Option<&Vec<String>> {
//         None
//     }
//     fn temperature(&self) -> Option<f64> {
//         None
//     }
//     fn top_p(&self) -> Option<f64> {
//         None
//     }
//     fn top_k(&self) -> Option<u64> {
//         None
//     }
//     fn tool_config(&self) -> Option<ToolConfig> {
//         None
//     }
//     fn max_iterations(&self) -> usize {
//         12
//     }
// }

// pub trait TypedAgent: Agent {
//     type Output: JsonSchema + DeserializeOwned;
// }
