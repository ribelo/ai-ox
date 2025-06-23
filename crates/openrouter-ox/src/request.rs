use async_stream::try_stream;
use bon::Builder;
use futures_util::stream::{BoxStream, StreamExt};
use schemars::{generate::SchemaSettings, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    message::{Message, Messages},
    provider_preference::ProviderPreferences,
    response::{ChatCompletionChunk, ChatCompletionResponse},
    tool::{ToolChoice, ToolSchema},
    ApiRequestError, ErrorResponse, OpenRouter, BASE_URL,
};

const API_URL: &str = "api/v1/chat/completions";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub r#type: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Builder)]
pub struct Request {
    #[builder(field)]
    pub messages: Messages,
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    #[serde(skip)]
    pub open_router: OpenRouter,
}

impl<S: request_builder::State> RequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = impl Into<Message>>) -> Self {
        self.messages = messages.into_iter().map(Into::into).collect();
        self
    }
    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }
    pub fn response_format<T: JsonSchema + DeserializeOwned>(mut self) -> Self {
        let type_name = std::any::type_name::<T>().split("::").last().unwrap();
        let mut schema_settings = SchemaSettings::draft2020_12();
        schema_settings.inline_subschemas = true;
        let schema_generator = schema_settings.into_generator();
        let json_schema = schema_generator.into_root_schema_for::<T>();
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {"name": type_name, "schema": json_schema},
        });
        self.response_format = Some(response_format);
        self
    }
}

impl Request {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }
    pub async fn send(&self) -> Result<ChatCompletionResponse, ApiRequestError> {
        let url = format!("{}/{}", BASE_URL, API_URL);

        // Use bearer_auth instead of manually constructing the Authorization header
        let res = self
            .open_router
            .client
            .post(&url)
            .bearer_auth(&self.open_router.api_key)
            .json(self)
            .send()
            .await?;

        if res.status().is_success() {
            // Parse the response body directly to the target type
            Ok(res.json::<ChatCompletionResponse>().await?)
        } else {
            Err(ApiRequestError::InvalidRequestError(res.json().await?))
        }
    }

    // pub fn stream(&self) -> Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, ApiRequestError>> + Send>> {
    pub fn stream(&self) -> BoxStream<'static, Result<ChatCompletionChunk, ApiRequestError>> {
        let client = self.open_router.client.clone();
        let api_key = self.open_router.api_key.clone();
        let url = format!("{}/{}", BASE_URL, API_URL);
        let request_data = self.clone();

        Box::pin(try_stream! {
            // Prepare request body with streaming enabled
            let mut body = serde_json::to_value(&request_data)?;
            body.as_object_mut()
                .expect("Request body must be a JSON object")
                .insert("stream".to_string(), serde_json::Value::Bool(true));

            // Send request
            let response = client
                .post(&url)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await?;
            let status = response.status();

            if !response.status().is_success() {
                // Handle error response
                match response.json::<ErrorResponse>().await {
                    Ok(error_response) => {
                        Err(ApiRequestError::InvalidRequestError(error_response))?
                    }
                    Err(json_err) => {
                        Err(ApiRequestError::Stream(format!(
                            "API error (status {status}): Failed to parse error response: {json_err}",
                        )))?
                    }
                }
            } else {
                // Process successful streaming response
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| ApiRequestError::Stream(format!("UTF-8 decode error: {}", e)))?;

                    for parse_result in ChatCompletionChunk::from_streaming_data(&chunk_str) {
                        yield parse_result?;
                    }
                }
            }
        })
    }
}

impl OpenRouter {
    pub fn chat_completion(&self) -> RequestBuilder<request_builder::SetOpenRouter> {
        Request::builder().open_router(self.clone())
    }
}

