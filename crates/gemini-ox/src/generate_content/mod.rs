use crate::{
    Gemini,
    content::{Content, Part, Role},
    internal::GeminiRequestHelper,
};
use bon::Builder;
use futures_util::stream::{self, BoxStream};
use request::GenerateContentRequest;
use response::GenerateContentResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::GeminiRequestError;

pub mod request;
pub mod response;
pub mod usage;

// Re-export commonly used types
// Re-export speech configuration types from live module to avoid duplication
pub use crate::live::request_configs::{PrebuiltVoiceConfig, SpeechConfig, VoiceConfig};

impl GenerateContentRequest {
    /// Sends a generate content request to the Gemini API
    ///
    /// This method makes a non-streaming request to the Gemini API and returns the complete response.
    ///
    /// # Errors
    ///
    /// This function can return the following error variants:
    /// - `GeminiRequestError::ReqwestError` - if the HTTP request fails
    /// - `GeminiRequestError::RateLimit` - if the API rate limit is exceeded (HTTP 429)
    /// - `GeminiRequestError::InvalidRequestError` - if the API returns a 4xx/5xx error with structured error data
    /// - `GeminiRequestError::JsonDeserializationError` - if the API response cannot be parsed as JSON
    /// - `GeminiRequestError::UnexpectedResponse` - if the API returns an unexpected response format or error
    pub async fn send(
        &self,
        gemini: &Gemini,
    ) -> Result<GenerateContentResponse, GeminiRequestError> {
        let helper = GeminiRequestHelper::for_generate(gemini)?;
        helper.send_generate_content_request(self, gemini).await
    }

    /// Streams a generate content request from the Gemini API
    ///
    /// This method makes a streaming request to the Gemini API using Server-Sent Events (SSE)
    /// and returns a stream of individual response chunks.
    ///
    /// # Note
    ///
    /// The stream implementation buffers data to handle lines split across network chunks
    /// and ensures proper UTF-8 decoding.
    ///
    /// # Errors
    ///
    /// The returned stream can yield the following error variants:
    /// - `GeminiRequestError::ReqwestError` - if the HTTP request fails
    /// - `GeminiRequestError::RateLimit` - if the API rate limit is exceeded (HTTP 429)
    /// - `GeminiRequestError::InvalidRequestError` - if the API returns a 4xx/5xx error with structured error data
    /// - `GeminiRequestError::JsonDeserializationError` - if an SSE data entry cannot be parsed as JSON
    /// - `GeminiRequestError::InvalidEventData` - if the stream contains invalid UTF-8 or other stream format issues
    /// - `GeminiRequestError::UnexpectedResponse` - if the API returns an unexpected response format or error
    #[must_use]
    pub fn stream(
        &self,
        gemini: &Gemini,
    ) -> BoxStream<'static, Result<GenerateContentResponse, GeminiRequestError>> {
        match GeminiRequestHelper::for_generate(gemini) {
            Ok(helper) => helper.stream_generate_content_request(self.clone(), gemini.clone()),
            Err(err) => Box::pin(stream::once(async move { Err(err) })),
        }
    }

    #[must_use]
    pub fn push_content(mut self, content: impl Into<Content>) -> Self {
        self.contents.push(content.into());
        self
    }
}

impl From<GenerateContentResponse> for Content {
    fn from(value: GenerateContentResponse) -> Self {
        // Note: This implementation clones the parts from each candidate's content.
        // This is generally acceptable because:
        // 1. Most `Part` instances contain small data (text) that is cheap to clone
        // 2. For binary data, it's a reference-counted String (for the base64 data), not the raw bytes
        // 3. The cost of cloning is outweighed by the convenience of the conversion API
        let parts = value
            .candidates
            .iter()
            .flat_map(|candidate| candidate.content.parts().clone())
            .collect::<Vec<_>>();
        Content::builder().role(Role::Model).parts(parts).build()
    }
}

impl From<GenerateContentResponse> for Vec<Part> {
    fn from(value: GenerateContentResponse) -> Self {
        // Note: This implementation clones each part via `to_owned`.
        // `Part` contains string data or references to string data that's cheap to clone.
        // Even for binary content (Blob), it's storing base64 strings, not large byte arrays.
        value
            .candidates
            .iter()
            .flat_map(|candidate| candidate.content.parts())
            .map(Part::to_owned)
            .collect()
    }
}

#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, strum::EnumString, strum::Display,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
/// Output only. The reason why the model stopped generating tokens.
///
/// If empty, the model has not stopped generating the tokens.
pub enum FinishReason {
    /// The finish reason is unspecified.
    FinishReasonUnspecified,
    /// Token generation reached a natural stopping point or a configured stop sequence.
    Stop,
    /// Token generation reached the configured maximum output tokens.
    MaxTokens,
    /// Token generation stopped because the content potentially contains safety violations.
    /// NOTE: When streaming, content is empty if content filters blocks the output.
    Safety,
    /// The token generation stopped because of potential recitation.
    Recitation,
    /// The token generation stopped because of using an unsupported language.
    Language,
    /// All other reasons that stopped the token generation.
    Other,
    /// Token generation stopped because the content contains forbidden terms.
    Blocklist,
    /// Token generation stopped for potentially containing prohibited content.
    ProhibitedContent,
    /// Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
    Spii,
    /// The function call generated by the model is invalid.
    MalformedFunctionCall,
    /// Token generation stopped because generated images have safety violations.
    ImageSafety,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    pub citation_sources: Vec<CitationSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    pub start_index: Option<u32>,
    pub end_index: Option<u32>,
    pub uri: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingAttribution {
    pub source_id: AttributionSourceId,
    pub content: Content,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttributionSourceId {
    #[serde(flatten)]
    pub source: AttributionSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum AttributionSource {
    GroundingPassage(GroundingPassageId),
    SemanticRetrieverChunk(SemanticRetrieverChunk),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingPassageId {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passage_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_index: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticRetrieverChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Web {
    pub uri: String,
    pub title: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GroundingChunk {
    Web(Web),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    pub grounding_chunk_indices: Vec<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_scores: Option<Vec<f64>>,
    pub segment: Segment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    pub end_index: i32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_blob: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_dynamic_retrieval_score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    grounding_chunks: Vec<GroundingChunk>,
    grounding_supports: Vec<GroundingSupport>,
    web_search_queries: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_entry_point: Option<SearchEntryPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retrieval_metadata: Option<RetrievalMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogprobsResult {
    pub top_candidates: Vec<TopLogpropsCandidates>,
    pub chosen_candidates: Vec<LogpropsCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopLogpropsCandidates {
    pub candidates: Vec<LogpropsCandidate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogpropsCandidate {
    pub token: String,
    pub token_id: i32,
    pub log_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseCandidate {
    pub content: Content,
    pub finish_reason: Option<FinishReason>,
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
    pub citation_metadata: Option<CitationMetadata>,
    pub token_count: Option<u32>,
    pub grounding_attributions: Option<Vec<GroundingAttribution>>,
    pub grounding_metadata: Option<GroundingMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprobs: Option<f64>,
    pub logprobs_result: Option<LogprobsResult>,
    pub index: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockReason {
    BlockReasonUnspecified,
    Safety,
    Other,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    pub block_reason: Option<BlockReason>,
    pub safety_ratings: Vec<SafetyRating>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetySettings(Vec<SafetySetting>);

impl Default for SafetySettings {
    fn default() -> Self {
        Self(Vec::default())
            // .category(
            //     HarmCategory::HarmCategoryUnspecified,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryDerogatory,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryToxicity,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryViolence,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategorySexual,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryMedical,
            //     HarmBlockThreshold::BlockNone,
            // )
            // .category(
            //     HarmCategory::HarmCategoryDangerous,
            //     HarmBlockThreshold::BlockNone,
            // )
            .with_category(
                HarmCategory::HarmCategoryHarassment,
                HarmBlockThreshold::BlockNone,
            )
            .with_category(
                HarmCategory::HarmCategoryHateSpeech,
                HarmBlockThreshold::BlockNone,
            )
            .with_category(
                HarmCategory::HarmCategorySexuallyExplicit,
                HarmBlockThreshold::BlockNone,
            )
            .with_category(
                HarmCategory::HarmCategoryDangerousContent,
                HarmBlockThreshold::BlockNone,
            )
    }
}

impl SafetySettings {
    #[must_use]
    pub fn with_category(mut self, category: HarmCategory, threshold: HarmBlockThreshold) -> Self {
        self.0.push((category, threshold).into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
}

impl From<(HarmCategory, HarmBlockThreshold)> for SafetySetting {
    fn from(value: (HarmCategory, HarmBlockThreshold)) -> Self {
        SafetySetting {
            category: value.0,
            threshold: value.1,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    HarmCategoryUnspecified,
    HarmCategoryDerogatory,
    HarmCategoryToxicity,
    HarmCategoryViolence,
    HarmCategorySexual,
    HarmCategoryMedical,
    HarmCategoryDangerous,
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    HarmBlockThresholdUnspecified,
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    pub category: HarmCategory,
    pub probability: String,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Indicates whether to include thoughts in the response.
    /// If true, thoughts are returned only when available.
    pub include_thoughts: bool,

    /// The number of thoughts tokens that the model should generate.
    pub thinking_budget: i32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Builder)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(with = |v: impl IntoIterator<Item = impl Into<String>>| v.into_iter().map(Into::into).collect())]
    pub stop_sequences: Option<Vec<String>>,
    /// Output response mimetype of the generated candidate text. Supported mimetype: text/plain: (default) Text output. application/json: JSON response in the candidates.
    #[builder(into)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Output response schema of the generated candidate text when response mime type can have schema. Schema can be objects, primitives or arrays and is a subset of OpenAPI schema.
    /// If set, a compatible responseMimeType must also be set. Compatible mimetypes: application/json: Schema for JSON response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,

    /// Specifies the modalities of the response from the model (e.g., TEXT, AUDIO).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(with = |v: impl IntoIterator<Item = impl Into<String>>| v.into_iter().map(Into::into).collect())]
    pub response_modalities: Option<Vec<String>>,
    /// Number of generated responses to return.
    /// Currently, this value can only be set to 1. If unset, this will default to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    /// The maximum number of tokens to include in a candidate.
    /// Note: The default value varies by model, see the `Model.output_token_limit` attribute of the Model returned from the getModel function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Controls the randomness of the output.
    /// Note: The default value varies by model, see the Model.temperature attribute of the Model returned from the getModel function.
    /// Values can range from [0.0, 2.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// The maximum cumulative probability of tokens to consider when sampling.
    /// The model uses combined Top-k and nucleus sampling.
    /// Tokens are sorted based on their assigned probabilities so that only the most likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while Nucleus sampling limits number of tokens based on the cumulative probability.
    /// Note: The default value varies by model, see the `Model.top_p` attribute of the Model returned from the getModel function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// The maximum number of tokens to consider when sampling.
    /// Models use nucleus sampling or combined Top-k and nucleus sampling. Top-k sampling considers the set of topK most probable tokens. Models running with nucleus sampling don't allow topK setting.
    /// Note: The default value varies by model, see the `Model.top_k` attribute of the Model returned from the getModel function. Empty topK field in Model indicates the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    /// Config for thinking features.
    /// Note: An error will be returned if this field is set for models that don't support thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,

    /// Optional. The speech generation config.
    #[serde(skip_serializing_if = "Option::is_none")]
    // No #[builder(default)] needed as Option fields default to None and builder method will take SpeechConfig
    pub speech_config: Option<SpeechConfig>,
}

// #[cfg(test)]
// mod tests {
//     use crate::{
//         Gemini, Model, ResponseSchema,
//         tool::{AsTools, error::FunctionCallError, google::GoogleSearch},
//     };
//     use schemars::JsonSchema;
//     use serde::{Deserialize, Serialize};
//     use serde_json::{Value, json}; // Added Serialize import
//     use std::sync::{
//         Arc,
//         atomic::{AtomicBool, Ordering},
//     }; // Adjusted sync imports

//     use super::*;

//     // fn get_api_key() -> String {
//     //     std::env::var("GOOGLE_AI_API_KEY").expect("GOOGLE_AI_API_KEY must be set")
//     // }

//     #[tokio::test]
//     #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
//     async fn test_generate_content_request() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
//     {
//         let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
//             // Skip test if API key isn't set instead of failing
//             println!("GOOGLE_AI_API_KEY not set, skipping test_generate_content_request");
//             return Ok(());
//         };

//         let gemini = Gemini::builder().api_key(api_key).build();
//         let request = gemini
//             .generate_content()
//             .content_list(vec![Content::from("hello")])
//             .model("gemini-1.5-flash")
//             .build();

//         let mut stream = request.stream();

//         let mut responses = Vec::new();
//         let mut stream_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

//         // Set a reasonable timeout (20 seconds) for the test
//         let timeout = tokio::time::timeout(std::time::Duration::from_secs(20), async {
//             while let Some(result) = stream.next().await {
//                 match result {
//                     Ok(item) => {
//                         // Extract text from the candidate's content if available
//                         if let Some(candidate) = item.candidates.first() {
//                             if let Some(part) = candidate.content.parts().first() {
//                                 if let Some(text) = part.as_text() {
//                                     responses.push(text.to_string());
//                                 }
//                             }
//                         }
//                     }
//                     Err(err) => {
//                         // Store the error instead of panicking
//                         stream_error = Some(Box::new(err));
//                         break;
//                     }
//                 }
//             }
//         })
//         .await;

//         // Check for timeout
//         if timeout.is_err() {
//             return Err(Box::<dyn std::error::Error + Send + Sync>::from(
//                 "Test timed out after 30 seconds ",
//             ));
//         }

//         // Check if we encountered a stream error
//         if let Some(err) = stream_error {
//             return Err(err);
//         }

//         assert!(
//             !responses.is_empty(),
//             "Stream should yield at least one response"
//         );

//         // Verify that the collected responses form a coherent response
//         let full_response = responses.join("");
//         assert!(
//             !full_response.is_empty(),
//             "Combined response should not be empty"
//         );
//         assert!(
//             full_response.len() > 10,
//             "Response should have meaningful content"
//         );

//         Ok(())
//     }

//     // --- Service and Tools for test_function_calling ---
//     #[derive(Clone, Debug, Default)]
//     struct FunctionCallingService;

//     #[derive(Serialize, Deserialize, JsonSchema, Debug)]
//     pub struct ToolOneParams {
//         pub number: i32,
//     }

//     #[derive(Serialize, Deserialize, JsonSchema, Debug)]
//     pub struct GenericParams {
//         #[serde(flatten)]
//         extra: serde_json::Value,
//     }

//     #[toolbox]
//     impl FunctionCallingService {
//         /// A simple tool that takes a number.
//         #[allow(clippy::unused_async)] // Added because async is required by macro but not used
//         pub async fn tool_one(&self, params: ToolOneParams) -> Result<String, FunctionCallError> {
//             Ok(format!("tool_one executed with number: {}", params.number))
//         }

//         /// A simple tool that takes generic JSON.
//         #[allow(clippy::unused_async)] // Added
//         pub async fn tool_two(&self, params: Option<GenericParams>) -> Result<String, String> {
//             Ok(format!("tool_two executed with params: {:?}", params))
//         }
//     }
//     // --- End Service and Tools for test_function_calling ---

//     #[tokio::test]
//     #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
//     async fn test_stream_tools() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         // Test with Arc<ToolBox>
//         let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
//             // Skip test if API key isn't set instead of failing
//             println!("GOOGLE_AI_API_KEY not set, skipping test_stream_tools ");
//             return Ok(());
//         };

//         let gemini = Gemini::builder().api_key(api_key).build();
//         // Instantiate the service and create the ToolBox
//         let service = FunctionCallingService; // Use unit struct directly

//         let request = gemini
//             .generate_content()
//             .content_list(vec![Content::from(
//                 r#"to finish this test use both "tool_one", "tool_two" tools"#,
//             )])
//             .model("gemini-1.5-flash")
//             .tools(service.as_tools())
//             .build();

//         // Set a reasonable timeout (15 seconds) for the test
//         let response =
//             match tokio::time::timeout(std::time::Duration::from_secs(15), request.send()).await {
//                 Ok(result) => result?,
//                 Err(_) => {
//                     return Err(Box::<dyn std::error::Error + Send + Sync>::from(
//                         "Request timed out after 15 seconds",
//                     ));
//                 }
//             };

//         // Validate the response structure
//         assert!(
//             !response.candidates.is_empty(),
//             "Response should have at least one candidate"
//         );

//         // NOTE: invoke_functions is commented out as it relies on the old Tool trait structure.
//         // The test now primarily verifies that the request with tools completes successfully.
//         let content = response.invoke_functions(service.clone()).await?;

//         // Verify that functions were invoked
//         if let Some(function_content) = content {
//             // Check that the function content contains tool responses
//             let text: String = function_content
//                 .parts()
//                 .iter()
//                 .filter_map(|p| p.as_text().map(ToString::to_string))
//                 .collect();

//             // For some models, sometimes no tool might be invoked or only basic responses
//             // are returned, especially when running in automated test environments
//             println!("Function response: {text}");
//         } else {
//             println!("No function invocation in response - API behavior can vary");
//         }
//         // Check if the response contains a function call part (basic check)
//         let has_function_call = response.candidates.iter().any(|c| {
//             c.content
//                 .parts()
//                 .iter()
//                 .any(|p| p.data.as_function_call().is_some())
//         });
//         println!("Response contains function call part: {has_function_call}");
//         // A more robust test would check the *specific* function call expected,
//         // but this provides a basic validation that tool usage was attempted.

//         Ok(())
//     }

//     #[tokio::test]
//     async fn test_json_output() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         #[derive(Debug, Serialize, Deserialize, JsonSchema)]
//         pub struct Book {
//             #[schemars(description = "Unique identifier for the book")]
//             pub id: i64,

//             #[schemars(description = "Title of the book")]
//             pub title: String,

//             #[schemars(description = "Author of the book")]
//             pub author: String,

//             #[schemars(description = "ISBN (International Standard Book Number)")]
//             pub isbn: String,

//             #[schemars(description = "Genre of the book")]
//             pub genre: String,

//             #[schemars(description = "Number of pages in the book")]
//             #[schemars(range(min = 1))]
//             pub page_count: i32,

//             #[schemars(description = "Average rating of the book")]
//             #[schemars(range(min = 0.0, max = 5.0))]
//             pub rating: f32,

//             #[schemars(description = "Whether the book is currently available")]
//             pub available: bool,

//             #[schemars(description = "Tags associated with the book")]
//             #[schemars(length(max = 5))]
//             pub tags: Vec<String>,
//         }

//         let api_key = match std::env::var("GOOGLE_AI_API_KEY") {
//             Ok(key) => key,
//             Err(_) => {
//                 // Skip test if API key isn't set instead of failing
//                 println!("GOOGLE_AI_API_KEY not set, skipping test_json_output ");
//                 return Ok(());
//             }
//         };
//         let gemini = Gemini::builder().api_key(api_key).build();
//         let response_schema = ResponseSchema::from::<Book>();
//         let config = GenerationConfig::builder()
//             .response_mime_type("application/json".to_string())
//             .response_schema(response_schema)
//             .build();

//         let request = gemini
//             .generate_content()
//             .content_list(vec![r"Describe Peter Watts Echopraxia book"])
//             .model("gemini-1.5-flash")
//             .generation_config(config)
//             .build();

//         // Send the request
//         let response = request.send().await?;

//         // Verify response structure
//         assert!(
//             !response.candidates.is_empty(),
//             "Response should have at least one candidate"
//         );
//         assert!(
//             !response.candidates[0].content.parts().is_empty(),
//             "Response should have content parts"
//         );

//         let text = response.candidates[0].content.parts()[0]
//             .as_text()
//             .ok_or("Response part should be text")?
//             .to_string();

//         // Verify the response is valid JSON and can be parsed as a Book
//         let json = serde_json::from_str::<Book>(&text)?;

//         // Verify book properties
//         assert_eq!(json.author, "Peter Watts", "Author should be Peter Watts");
//         assert_eq!(json.title, "Echopraxia", "Title should be Echopraxia");
//         assert!(json.page_count > 0, "Page count should be positive");
//         assert!(
//             json.rating >= 0.0 && json.rating <= 5.0,
//             "Rating should be between 0 and 5"
//         );
//         assert!(json.tags.len() <= 5, "Should have at most 5 tags");

//         Ok(())
//     }

//     // --- Service and Tools for test_messages_request_success ---
//     #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
//     struct TestHandlerProps {
//         /// Any valid number will be good.
//         random_number: i32,
//     }

//     #[derive(Clone, Debug, Default)]
//     struct MultiTurnService {
//         test_tool_called: Arc<AtomicBool>,
//         finish_tool_called: Arc<AtomicBool>,
//     }

//     #[toolbox]
//     impl MultiTurnService {
//         /// First tool to call in the sequence.
//         #[allow(clippy::unused_async)]
//         pub async fn test_tool(
//             &self,
//             _input: TestHandlerProps,
//         ) -> Result<Value, FunctionCallError> {
//             self.test_tool_called.store(true, Ordering::Relaxed);
//             Ok(
//                 json! ({"message": "To finish this test use [finish_tool] tool. Don't ask any question to user. You are on your own."}),
//             )
//         }

//         /// Final tool to call in the sequence.
//         #[allow(clippy::unused_async)]
//         pub async fn finish_tool(
//             &self,
//             _input: Option<serde_json::Value>,
//         ) -> Result<String, String> {
//             // Input is Option<Value> as it might not be provided by the LLM
//             self.finish_tool_called.store(true, Ordering::Relaxed);
//             Ok("Congratulations! You finished the test.".to_string())
//         }
//     }
//     // --- End Service and Tools for test_messages_request_success ---

//     #[tokio::test]
//     async fn test_messages_request_success() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
//     {
//         let api_key = match std::env::var("GOOGLE_AI_API_KEY") {
//             Ok(key) => key,
//             Err(_) => {
//                 // Skip test if API key isn't set instead of failing
//                 println!("GOOGLE_AI_API_KEY not set, skipping test_messages_request_success");
//                 return Ok(());
//             }
//         };

//         let gemini = Gemini::builder().api_key(api_key).build();

//         // Instantiate the service and create the ToolBox
//         let service = MultiTurnService::default();

//         // Get handles to the atomic flags for assertion
//         let test_tool_called_handle = service.test_tool_called.clone();
//         let finish_tool_called_handle = service.finish_tool_called.clone();

//         let mut contents = vec![Content::from(
//             "This is testing environment. To continue this test use [test_tool]",
//         )];

//         let mut i = 0;
//         let max_iterations = 3; // Set a reasonable limit to avoid infinite loops

//         // Set a reasonable timeout (30 seconds) for the whole test
//         let timeout = tokio::time::timeout(std::time::Duration::from_secs(30), async {
//             loop {
//                 // Ensure we don't loop indefinitely
//                 if i >= max_iterations {
//                     // If we reached the max iterations but tools were called successfully,
//                     // consider the test passed (don't assert)
//                     if test_tool_called_handle.load(Ordering::Relaxed)
//                         && finish_tool_called_handle.load(Ordering::Relaxed)
//                     {
//                         println!("Test passed after reaching max_iterations");
//                     } else {
//                         // Only assert if tools weren't called
//                         assert!(
//                             test_tool_called_handle.load(Ordering::Relaxed),
//                             "test_tool should have been called within {max_iterations} iterations "
//                         );
//                         assert!(
//                             finish_tool_called_handle.load(Ordering::Relaxed),
//                             "finish_tool should have been called within {max_iterations} iterations "
//                         );
//                     }
//                     break;
//                 }
//                 i += 1;

//                 let request = gemini
//                     .generate_content()
//                     .content_list(contents.clone())
//                     .model(Model::Gemini25FlashPreview0520)
//                     .tools(service.as_tools()) // Removed: Tools are not set here in the new structure
//                     .build();
//                 println!("{i}\n{}", serde_json::to_string_pretty(&request).unwrap());

//                 // Send the request
//                 let response = match request.send().await {
//                     Ok(resp) => resp,
//                     Err(e) => return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
//                 };

//                 // Add model response to conversation history
//                 contents.push(response.clone().into());

//                 // NOTE: invoke_functions is commented out as it relies on the old Tool trait structure.
//                 // The test now primarily verifies that the request loop with tools completes
//                 // and checks the atomic flags set by the tool methods directly.

//                 // Process function calls
//                 let tool_results = match response.invoke_functions(service.clone()).await {
//                     Ok(result) => result,
//                     Err(e) => return Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
//                 };

//                 match tool_results {
//                     None => {
//                         contents.push(Content::from("Follow instructions and use tools!"));
//                     }
//                     Some(results) => {
//                         contents.push(results);
//                     }
//                 }

//                 // Check if the response *contains* a function call part to simulate adding results
//                  if let Some(candidate) = response.candidates.first() {
//                      if candidate.content.parts().iter().any(|p| p.data.as_function_call().is_some()) {
//                          // Simulate adding a generic function response part if a call was made.
//                          // In a real scenario, you'd need a different way to get the *actual*
//                          // result from the toolbox based on the call details.
//                          contents.push(Content::function_response(
//                               "simulated_tool".to_string(),
//                               json!({"status": "simulated execution"}), // Use json! macro
//                           ).expect("Failed to create simulated response"));
//                       }
//                   }

//                 // Check if we've achieved the success condition by checking the atomic flags
//                 if finish_tool_called_handle.load(Ordering::Relaxed)
//                     && test_tool_called_handle.load(Ordering::Relaxed)
//                 {
//                     break;
//                 }
//             }

//             // Success
//             Ok(())
//         }).await;

//         // Check for timeout
//         if timeout.is_err() {
//             return Err(Box::<dyn std::error::Error + Send + Sync>::from(
//                 "Test timed out after 30 seconds ",
//             ));
//         }

//         // Check for errors inside the timeout block
//         match timeout.unwrap() {
//             Ok(_) => {
//                 // Final assertions using the handles
//                 assert!(
//                     test_tool_called_handle.load(Ordering::Relaxed),
//                     "test_tool should have been called "
//                 );
//                 assert!(
//                     finish_tool_called_handle.load(Ordering::Relaxed),
//                     "finish_tool should have been called "
//                 );
//                 Ok(())
//             }
//             Err(e) => Err(e),
//         }
//     }

//     // // --- Service and Tools for test_stream_tools ---
//     // #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
//     // struct AddToolInput {
//     //     a: i32,
//     //     b: i32,
//     // }

//     // #[derive(Clone, Debug, Default)]
//     // struct StreamingToolService;

//     // #[toolbox]
//     // impl StreamingToolService {
//     //     /// Adds two numbers.
//     //     #[allow(clippy::unused_async)]
//     //     pub async fn add_tool(&self, input: AddToolInput) -> Result<String, FunctionCallError> {
//     //         let result = input.a + input.b;
//     //         Ok(format!("{} + {} = {}", input.a, input.b, result))
//     //     }
//     // }
//     // // --- End Service and Tools for test_stream_tools ---

//     // #[tokio::test]
//     // async fn test_stream_tools() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//     //     let api_key = match std::env::var("GOOGLE_AI_API_KEY") {
//     //         Ok(key) => key,
//     //         Err(_) => {
//     //             // Skip test if API key isn't set instead of failing
//     //             println!("GOOGLE_AI_API_KEY not set, skipping test_stream_tools ");
//     //             return Ok(());
//     //         }
//     //     };

//     //     let gemini = Gemini::builder().api_key(api_key).build();
//     //     let service = StreamingToolService::default();
//     //     let tools = service.into_tool_box(); // Use IntoToolBox

//     //     let initial_prompt = Content::text(
//     //         "This is a testing environment. Add 2 and 2 using the [add_tool]. Do not explain, just use the tool directly.",
//     //     );

//     //     let mut contents = vec![initial_prompt];
//     //     let request = gemini
//     //         .generate_content()
//     //         .content_list(contents.clone())
//     //         .model(Model::Gemini20Flash)
//     //         // .tools(tools.clone()) // Removed: Tools are not set here in the new structure
//     //         .build();

//     //     let mut stream = request.stream();
//     //     let mut function_call_detected = false;
//     //     let mut response_parts = Vec::new();
//     //     let mut stream_error = None;
//     //     let mut correct_result_detected = false; // Flag to check if the correct result was simulated

//     //     // Set a reasonable timeout (30 seconds) for the test
//     //     let timeout = tokio::time::timeout(std::time::Duration::from_secs(30), async {
//     //         // Collect stream responses until we get a complete function call
//     //         while let Some(result) = stream.next().await {
//     //             match result {
//     //                 Ok(response) => {
//     //                     // Store response parts for later use
//     //                     response_parts.push(response.clone());

//     //                     // Add response content to the conversation history
//     //                     contents.extend(response.content_owned());

//     //                     // Check if we have a function call
//     //                     let function_calls: Vec<_> = response.function_calls().collect();
//     //                     if !function_calls.is_empty() {
//     //                         function_call_detected = true;

//     //                         // NOTE: invoke_functions is commented out.
//     //                         // Instead, simulate adding a function result based on the call.
//     //                         for fc in function_calls {
//     //                             if fc.name == "add_tool" {
//     //                                 // Simulate the expected output of add_tool(2, 2)
//     //                                 let simulated_result_content = Content::function_response(
//     //                                     "add_tool".to_string(),
//     //                                     json!("2 + 2 = 4"), // Use json! macro
//     //                                 )
//     //                                 .expect("Failed to create simulated result");

//     //                                 // Add the simulated result to conversation
//     //                                 contents.push(simulated_result_content.clone());

//     //                                 // Check if the simulated result is correct
//     //                                 let fn_text: String = simulated_result_content
//     //                                     .parts()
//     //                                     .iter()
//     //                                     .filter_map(|p| p.as_text().map(ToString::to_string))
//     //                                     .collect::<String>();

//     //                                 if fn_text.contains("2 + 2 = 4") {
//     //                                     correct_result_detected = true;
//     //                                     // Test complete at this point
//     //                                     break;
//     //                                 }
//     //                             }
//     //                         }
//     //                         if correct_result_detected {
//     //                             break;
//     //                         } // Exit loop if correct result simulated
//     //                     }
//     //                 }
//     //                 Err(e) => {
//     //                     stream_error =
//     //                         Some(Box::new(e) as Box<dyn std::error::Error + Send + Sync>);
//     //                     break;
//     //                 }
//     //             }
//     //         }
//     //     })
//     //     .await;

//     //     // Check for timeout
//     //     if timeout.is_err() {
//     //         return Err(Box::<dyn std::error::Error + Send + Sync>::from(
//     //             "Test timed out after 30 seconds ",
//     //         ));
//     //     }

//     //     // Check if we encountered a stream error
//     //     if let Some(err) = stream_error {
//     //         return Err(err);
//     //     }

//     //     // Verify we found at least one function call
//     //     assert!(
//     //         function_call_detected,
//     //         "Should detect at least one function call in the stream "
//     //     );
//     //     assert!(
//     //         !response_parts.is_empty(),
//     //         "Should receive at least one response part "
//     //     );
//     //     assert!(
//     //         correct_result_detected,
//     //         "Correct simulated tool result ('2 + 2 = 4') was not detected "
//     //     );

//     //     Ok(())
//     // }
//     #[tokio::test]
//     async fn test_google_search() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//         let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
//             // Skip test if API key isn't set instead of failing
//             println!("GOOGLE_AI_API_KEY not set, skipping test_generate_content_request");
//             return Ok(());
//         };

//         let gemini = Gemini::builder().api_key(api_key).build();
//         let request = gemini
//             .generate_content()
//             .content_list(vec![Content::from(
//                 "Znajdź i podaj cenę zamknięcia sp500 na dzień 2025-05-02",
//             )])
//             .tool(GoogleSearch::default())
//             .model(Model::Gemini25FlashPreview0520)
//             .generation_config(GenerationConfig {
//                 thinking_config: Some(ThinkingConfig {
//                     include_thoughts: true,
//                     thinking_budget: 0,
//                 }),
//                 ..Default::default()
//             })
//             .build();

//         println!("{}", serde_json::to_string_pretty(&request).unwrap());
//         let response = request.send().await.unwrap();
//         dbg!(&response);

//         Ok(())
//     }
// }
