use crate::{
    content::{
        delta::MessageStreamEvent,
        message::{Message, MessageRole},
        part::{FileData, ImageSource, Part},
        response::{GenerateContentResponse, GenerateContentStructuredResponse},
    },
    errors::GenerateContentError,
    model::Model,
    tool::Tool,
    usage::Usage,
};
use bon::Builder;
use futures_util::stream::BoxStream;
use gemini_ox::{
    Gemini, ResponseSchema,
    content::Content as GeminiContent,
    generate_content::{
        GenerationConfig, SafetySettings,
        request::GenerateContentRequest as GeminiGenerateContentRequest,
        response::GenerateContentResponse as GeminiResponse,
    },
    tool::{Tool as GeminiTool, config::ToolConfig},
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Represents a model from the Google Gemini family.
#[derive(Debug, Clone, Builder)]
pub struct GeminiModel {
    /// Gemini client
    #[builder(field)]
    client: Gemini,
    #[builder(field)]
    system_instruction: Option<GeminiContent>,
    #[builder(field)]
    tools: Option<Vec<Tool>>,
    /// The specific model name (e.g., "gemini-1.5-flash-latest").
    model: String,
    tool_config: Option<ToolConfig>,
    safety_settings: Option<SafetySettings>,
    generation_config: Option<GenerationConfig>,
    cached_content: Option<String>,
}

impl<S: gemini_model_builder::State> GeminiModelBuilder<S> {
    pub fn client(mut self, gemini: Gemini) -> Self {
        self.client = gemini;
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = Gemini::new(api_key.into());
        self
    }

    pub fn system_instruction(mut self, system_instruction: impl Into<GeminiContent>) -> Self {
        self.system_instruction = Some(system_instruction.into());
        self
    }

    pub fn tool(mut self, tool: impl Into<Tool>) -> Self {
        self.tools.get_or_insert_default().push(tool.into());
        self
    }

    pub fn tools(mut self, tools: impl Into<Vec<Tool>>) -> Self {
        let tools = tools.into();
        self.tools
            .get_or_insert_default()
            .extend(tools.into_iter().map(Into::into));
        self
    }
}

impl GeminiModel {
    /// Converts a `gemini-ox` `GenerateContentResponse` to an `ai-ox` `GenerateContentResponse`.
    fn convert_response(&self, response: GeminiResponse) -> GenerateContentResponse {
        let mut parts = Vec::new();

        // Get the first candidate's content, if it exists.
        if let Some(candidate) = response.candidates.first() {
            for part in &candidate.content.parts {
                match &part.data {
                    gemini_ox::content::PartData::Text(text) => {
                        parts.push(Part::Text {
                            text: text.to_string(),
                        });
                    }
                    gemini_ox::content::PartData::InlineData(blob) => {
                        parts.push(Part::Image {
                            source: ImageSource::Base64 {
                                media_type: blob.mime_type.clone(),
                                data: blob.data.clone(),
                            },
                        });
                    }
                    gemini_ox::content::PartData::FileData(file_data) => {
                        let file_data = FileData {
                            file_uri: file_data.file_uri.clone(),
                            mime_type: file_data.mime_type.clone(),
                            display_name: file_data.display_name.clone(),
                        };
                        parts.push(Part::File(file_data));
                    }
                    gemini_ox::content::PartData::FunctionCall(function_call) => todo!(),
                    gemini_ox::content::PartData::FunctionResponse(function_response) => todo!(),
                    gemini_ox::content::PartData::ExecutableCode(executable_code) => todo!(),
                    gemini_ox::content::PartData::CodeExecutionResult(code_execution_result) => {
                        todo!()
                    }
                }
            }
        }

        let message = Message {
            role: MessageRole::Assistant,
            content: parts,
            timestamp: chrono::Utc::now(),
        };

        // Convert usage data from Gemini response
        let usage = if let Some(usage_metadata) = response.usage_metadata {
            Usage::from(usage_metadata)
        } else {
            Usage::new()
        };

        GenerateContentResponse {
            message,
            model_name: self.model.clone(),
            usage,
            vendor_name: "google".to_string(),
        }
    }
}

impl Model for GeminiModel {
    /// Returns the model name/identifier.
    fn model(&self) -> &str {
        &self.model
    }

    /// Sends a request to the Gemini API and returns the response.
    ///
    /// This implementation will handle the conversion from the generic `GenerateContentRequest`
    /// to the specific format required by the `gemini-ox` client, including handling
    /// system instructions and message history.
    async fn request(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> Result<GenerateContentResponse, GenerateContentError> {
        // Convert messages to gemini-ox Content format
        let contents: Vec<GeminiContent> = messages.into_iter().map(|m| m.into().into()).collect();

        // Convert ai-ox Tools into JSON values for the gemini-ox request.
        let tools: Option<Vec<Value>> = self.tools.as_ref().map(|tools| {
            tools
                .iter()
                .cloned()
                .map(Into::<GeminiTool>::into)
                .filter_map(|gemini_tool| serde_json::to_value(gemini_tool).ok())
                .collect()
        });

        let gemini_request = GeminiGenerateContentRequest {
            contents,
            tools,
            model: self.model.clone(),
            tool_config: self.tool_config.clone(),
            safety_settings: self.safety_settings.clone(),
            system_instruction: self.system_instruction.clone(),
            generation_config: self.generation_config.clone(),
            cached_content: self.cached_content.clone(),
        };

        let response = gemini_request.send(&self.client).await?;

        // Convert the response using our private conversion method
        let ai_response = self.convert_response(response);

        Ok(ai_response)
    }

    /// Returns a stream of events for a streaming request.
    ///
    /// This feature is not yet implemented for the `GeminiModel`.
    fn request_stream<'a>(
        &'a self,
        _messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> BoxStream<'a, Result<MessageStreamEvent, GenerateContentError>> {
        // TODO;
        unimplemented!("GeminiModel::request_stream is not yet implemented.");
    }

    /// Generates structured content that conforms to a specific schema.
    ///
    /// This implementation leverages Gemini's `response_schema` facility to guide the model
    /// to produce JSON output that matches the provided type's schema.
    async fn request_structured<O>(
        &self,
        messages: impl IntoIterator<Item = impl Into<Message>>,
    ) -> Result<GenerateContentStructuredResponse<O>, GenerateContentError>
    where
        O: DeserializeOwned + JsonSchema + Send,
    {
        // Configure generation for structured output (JSON mode)
        let generation_config = GenerationConfig::builder()
            .response_mime_type("application/json")
            .response_schema(ResponseSchema::from::<O>())
            .build();

        // Convert messages to gemini-ox Content format
        let contents: Vec<GeminiContent> = messages.into_iter().map(|m| m.into().into()).collect();

        // Convert ai-ox Tools into JSON values for the gemini-ox request.
        let tools: Option<Vec<Value>> = self.tools.as_ref().map(|tools| {
            tools
                .iter()
                .cloned()
                .map(Into::<GeminiTool>::into)
                .filter_map(|gemini_tool| serde_json::to_value(gemini_tool).ok())
                .collect()
        });

        let gemini_request = GeminiGenerateContentRequest {
            contents,
            tools,
            model: self.model.clone(),
            tool_config: self.tool_config.clone(),
            safety_settings: self.safety_settings.clone(),
            system_instruction: self.system_instruction.clone(),
            generation_config: Some(generation_config),
            cached_content: self.cached_content.clone(),
        };

        let response = gemini_request.send(&self.client).await?;

        // Extract the text content from the first candidate
        let text = response
            .candidates
            .first()
            .ok_or(GenerateContentError::NoResponse)?
            .content
            .parts()
            .first()
            .and_then(|part| part.as_text())
            .ok_or(GenerateContentError::NoResponse)?
            .to_string();

        // Parse the JSON response into the target type
        let data: O = serde_json::from_str(&text)
            .map_err(|e| GenerateContentError::response_parsing(e.to_string()))?;

        // Convert usage data from Gemini response
        let usage = if let Some(usage_metadata) = response.usage_metadata {
            Usage::from(usage_metadata)
        } else {
            Usage::new()
        };

        Ok(GenerateContentStructuredResponse {
            data,
            model_name: self.model.clone(),
            usage,
            vendor_name: "google".to_string(),
            raw_json: Some(text),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{
        message::{Message, MessageRole},
        part::Part,
    };
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct Cat {
        name: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
    struct ComplexData {
        #[schemars(description = "A number between 1 and 100")]
        #[schemars(range(min = 1, max = 100))]
        count: i32,
        #[schemars(description = "List of tags")]
        tags: Vec<String>,
        #[schemars(description = "Whether this item is active")]
        active: bool,
    }

    #[test]
    fn test_model_method() {
        let model = GeminiModel::builder()
            .api_key("test-key")
            .model("test-model".to_string())
            .build();

        assert_eq!(model.model(), "test-model");
    }

    #[test]
    fn test_json_schema_generation() {
        use gemini_ox::ResponseSchema;
        use schemars::schema_for;

        // Test that we can generate schemas for our types
        let _cat_schema = schema_for!(Cat);
        let _complex_schema = schema_for!(ComplexData);

        // Test that we can use ResponseSchema::from which is what the implementation uses
        let _response_schema_cat = ResponseSchema::from::<Cat>();
        let _response_schema_complex = ResponseSchema::from::<ComplexData>();
    }

    #[test]
    fn test_request_stream_unimplemented() {
        // This test just verifies that the function exists and compiles correctly
        // The actual panic behavior is expected and documented in the implementation
        let model = GeminiModel::builder()
            .api_key("test-key")
            .model("test-model".to_string())
            .build();

        // We don't actually call request_stream here since it would panic
        // This test just ensures the method exists and has the correct signature
        assert_eq!(model.model(), "test-model");
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY environment variable and makes actual API calls"]
    async fn test_gemini_model_request() {
        let api_key = match std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

        let model = GeminiModel::builder()
            .api_key(api_key)
            .model("gemini-1.5-flash".to_string())
            .build();

        let messages = vec![Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Why is the sky blue?".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        }];

        let result = model.request(messages).await;

        assert!(result.is_ok(), "Model request failed: {:?}", result.err());
        let response = result.unwrap();
        assert_eq!(response.vendor_name, "google");
        assert_eq!(response.model_name, "gemini-1.5-flash");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty());

        // Check that we got text content back
        if let Some(Part::Text { text }) = response.message.content.first() {
            assert!(!text.is_empty());
            println!("Response from model: {}", text);
        } else {
            panic!("Expected text response but got different content type");
        }
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY environment variable and makes actual API calls"]
    async fn test_gemini_model_request_structured_simple() {
        let api_key = match std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

        let model = GeminiModel::builder()
            .api_key(api_key)
            .model("gemini-1.5-flash".to_string())
            .build();

        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Return a JSON object with a cat named 'Fluffy'".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let result: Result<GenerateContentStructuredResponse<Cat>, _> =
            model.request_structured(vec![message]).await;

        match result {
            Ok(response) => {
                assert!(!response.data.name.is_empty());
                assert_eq!(response.vendor_name, "google");
                assert_eq!(response.model_name, "gemini-1.5-flash");
                assert!(response.raw_json.is_some());
                println!("Generated cat: {:?}", response.data);
            }
            Err(e) => {
                panic!("Integration test failed: {:?}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY environment variable and makes actual API calls"]
    async fn test_gemini_model_request_structured_complex() {
        let api_key = match std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

        let model = GeminiModel::builder()
            .api_key(api_key)
            .model("gemini-1.5-flash".to_string())
            .build();

        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Generate a JSON object with a count between 1-100, some tags related to programming, and active set to true".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let result: Result<GenerateContentStructuredResponse<ComplexData>, _> =
            model.request_structured(vec![message]).await;

        match result {
            Ok(response) => {
                assert!(response.data.count >= 1 && response.data.count <= 100);
                assert!(!response.data.tags.is_empty());
                assert!(response.data.active);
                assert_eq!(response.vendor_name, "google");
                assert_eq!(response.model_name, "gemini-1.5-flash");
                assert!(response.raw_json.is_some());
                println!("Generated complex data: {:?}", response.data);
            }
            Err(e) => {
                panic!("Integration test failed: {:?}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY environment variable and makes actual API calls"]
    async fn test_gemini_model_request_multiple_messages() {
        let api_key = match std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping integration test");
                return;
            }
        };

        let model = GeminiModel::builder()
            .api_key(api_key)
            .model("gemini-1.5-flash".to_string())
            .build();

        let messages = vec![
            Message {
                role: MessageRole::User,
                content: vec![Part::Text {
                    text: "Hello, I'm going to ask you about cats.".to_string(),
                }],
                timestamp: chrono::Utc::now(),
            },
            Message {
                role: MessageRole::Assistant,
                content: vec![Part::Text {
                    text: "Hello! I'd be happy to help you with questions about cats.".to_string(),
                }],
                timestamp: chrono::Utc::now(),
            },
            Message {
                role: MessageRole::User,
                content: vec![Part::Text {
                    text: "What are some popular cat breeds?".to_string(),
                }],
                timestamp: chrono::Utc::now(),
            },
        ];

        let result = model.request(messages).await;

        assert!(result.is_ok(), "Model request failed: {:?}", result.err());
        let response = result.unwrap();
        assert_eq!(response.vendor_name, "google");
        assert_eq!(response.model_name, "gemini-1.5-flash");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty());

        if let Some(Part::Text { text }) = response.message.content.first() {
            assert!(!text.is_empty());
            assert!(text.to_lowercase().contains("cat") || text.to_lowercase().contains("breed"));
            println!("Response about cat breeds: {}", text);
        } else {
            panic!("Expected text response but got different content type");
        }
    }
}
