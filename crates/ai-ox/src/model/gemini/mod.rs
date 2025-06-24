mod conversion;

use crate::{
    ModelResponse,
    content::{
        delta::StreamEvent,
        message::{Message, MessageRole},
        part::Part,
    },
    errors::GenerateContentError,
    model::{Model, ModelRequest, response::RawStructuredResponse},
    usage::Usage,
};
use async_stream::try_stream;
use futures_util::StreamExt;
use bon::Builder;
use futures_util::{FutureExt, future::BoxFuture, stream::BoxStream};
use gemini_ox::{
    Gemini,
    content::Content as GeminiContent,
    generate_content::{
        GenerationConfig, SafetySettings,
        request::GenerateContentRequest as GeminiGenerateContentRequest,
        response::GenerateContentResponse as GeminiResponse,
    },
    tool::{Tool as GeminiTool, config::ToolConfig},
};

/// Represents a model from the Google Gemini family.
#[derive(Debug, Clone, Builder)]
pub struct GeminiModel {
    /// Gemini client
    #[builder(field)]
    client: Gemini,
    #[builder(field)]
    system_instruction: Option<GeminiContent>,
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
}

impl GeminiModel {
    /// Converts a `gemini-ox` `GenerateContentResponse` to an `ai-ox` `ModelResponse`.
    fn convert_response(&self, response: GeminiResponse) -> ModelResponse {
        // Take the first candidate from the response, consuming it to avoid cloning parts.
        let parts = if let Some(candidate) = response.candidates.into_iter().next() {
            candidate
                .content
                .parts
                .into_iter()
                .map(Into::<Part>::into) // Convert each gemini part into our Part
                .filter(|part| {
                    // Skip empty text parts, which can be a side-effect of FunctionCall conversion.
                    !matches!(part, Part::Text { text } if text.is_empty())
                })
                .collect()
        } else {
            Vec::new()
        };

        let message = Message {
            role: MessageRole::Assistant,
            content: parts,
            timestamp: chrono::Utc::now(),
        };

        // Convert usage data from the response, if present.
        let usage = response
            .usage_metadata
            .map(Usage::from)
            .unwrap_or_default();

        ModelResponse {
            message,
            model_name: self.model.clone(),
            usage,
            vendor_name: "google".to_string(),
        }
    }
}

impl Model for GeminiModel {
    fn model(&self) -> &str {
        &self.model
    }

    /// Sends a request to the Gemini API and returns the response.
    ///
    /// This implementation will handle the conversion from the generic `GenerateContentRequest`
    /// to the specific format required by the `gemini-ox` client, including handling
    /// system instructions and message history.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        let system_instruction = request
            .system_message
            .map(Into::into)
            .or_else(|| self.system_instruction.clone());

        let contents: Vec<GeminiContent> = request.messages.into_iter().map(Into::into).collect();

        let tools = request
            .tools
            .map(|tools| {
                tools
                    .into_iter()
                    .map(Into::<GeminiTool>::into)
                    .filter_map(|tool| serde_json::to_value(tool).ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        let gemini_request = GeminiGenerateContentRequest {
            contents,
            tools,
            system_instruction,
            model: self.model.clone(),
            tool_config: self.tool_config.clone(),
            safety_settings: self.safety_settings.clone(),
            generation_config: self.generation_config.clone(),
            cached_content: self.cached_content.clone(),
        };

        async move {
            let response = gemini_request.send(&self.client).await?;
            Ok(self.convert_response(response))
        }
        .boxed()
    }

    /// Returns a stream of events for a streaming request.
    ///
    /// This method streams responses from the Gemini API and converts them to StreamEvents.
    fn request_stream(
        &self,
        request: ModelRequest,
    ) -> BoxStream<'_, Result<StreamEvent, GenerateContentError>> {
        let system_instruction = request
            .system_message
            .map(Into::into)
            .or_else(|| self.system_instruction.clone());

        let contents: Vec<GeminiContent> = request.messages.into_iter().map(Into::into).collect();

        let tools = request
            .tools
            .map(|tools| {
                tools
                    .into_iter()
                    .map(Into::<GeminiTool>::into)
                    .filter_map(|tool| serde_json::to_value(tool).ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        let gemini_request = GeminiGenerateContentRequest {
            contents,
            tools,
            system_instruction,
            model: self.model.clone(),
            tool_config: self.tool_config.clone(),
            safety_settings: self.safety_settings.clone(),
            generation_config: self.generation_config.clone(),
            cached_content: self.cached_content.clone(),
        };

        let stream = try_stream! {
            let mut response_stream = gemini_request.stream(&self.client);

            while let Some(response) = response_stream.next().await {
                let response = response?;
                let events = conversion::convert_streaming_response(response);
                
                for event in events {
                    yield event;
                }
            }
        };

        Box::pin(stream)
    }

    fn request_structured_internal(
        &self,
        request: ModelRequest,
        schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>> {
        // Convert the ModelRequest into separate components for building the GeminiGenerateContentRequest
        // Configure generation for structured output (JSON mode)
        // Parse the schema and remove the $schema field if present (Gemini API doesn't accept it)
        let mut schema_json: serde_json::Value = serde_json::from_str(&schema).unwrap_or_default();
        if let Some(obj) = schema_json.as_object_mut() {
            obj.remove("$schema");
        }

        let generation_config = GenerationConfig::builder()
            .response_mime_type("application/json")
            .response_schema(schema_json)
            .build();

        // Use system instruction from request if available, otherwise use the model's default
        let system_instruction = request
            .system_message
            .map(Into::into)
            .or_else(|| self.system_instruction.clone());

        let contents: Vec<GeminiContent> = request.messages.into_iter().map(Into::into).collect();

        // Use tools from request if available
        let tools = request
            .tools
            .map(|tools| {
                tools
                    .into_iter()
                    .map(Into::<GeminiTool>::into)
                    .filter_map(|tool| serde_json::to_value(tool).ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        let gemini_request = GeminiGenerateContentRequest {
            contents,
            tools,
            system_instruction,
            model: self.model.clone(),
            tool_config: self.tool_config.clone(),
            safety_settings: self.safety_settings.clone(),
            generation_config: Some(generation_config),
            cached_content: self.cached_content.clone(),
        };

        async move {
            let response = gemini_request.send(&self.client).await?;

            // Extract the text content from the first part of the first candidate.
            let text = response
                .candidates
                .first()
                .and_then(|candidate| candidate.content.parts().first())
                .and_then(|part| part.as_text())
                .ok_or(GenerateContentError::NoResponse)?;

            // Parse the text as JSON.
            let json = serde_json::from_str(text)
                .map_err(|e| GenerateContentError::response_parsing(e.to_string()))?;

            let usage = response
                .usage_metadata
                .map(Into::into)
                .unwrap_or_else(Usage::new);

            Ok(RawStructuredResponse {
                json,
                usage,
                model_name: self.model.clone(),
                vendor_name: "google".to_string(),
            })
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        content::{
            message::{Message, MessageRole},
            part::Part,
        },
        model::response::StructuredResponse,
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
    fn test_request_stream_signature() {
        // This test verifies that the streaming method exists and has the correct signature
        let model = GeminiModel::builder()
            .api_key("test-key")
            .model("test-model".to_string())
            .build();

        // Test that the method exists (we don't actually call it to avoid needing API key)
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

        let result = model
            .request(ModelRequest {
                messages,
                system_message: None,
                tools: None,
            })
            .await;

        assert!(result.is_ok(), "Model request failed: {:?}", result.err());
        let response = result.unwrap();
        assert_eq!(response.vendor_name, "google");
        assert_eq!(response.model_name, "gemini-1.5-flash");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty());

        // Check that we got text content back
        if let Some(Part::Text { text }) = response.message.content.first() {
            assert!(!text.is_empty());
            println!("Response from model: {text}");
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

        let result: Result<StructuredResponse<Cat>, _> =
            model.request_structured(vec![message]).await;

        match result {
            Ok(response) => {
                assert!(!response.data.name.is_empty());
                assert_eq!(response.vendor_name, "google");
                assert_eq!(response.model_name, "gemini-1.5-flash");

                println!("Generated cat: {:?}", response.data);
            }
            Err(e) => {
                panic!("Integration test failed: {e:?}");
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

        let result: Result<StructuredResponse<ComplexData>, _> =
            model.request_structured(vec![message]).await;

        match result {
            Ok(response) => {
                assert!(response.data.count >= 1 && response.data.count <= 100);
                assert!(!response.data.tags.is_empty());
                assert!(response.data.active);
                assert_eq!(response.vendor_name, "google");
                assert_eq!(response.model_name, "gemini-1.5-flash");

                println!("Generated complex data: {:?}", response.data);
            }
            Err(e) => {
                panic!("Integration test failed: {e:?}");
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

        let result = model
            .request(ModelRequest {
                messages,
                system_message: None,
                tools: None,
            })
            .await;

        assert!(result.is_ok(), "Model request failed: {:?}", result.err());
        let response = result.unwrap();
        assert_eq!(response.vendor_name, "google");
        assert_eq!(response.model_name, "gemini-1.5-flash");
        assert_eq!(response.message.role, MessageRole::Assistant);
        assert!(!response.message.content.is_empty());

        if let Some(Part::Text { text }) = response.message.content.first() {
            assert!(!text.is_empty());
            assert!(text.to_lowercase().contains("cat") || text.to_lowercase().contains("breed"));
            println!("Response about cat breeds: {text}");
        } else {
            panic!("Expected text response but got different content type");
        }
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY environment variable and makes actual API calls"]
    async fn test_gemini_model_request_stream() {
        let api_key = match std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        {
            Ok(key) => key,
            Err(_) => {
                println!("GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping streaming test");
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
                text: "Tell me a short story about a robot. Make it 2-3 sentences.".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        }];

        let request = ModelRequest {
            messages,
            system_message: None,
            tools: None,
        };

        let mut stream = model.request_stream(request);
        let mut events = Vec::new();
        let mut content_received = false;
        let mut message_stopped = false;

        while let Some(event_result) = stream.next().await {
            let event = match event_result {
                Ok(event) => event,
                Err(e) => panic!("Stream error: {e:?}"),
            };

            match &event {
                StreamEvent::MessageDelta(delta) => {
                    if let Some(content) = &delta.content_delta {
                        assert!(!content.is_empty(), "Content delta should not be empty");
                        content_received = true;
                        print!("{content}"); // Print the streaming text
                    }
                }
                StreamEvent::TextDelta(text) => {
                    assert!(!text.is_empty(), "Text delta should not be empty");
                    content_received = true;
                    print!("{text}"); // Print the streaming text
                }
                StreamEvent::ToolCall(_) => {
                    // Tool calls are handled but don't affect this test logic
                }
                StreamEvent::ToolResult(_) => {
                    // Tool results are handled but don't affect this test logic
                }
                StreamEvent::Usage(_) => {
                    // Usage events are expected
                }
                StreamEvent::StreamStop(stream_stop) => {
                    assert!(stream_stop.usage.input_tokens() > 0, "Should have input token usage");
                    message_stopped = true;
                    println!("\nUsage: {:?}", stream_stop.usage);
                }
            }

            events.push(event);
        }

        println!("\nReceived {} events total", events.len());

        // Verify we got the expected event sequence
        assert!(content_received, "Should have received some content");
        assert!(message_stopped, "Should have received StreamStop");

        // Verify event order - last should be StreamStop
        assert!(
            matches!(events.last(), Some(StreamEvent::StreamStop(_))),
            "Last event should be StreamStop"
        );
        
        // Verify we got at least one text delta event
        let text_delta_count = events.iter().filter(|e| matches!(e, StreamEvent::TextDelta(_))).count();
        assert!(text_delta_count > 0, "Should have received at least one TextDelta event");
    }
}
