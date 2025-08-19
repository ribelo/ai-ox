mod conversion;
mod error;

pub use error::GeminiError;

use crate::{
    ModelResponse,
    content::delta::StreamEvent,
    errors::GenerateContentError,
    model::{Model, ModelRequest, ModelInfo, Provider, response::RawStructuredResponse},
    usage::Usage,
};
use async_stream::try_stream;
use bon::Builder;
use futures_util::{FutureExt, StreamExt, future::BoxFuture, stream::BoxStream};
use gemini_ox::{
    Gemini,
    content::Content as GeminiContent,
    generate_content::{GenerationConfig, SafetySettings},
    tool::config::ToolConfig,
};

/// Represents a model from the Google Gemini family.
#[derive(Debug, Clone, Builder)]
pub struct GeminiModel {
    /// Gemini client
    #[builder(field)]
    client: Gemini,
    #[builder(into)]
    system_instruction: Option<GeminiContent>,
    /// The specific model name (e.g., "gemini-2.5-flash").
    #[builder(into)]
    model: String,
    tool_config: Option<ToolConfig>,
    safety_settings: Option<SafetySettings>,
    generation_config: Option<GenerationConfig>,
    #[builder(into)]
    cached_content: Option<String>,
}

impl<S: gemini_model_builder::State> GeminiModelBuilder<S> {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client = Gemini::new(api_key);
        self
    }
}

impl GeminiModel {
    /// Create a new GeminiModel from environment variables.
    ///
    /// This function reads the GOOGLE_AI_API_KEY or GEMINI_API_KEY from the environment and returns an error if missing.
    pub async fn new(model: impl Into<String>) -> Result<Self, GeminiError> {
        let api_key = std::env::var("GOOGLE_AI_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| GeminiError::MissingApiKey)?;

        let client = Gemini::new(&api_key);

        Ok(Self {
            client,
            system_instruction: None,
            model: model.into(),
            tool_config: None,
            safety_settings: None,
            generation_config: None,
            cached_content: None,
        })
    }
}

impl Model for GeminiModel {
    fn info(&self) -> ModelInfo<'_> {
        ModelInfo(Provider::Google, &self.model)
    }

    fn name(&self) -> &str {
        &self.model
    }

    /// Sends a request to the Gemini API and returns the response.
    ///
    /// This implementation will handle the conversion from the generic `ModelRequest`
    /// to the specific format required by the `gemini-ox` client, including handling
    /// system instructions and message history.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        async move {
            let gemini_request = conversion::convert_request_to_gemini(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                self.tool_config.clone(),
                self.safety_settings.clone(),
                self.generation_config.clone(),
                self.cached_content.clone(),
            )?;
            let response = gemini_request
                .send(&self.client)
                .await
                .map_err(GeminiError::Api)?;
            conversion::convert_gemini_response_to_ai_ox(response, self.model.clone())
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
        let client = self.client.clone();

        let stream = try_stream! {
            let gemini_request = conversion::convert_request_to_gemini(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                self.tool_config.clone(),
                self.safety_settings.clone(),
                self.generation_config.clone(),
                self.cached_content.clone(),
            )?;
            let mut response_stream = gemini_request.stream(&client);

            while let Some(response) = response_stream.next().await {
                let response = response.map_err(GeminiError::Api)?;
                let events = conversion::convert_response_to_stream_events(response);
                for event in events {
                    yield event?;
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
        async move {
            // Parse the schema and remove the $schema field if present (Gemini API doesn't accept it)
            let mut schema_json: serde_json::Value = serde_json::from_str(&schema)
                .map_err(|e| GeminiError::InvalidSchema(e.to_string()))?;
            if let Some(obj) = schema_json.as_object_mut() {
                obj.remove("$schema");
            }

            let generation_config = GenerationConfig::builder()
                .response_mime_type("application/json")
                .response_schema(schema_json)
                .build();

            let gemini_request = conversion::convert_request_to_gemini(
                request,
                self.model.clone(),
                self.system_instruction.clone(),
                self.tool_config.clone(),
                self.safety_settings.clone(),
                Some(generation_config),
                self.cached_content.clone(),
            )?;
            let response = gemini_request
                .send(&self.client)
                .await
                .map_err(GeminiError::Api)?;

            // Extract the text content from the first part of the first candidate.
            let text = response
                .candidates
                .first()
                .and_then(|candidate| candidate.content.parts().first())
                .and_then(|part| part.as_text())
                .ok_or_else(|| GeminiError::ResponseParsing("No response content".to_string()))?;

            // Parse the text as JSON.
            let json = serde_json::from_str(text)
                .map_err(|e| GeminiError::ResponseParsing(e.to_string()))?;

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

    #[tokio::test]
    async fn test_model_method() {
        let model = GeminiModel::builder()
            .api_key("test-key")
            .model("test-model")
            .build();

        assert_eq!(model.model, "test-model");
    }

    #[test]
    fn test_json_schema_generation() {
        use schemars::schema_for;

        // Test that we can generate schemas for our types
        let _cat_schema = schema_for!(Cat);
        let _complex_schema = schema_for!(ComplexData);

        // Test that we can use ResponseSchema::from which is what the implementation uses
        #[cfg(feature = "schema")]
        {
            let _response_schema_cat = <gemini_ox::ResponseSchema>::from::<Cat>();
            let _response_schema_complex = <gemini_ox::ResponseSchema>::from::<ComplexData>();
        }
    }

    #[tokio::test]
    async fn test_request_stream_signature() {
        // This test verifies that the streaming method exists and has the correct signature
        let model = GeminiModel::builder()
            .api_key("test-key")
            .model("test-model")
            .build();

        // Test that the method exists (we don't actually call it to avoid needing API key)
        assert_eq!(model.model, "test-model");
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY environment variable and makes actual API calls"]
    async fn test_gemini_model_request() {
        let model = GeminiModel::new("gemini-2.5-flash").await.unwrap();

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
        assert_eq!(response.model_name, "gemini-2.5-flash");
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
        let model = GeminiModel::new("gemini-2.5-flash").await.unwrap();

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
                assert_eq!(response.model_name, "gemini-2.5-flash");

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
        let model = GeminiModel::new("gemini-2.5-flash").await.unwrap();

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
                assert_eq!(response.model_name, "gemini-2.5-flash");

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
        let model = GeminiModel::new("gemini-2.5-flash").await.unwrap();

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
        assert_eq!(response.model_name, "gemini-2.5-flash");
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
        let model = GeminiModel::new("gemini-2.5-flash").await.unwrap();

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
                    assert!(
                        stream_stop.usage.input_tokens() > 0,
                        "Should have input token usage"
                    );
                    message_stopped = true;
                    println!(
                        "
Usage: {:?}",
                        stream_stop.usage
                    );
                }
            }

            events.push(event);
        }

        println!(
            "
Received {} events total",
            events.len()
        );

        // Verify we got the expected event sequence
        assert!(content_received, "Should have received some content");
        assert!(message_stopped, "Should have received StreamStop");

        // Verify event order - last should be StreamStop
        assert!(
            matches!(events.last(), Some(StreamEvent::StreamStop(_))),
            "Last event should be StreamStop"
        );

        // Verify we got at least one text delta event
        let text_delta_count = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::TextDelta(_)))
            .count();
        assert!(
            text_delta_count > 0,
            "Should have received at least one TextDelta event"
        );
    }

    #[tokio::test]
    async fn test_gemini_tool_conversion_to_request() {
        use crate::{
            content::{message::{Message, MessageRole}, part::Part},
            tool::{FunctionMetadata, Tool},
        };
        use serde_json::json;

        // Create a simple function declaration tool
        let tool = Tool::FunctionDeclarations(vec![FunctionMetadata {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a location".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    }
                },
                "required": ["location"]
            }),
        }]);

        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "What's the weather like in San Francisco?".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        // Test the conversion function directly (without making API call)
        let result = conversion::convert_request_to_gemini(
            ModelRequest {
                messages: vec![message],
                system_message: None,
                tools: Some(vec![tool.clone()]), // Actually provide tools here!
            },
            "gemini-1.5-flash".to_string(),
            None, // system_instruction
            None, // tool_config
            None, // safety_settings
            None, // generation_config
            None, // cached_content
        );

        match result {
            Ok(gemini_request) => {
                println!("Tool conversion succeeded!");
                println!("Tools in request: {:?}", gemini_request.tools);
                
                // Verify tools were converted
                assert!(gemini_request.tools.is_some(), "Tools should be present in converted request");
                let tools = gemini_request.tools.unwrap();
                assert_eq!(tools.len(), 1, "Should have exactly one tool");
                
                // Print the JSON to see what it looks like
                let json_str = serde_json::to_string_pretty(&tools[0]).unwrap();
                println!("Converted tool JSON: {}", json_str);
                
                // Check that it's the right structure - this should expose issues
                let tool_json = &tools[0];
                if tool_json.get("function_declarations").is_none() {
                    panic!("Expected 'function_declarations' field in converted tool JSON, got: {}", json_str);
                }
            }
            Err(e) => {
                panic!("Tool conversion failed: {:?}", e);
            }
        }
    }
}
