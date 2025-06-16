use crate::{
    content::{delta::MessageStreamEvent, message::{Message, MessageRole}, response::GenerateContentResponse, part::{ImageSource, Part, FileData}},
    errors::GenerateContentError,
    model::Model,
    usage::Usage,
};
use bon::Builder;
use futures_util::stream::BoxStream;
use gemini_ox::{
    Gemini,
    content::Content as GeminiContent,
    generate_content::{
        GenerationConfig, SafetySettings,
        request::GenerateContentRequest as GeminiGenerateContentRequest,
        response::GenerateContentResponse as GeminiResponse,
    },
    tool::config::ToolConfig,
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
                    _ => {
                        // Skip other part types we don't handle yet
                        continue;
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
        let mut content_list = Vec::new();

        // Convert conversation messages
        for message in messages {
            let content: GeminiContent = message.into().into();
            content_list.push(content);
        }

        let gemini_request = GeminiGenerateContentRequest {
            contents: content_list,
            tools: None, // TODO: Convert tools when tool support is added
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
        unimplemented!("GeminiModel::request_stream is not yet implemented.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{
        message::{Message, MessageRole},
        part::Part,
    };

    #[tokio::test]
    #[ignore = "This test makes a real API call and requires a GEMINI_API_KEY"]
    async fn test_gemini_model_request_success() {
        let model = GeminiModel::builder()
            .api_key(std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set"))
            .model("gemini-1.5-flash-latest".to_string())
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
        assert!(!response.model_name.is_empty());
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

    // TODO: Add integration test for response conversion once gemini-ox types are properly exposed
}
