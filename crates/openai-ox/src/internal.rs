use ai_ox_common::{
    request_builder::{RequestBuilder, RequestConfig, Endpoint, HttpMethod, AuthMethod, MultipartForm},
    error::ProviderError, BoxStream
};
use futures_util::stream::BoxStream as FuturesBoxStream;
use crate::{OpenAIRequestError, ChatRequest, ChatResponse, ResponsesRequest, ResponsesResponse, ResponsesStreamChunk};

/// OpenAI client helper methods using the common RequestBuilder
pub struct OpenAIRequestHelper {
    request_builder: RequestBuilder,
}

impl OpenAIRequestHelper {
    pub fn new(client: reqwest::Client, base_url: &str, api_key: &str) -> Self {
        let config = RequestConfig::new(base_url)
            .with_auth(AuthMethod::Bearer(api_key.to_string()));

        let request_builder = RequestBuilder::new(client, config);

        Self { request_builder }
    }

    /// Send a chat completion request
    pub async fn send_chat_request(&self, request: &ChatRequest) -> Result<ChatResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("chat/completions", HttpMethod::Post);
        
        Ok(self.request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Stream a chat completion request
    pub fn stream_chat_request(
        &self, 
        request: &ChatRequest
    ) -> FuturesBoxStream<'static, Result<ChatResponse, OpenAIRequestError>> {
        let endpoint = Endpoint::new("chat/completions", HttpMethod::Post);
        
        // Use the common streaming implementation (no conversion needed - same type)
        let stream: BoxStream<'static, Result<ChatResponse, ProviderError>> = 
            self.request_builder.stream(&endpoint, Some(request));
        
        // Direct cast since OpenAIRequestError = ProviderError
        stream
    }

    /// List available models
    pub async fn list_models(&self) -> Result<crate::response::ModelsResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("models", HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Generate embeddings
    pub async fn create_embeddings(&self, request: &crate::request::EmbeddingsRequest) -> Result<crate::response::EmbeddingsResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("embeddings", HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, Some(request)).await?)
    }

    /// Moderate content
    pub async fn create_moderation(&self, request: &crate::request::ModerationRequest) -> Result<crate::response::ModerationResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("moderations", HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, Some(request)).await?)
    }

    /// Generate images
    pub async fn create_image(&self, request: &crate::request::ImageRequest) -> Result<crate::response::ImageResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("images/generations", HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, Some(request)).await?)
    }

    /// List files
    pub async fn list_files(&self) -> Result<crate::response::FilesResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("files", HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Get file information
    pub async fn retrieve_file(&self, file_id: &str) -> Result<crate::response::FileInfo, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("files/{}", file_id), HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Delete file
    pub async fn delete_file(&self, file_id: &str) -> Result<crate::response::FileInfo, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("files/{}", file_id), HttpMethod::Delete);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// List fine-tuning jobs
    pub async fn list_fine_tuning_jobs(&self) -> Result<crate::response::FineTuningJobsResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("fine_tuning/jobs", HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Create fine-tuning job
    pub async fn create_fine_tuning_job(&self, request: &crate::request::FineTuningRequest) -> Result<crate::response::FineTuningJob, OpenAIRequestError> {
        let endpoint = Endpoint::new("fine_tuning/jobs", HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, Some(request)).await?)
    }

    /// Get fine-tuning job
    pub async fn retrieve_fine_tuning_job(&self, job_id: &str) -> Result<crate::response::FineTuningJob, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("fine_tuning/jobs/{}", job_id), HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Cancel fine-tuning job
    pub async fn cancel_fine_tuning_job(&self, job_id: &str) -> Result<crate::response::FineTuningJob, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("fine_tuning/jobs/{}/cancel", job_id), HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// List assistants
    pub async fn list_assistants(&self) -> Result<crate::response::AssistantsResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("assistants", HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Create assistant
    pub async fn create_assistant(&self, request: &crate::request::AssistantRequest) -> Result<crate::response::AssistantInfo, OpenAIRequestError> {
        let endpoint = Endpoint::new("assistants", HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, Some(request)).await?)
    }

    /// Get assistant
    pub async fn retrieve_assistant(&self, assistant_id: &str) -> Result<crate::response::AssistantInfo, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("assistants/{}", assistant_id), HttpMethod::Get);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Update assistant
    pub async fn modify_assistant(&self, assistant_id: &str, request: &crate::request::AssistantRequest) -> Result<crate::response::AssistantInfo, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("assistants/{}", assistant_id), HttpMethod::Post);
        Ok(self.request_builder.request_json(&endpoint, Some(request)).await?)
    }

    /// Delete assistant
    pub async fn delete_assistant(&self, assistant_id: &str) -> Result<serde_json::Value, OpenAIRequestError> {
        let endpoint = Endpoint::new(format!("assistants/{}", assistant_id), HttpMethod::Delete);
        Ok(self.request_builder.request_json(&endpoint, None::<&()>).await?)
    }

    /// Upload file
    pub async fn upload_file(&self, request: &crate::request::AudioRequest) -> Result<crate::response::FileUploadResponse, OpenAIRequestError> {
        let form = MultipartForm::new()
            .file_from_bytes("file", &request.filename, request.file.clone())
            .text("model", &request.model);

        let form = if let Some(ref language) = request.language {
            form.text("language", language)
        } else {
            form
        };

        let form = if let Some(ref prompt) = request.prompt {
            form.text("prompt", prompt) 
        } else {
            form
        };

        let form = if let Some(ref format) = request.response_format {
            form.text("response_format", format)
        } else {
            form
        };

        let form = if let Some(temp) = request.temperature {
            form.text("temperature", temp.to_string())
        } else {
            form
        };

        let endpoint = Endpoint::new("files", HttpMethod::Post);
        Ok(self.request_builder.request_multipart(&endpoint, form.build()).await?)
    }

    /// Transcribe audio
    pub async fn create_transcription(&self, request: &crate::request::AudioRequest) -> Result<crate::response::AudioResponse, OpenAIRequestError> {
        let form = MultipartForm::new()
            .file_from_bytes("file", &request.filename, request.file.clone())
            .text("model", &request.model);

        let form = if let Some(ref language) = request.language {
            form.text("language", language)
        } else {
            form
        };

        let form = if let Some(ref prompt) = request.prompt {
            form.text("prompt", prompt)
        } else {
            form
        };

        let form = if let Some(ref format) = request.response_format {
            form.text("response_format", format)
        } else {
            form
        };

        let form = if let Some(temp) = request.temperature {
            form.text("temperature", temp.to_string())
        } else {
            form
        };

        let endpoint = Endpoint::new("audio/transcriptions", HttpMethod::Post);
        Ok(self.request_builder.request_multipart(&endpoint, form.build()).await?)
    }

    /// Translate audio
    pub async fn create_translation(&self, request: &crate::request::AudioRequest) -> Result<crate::response::AudioResponse, OpenAIRequestError> {
        let form = MultipartForm::new()
            .file_from_bytes("file", &request.filename, request.file.clone())
            .text("model", &request.model);

        let form = if let Some(ref prompt) = request.prompt {
            form.text("prompt", prompt)
        } else {
            form
        };

        let form = if let Some(ref format) = request.response_format {
            form.text("response_format", format)
        } else {
            form
        };

        let form = if let Some(temp) = request.temperature {
            form.text("temperature", temp.to_string())
        } else {
            form
        };

        let endpoint = Endpoint::new("audio/translations", HttpMethod::Post);
        Ok(self.request_builder.request_multipart(&endpoint, form.build()).await?)
    }

    /// Send a Responses API request
    pub async fn send_responses_request(&self, request: &ResponsesRequest) -> Result<ResponsesResponse, OpenAIRequestError> {
        let endpoint = Endpoint::new("responses", HttpMethod::Post);
        
        Ok(self.request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Stream a Responses API request
    pub fn stream_responses_request(
        &self, 
        request: &ResponsesRequest
    ) -> FuturesBoxStream<'static, Result<ResponsesStreamChunk, OpenAIRequestError>> {
        let endpoint = Endpoint::new("responses", HttpMethod::Post);
        
        // Use the common streaming implementation
        let stream: BoxStream<'static, Result<ResponsesStreamChunk, ProviderError>> = 
            self.request_builder.stream(&endpoint, Some(request));
        
        // Direct cast since OpenAIRequestError = ProviderError
        stream
    }
}