use bon::Builder;
use std::time::Duration;

use crate::{OpenAIRequestError, ChatRequest, ChatResponse, internal::OpenAIRequestHelper};

/// OpenAI AI API client
#[derive(Debug, Clone, Builder)]
pub struct OpenAI {
    /// API key for authentication
    api_key: String,

    /// Base URL for the API (allows for custom endpoints)
    #[builder(default = "https://api.openai.com/v1".to_string(), into)]
    pub base_url: String,

    /// HTTP client for making requests
    #[builder(skip)]
    client: reqwest::Client,

    /// Rate limiter (optional)
    #[cfg(feature = "leaky-bucket")]
    #[builder(skip)]
    rate_limiter: Option<leaky_bucket::RateLimiter>,
}

impl OpenAI {
    /// Create a new OpenAI client with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            client,
            #[cfg(feature = "leaky-bucket")]
            rate_limiter: None,
        }
    }

    /// Create request helper for internal use
    fn request_helper(&self) -> OpenAIRequestHelper {
        OpenAIRequestHelper::new(self.client.clone(), &self.base_url, &self.api_key)
    }

    /// Create a new OpenAI client from environment variable
    pub fn from_env() -> Result<Self, OpenAIRequestError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| OpenAIRequestError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    /// Create a chat request builder
    pub fn chat(&self) -> crate::request::ChatRequestBuilder {
        ChatRequest::builder()
    }

    /// Send a chat request and get a response
    pub async fn send(&self, request: &ChatRequest) -> Result<ChatResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().send_chat_request(request).await
    }

    /// Send a chat request and get a streaming response
    pub fn stream(
        &self,
        request: &ChatRequest,
    ) -> futures_util::stream::BoxStream<'static, Result<ChatResponse, OpenAIRequestError>> {
        use async_stream::try_stream;
        
        let helper = self.request_helper();
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        #[cfg(feature = "leaky-bucket")]
        let rate_limiter = self.rate_limiter.clone();

        Box::pin(try_stream! {
            #[cfg(feature = "leaky-bucket")]
            if let Some(ref limiter) = rate_limiter {
                limiter.acquire_one().await;
            }

            let mut stream = helper.stream_chat_request(&request_data);
            use futures_util::StreamExt;
            
            while let Some(result) = stream.next().await {
                yield result?;
            }
        })
    }

    /// List available models
    pub async fn list_models(&self) -> Result<crate::response::ModelsResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().list_models().await
    }

    /// Create embeddings
    pub async fn create_embeddings(&self, request: &crate::request::EmbeddingsRequest) -> Result<crate::response::EmbeddingsResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_embeddings(request).await
    }

    /// Moderate content
    pub async fn create_moderation(&self, request: &crate::request::ModerationRequest) -> Result<crate::response::ModerationResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_moderation(request).await
    }

    /// Generate images
    pub async fn create_image(&self, request: &crate::request::ImageRequest) -> Result<crate::response::ImageResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_image(request).await
    }

    /// List files
    pub async fn list_files(&self) -> Result<crate::response::FilesResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().list_files().await
    }

    /// Get file information
    pub async fn retrieve_file(&self, file_id: &str) -> Result<crate::response::FileInfo, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().retrieve_file(file_id).await
    }

    /// Delete file
    pub async fn delete_file(&self, file_id: &str) -> Result<crate::response::FileInfo, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().delete_file(file_id).await
    }

    /// List fine-tuning jobs
    pub async fn list_fine_tuning_jobs(&self) -> Result<crate::response::FineTuningJobsResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().list_fine_tuning_jobs().await
    }

    /// Create fine-tuning job
    pub async fn create_fine_tuning_job(&self, request: &crate::request::FineTuningRequest) -> Result<crate::response::FineTuningJob, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_fine_tuning_job(request).await
    }

    /// Get fine-tuning job
    pub async fn retrieve_fine_tuning_job(&self, job_id: &str) -> Result<crate::response::FineTuningJob, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().retrieve_fine_tuning_job(job_id).await
    }

    /// Cancel fine-tuning job
    pub async fn cancel_fine_tuning_job(&self, job_id: &str) -> Result<crate::response::FineTuningJob, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().cancel_fine_tuning_job(job_id).await
    }

    /// List assistants
    pub async fn list_assistants(&self) -> Result<crate::response::AssistantsResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().list_assistants().await
    }

    /// Create assistant
    pub async fn create_assistant(&self, request: &crate::request::AssistantRequest) -> Result<crate::response::AssistantInfo, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_assistant(request).await
    }

    /// Get assistant
    pub async fn retrieve_assistant(&self, assistant_id: &str) -> Result<crate::response::AssistantInfo, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().retrieve_assistant(assistant_id).await
    }

    /// Update assistant
    pub async fn modify_assistant(&self, assistant_id: &str, request: &crate::request::AssistantRequest) -> Result<crate::response::AssistantInfo, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().modify_assistant(assistant_id, request).await
    }

    /// Delete assistant
    pub async fn delete_assistant(&self, assistant_id: &str) -> Result<serde_json::Value, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().delete_assistant(assistant_id).await
    }

    /// Upload file
    pub async fn upload_file(&self, request: &crate::request::AudioRequest) -> Result<crate::response::FileUploadResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().upload_file(request).await
    }

    /// Transcribe audio
    pub async fn create_transcription(&self, request: &crate::request::AudioRequest) -> Result<crate::response::AudioResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_transcription(request).await
    }

    /// Translate audio
    pub async fn create_translation(&self, request: &crate::request::AudioRequest) -> Result<crate::response::AudioResponse, OpenAIRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire_one().await;
        }

        self.request_helper().create_translation(request).await
    }
}

#[cfg(feature = "leaky-bucket")]
impl OpenAI {
    /// Set rate limiter
    pub fn with_rate_limiter(mut self, rate_limiter: leaky_bucket::RateLimiter) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }
}