#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items
)]

pub mod audio;
pub mod content;
pub mod error;
mod internal;
pub mod message;
pub mod model;
pub mod request;
pub mod response;
pub mod tool;
pub mod usage;

// Re-export main types
pub use audio::{TranscriptionRequest, TranscriptionResponse};
pub use error::MistralRequestError;
pub use model::Model;

// Re-export request types
pub use request::{
    ChatRequest, EmbeddingsRequest, ModerationRequest, ChatModerationRequest,
    FineTuningRequest, BatchJobRequest, FimRequest, AgentsRequest,
    EmbeddingInput, ModerationInput, TrainingFile, FineTuningHyperparameters
};

// Re-export response types  
pub use response::{
    ChatResponse, ChatCompletionChunk, ModelsResponse, EmbeddingsResponse,
    ModerationResponse, FineTuningJobsResponse, FineTuningJob, BatchJobsResponse,
    BatchJob, FilesResponse, FileInfo, FileUploadResponse, FileDeleteResponse,
    ModelInfo, EmbeddingData, ModerationResult, BatchRequestCounts
};

use bon::Builder;
use core::fmt;
use futures_util::stream::BoxStream;
#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
#[cfg(feature = "leaky-bucket")]
use std::sync::Arc;

use crate::internal::MistralRequestHelper;

const BASE_URL: &str = "https://api.mistral.ai";

#[derive(Clone, Default, Builder)]
pub struct Mistral {
    #[builder(into)]
    pub(crate) api_key: String,
    #[builder(default)]
    pub(crate) client: reqwest::Client,
    #[cfg(feature = "leaky-bucket")]
    pub(crate) leaky_bucket: Option<Arc<RateLimiter>>,
    #[builder(default = BASE_URL.to_string(), into)]
    pub(crate) base_url: String,
}

impl Mistral {
    /// Create a new Mistral client with the provided API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket: None,
            base_url: BASE_URL.to_string(),
        }
    }

    pub fn load_from_env() -> Result<Self, std::env::VarError> {
        let api_key = std::env::var("MISTRAL_API_KEY")?;
        Ok(Self::builder().api_key(api_key).build())
    }

    /// Create request helper for internal use
    fn request_helper(&self) -> MistralRequestHelper {
        MistralRequestHelper::new(self.client.clone(), &self.base_url, &self.api_key)
    }
}

impl Mistral {
    pub async fn send(
        &self,
        request: &request::ChatRequest,
    ) -> Result<response::ChatResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().send_chat_request(request).await
    }

    pub fn stream(
        &self,
        request: &request::ChatRequest,
    ) -> BoxStream<'static, Result<response::ChatCompletionChunk, MistralRequestError>> {
        use async_stream::try_stream;
        
        let helper = self.request_helper();
        let mut request_data = request.clone();
        request_data.stream = Some(true);

        #[cfg(feature = "leaky-bucket")]
        let rate_limiter = self.leaky_bucket.clone();

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
    pub async fn list_models(&self) -> Result<response::ModelsResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().list_models().await
    }

    /// Create embeddings
    pub async fn create_embeddings(&self, request: &request::EmbeddingsRequest) -> Result<response::EmbeddingsResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_embeddings(request).await
    }

    /// Moderate content
    pub async fn create_moderation(&self, request: &request::ModerationRequest) -> Result<response::ModerationResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_moderation(request).await
    }

    /// Moderate chat content
    pub async fn create_chat_moderation(&self, request: &request::ChatModerationRequest) -> Result<response::ModerationResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_chat_moderation(request).await
    }

    /// List fine-tuning jobs
    pub async fn list_fine_tuning_jobs(&self) -> Result<response::FineTuningJobsResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().list_fine_tuning_jobs().await
    }

    /// Create fine-tuning job
    pub async fn create_fine_tuning_job(&self, request: &request::FineTuningRequest) -> Result<response::FineTuningJob, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_fine_tuning_job(request).await
    }

    /// Get fine-tuning job
    pub async fn retrieve_fine_tuning_job(&self, job_id: &str) -> Result<response::FineTuningJob, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().retrieve_fine_tuning_job(job_id).await
    }

    /// Cancel fine-tuning job
    pub async fn cancel_fine_tuning_job(&self, job_id: &str) -> Result<response::FineTuningJob, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().cancel_fine_tuning_job(job_id).await
    }

    /// List batch jobs
    pub async fn list_batch_jobs(&self) -> Result<response::BatchJobsResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().list_batch_jobs().await
    }

    /// Create batch job
    pub async fn create_batch_job(&self, request: &request::BatchJobRequest) -> Result<response::BatchJob, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_batch_job(request).await
    }

    /// Get batch job
    pub async fn retrieve_batch_job(&self, job_id: &str) -> Result<response::BatchJob, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().retrieve_batch_job(job_id).await
    }

    /// Cancel batch job
    pub async fn cancel_batch_job(&self, job_id: &str) -> Result<response::BatchJob, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().cancel_batch_job(job_id).await
    }

    /// List files
    pub async fn list_files(&self) -> Result<response::FilesResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().list_files().await
    }

    /// Get file information
    pub async fn retrieve_file(&self, file_id: &str) -> Result<response::FileInfo, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().retrieve_file(file_id).await
    }

    /// Delete file
    pub async fn delete_file(&self, file_id: &str) -> Result<response::FileDeleteResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().delete_file(file_id).await
    }

    /// Fill-in-the-middle completion
    pub async fn create_fim_completion(&self, request: &request::FimRequest) -> Result<response::ChatResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_fim_completion(request).await
    }

    /// Agents completion
    pub async fn create_agents_completion(&self, request: &request::AgentsRequest) -> Result<response::ChatResponse, MistralRequestError> {
        #[cfg(feature = "leaky-bucket")]
        if let Some(ref limiter) = self.leaky_bucket {
            limiter.acquire_one().await;
        }

        self.request_helper().create_agents_completion(request).await
    }
}

impl fmt::Debug for Mistral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mistral")
            .field("api_key", &"[REDACTED]")
            .field("client", &self.client)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}