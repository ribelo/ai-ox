use crate::{ChatRequest, ChatResponse, MistralRequestError, response::ChatCompletionChunk};
use ai_ox_common::{
    BoxStream,
    error::ProviderError,
    request_builder::{AuthMethod, Endpoint, HttpMethod, RequestBuilder, RequestConfig},
};
use futures_util::stream::BoxStream as FuturesBoxStream;

/// Mistral client helper methods using the common RequestBuilder
pub struct MistralRequestHelper {
    client: reqwest::Client,
    config: RequestConfig,
    request_builder: RequestBuilder,
}

impl std::fmt::Debug for MistralRequestHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralRequestHelper")
            .field("request_builder", &"<RequestBuilder>")
            .finish()
    }
}

impl Clone for MistralRequestHelper {
    fn clone(&self) -> Self {
        let client = self.client.clone();
        let config = self.config.clone();
        let request_builder = RequestBuilder::new(client.clone(), config.clone());

        Self {
            client,
            config,
            request_builder,
        }
    }
}

impl MistralRequestHelper {
    pub fn new(client: reqwest::Client, base_url: &str, api_key: &str) -> Self {
        let config = RequestConfig::new(base_url)
            .with_auth(AuthMethod::Bearer(api_key.to_string()))
            .with_header("accept", "application/json");

        let request_builder = RequestBuilder::new(client.clone(), config.clone());

        Self {
            client,
            config,
            request_builder,
        }
    }

    /// Send a chat completion request
    pub async fn send_chat_request(
        &self,
        request: &ChatRequest,
    ) -> Result<ChatResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/chat/completions", HttpMethod::Post);

        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Stream a chat completion request
    pub fn stream_chat_request(
        &self,
        request: &ChatRequest,
    ) -> FuturesBoxStream<'static, Result<ChatCompletionChunk, MistralRequestError>> {
        let endpoint = Endpoint::new("v1/chat/completions", HttpMethod::Post);

        // Use the common streaming implementation (no conversion needed - same type)
        let stream: BoxStream<'static, Result<ChatCompletionChunk, ProviderError>> =
            self.request_builder.stream(&endpoint, Some(request));

        // Direct cast since MistralRequestError = ProviderError
        stream
    }

    /// List available models
    pub async fn list_models(
        &self,
    ) -> Result<crate::response::ModelsResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/models", HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Generate embeddings
    pub async fn create_embeddings(
        &self,
        request: &crate::request::EmbeddingsRequest,
    ) -> Result<crate::response::EmbeddingsResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/embeddings", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Moderate content
    pub async fn create_moderation(
        &self,
        request: &crate::request::ModerationRequest,
    ) -> Result<crate::response::ModerationResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/moderations", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Moderate chat content
    pub async fn create_chat_moderation(
        &self,
        request: &crate::request::ChatModerationRequest,
    ) -> Result<crate::response::ModerationResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/chat/moderations", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// List fine-tuning jobs
    pub async fn list_fine_tuning_jobs(
        &self,
    ) -> Result<crate::response::FineTuningJobsResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/fine_tuning/jobs", HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Send a transcription request
    pub async fn send_transcription_request(
        &self,
        request: &crate::audio::TranscriptionRequest,
    ) -> Result<crate::audio::TranscriptionResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/audio/transcriptions", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Create fine-tuning job
    pub async fn create_fine_tuning_job(
        &self,
        request: &crate::request::FineTuningRequest,
    ) -> Result<crate::response::FineTuningJob, MistralRequestError> {
        let endpoint = Endpoint::new("v1/fine_tuning/jobs", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Get fine-tuning job
    pub async fn retrieve_fine_tuning_job(
        &self,
        job_id: &str,
    ) -> Result<crate::response::FineTuningJob, MistralRequestError> {
        let endpoint = Endpoint::new(format!("v1/fine_tuning/jobs/{}", job_id), HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Cancel fine-tuning job
    pub async fn cancel_fine_tuning_job(
        &self,
        job_id: &str,
    ) -> Result<crate::response::FineTuningJob, MistralRequestError> {
        let endpoint = Endpoint::new(
            format!("v1/fine_tuning/jobs/{}/cancel", job_id),
            HttpMethod::Post,
        );
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// List batch jobs
    pub async fn list_batch_jobs(
        &self,
    ) -> Result<crate::response::BatchJobsResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/batch/jobs", HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Create batch job
    pub async fn create_batch_job(
        &self,
        request: &crate::request::BatchJobRequest,
    ) -> Result<crate::response::BatchJob, MistralRequestError> {
        let endpoint = Endpoint::new("v1/batch/jobs", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Get batch job
    pub async fn retrieve_batch_job(
        &self,
        job_id: &str,
    ) -> Result<crate::response::BatchJob, MistralRequestError> {
        let endpoint = Endpoint::new(format!("v1/batch/jobs/{}", job_id), HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Cancel batch job
    pub async fn cancel_batch_job(
        &self,
        job_id: &str,
    ) -> Result<crate::response::BatchJob, MistralRequestError> {
        let endpoint = Endpoint::new(format!("v1/batch/jobs/{}/cancel", job_id), HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// List files
    pub async fn list_files(&self) -> Result<crate::response::FilesResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/files", HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Get file information
    pub async fn retrieve_file(
        &self,
        file_id: &str,
    ) -> Result<crate::response::FileInfo, MistralRequestError> {
        let endpoint = Endpoint::new(format!("v1/files/{}", file_id), HttpMethod::Get);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Delete file
    pub async fn delete_file(
        &self,
        file_id: &str,
    ) -> Result<crate::response::FileDeleteResponse, MistralRequestError> {
        let endpoint = Endpoint::new(format!("v1/files/{}", file_id), HttpMethod::Delete);
        Ok(self
            .request_builder
            .request_json(&endpoint, None::<&()>)
            .await?)
    }

    /// Fill-in-the-middle completion
    pub async fn create_fim_completion(
        &self,
        request: &crate::request::FimRequest,
    ) -> Result<crate::response::ChatResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/fim/completions", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }

    /// Agents completion
    pub async fn create_agents_completion(
        &self,
        request: &crate::request::AgentsRequest,
    ) -> Result<crate::response::ChatResponse, MistralRequestError> {
        let endpoint = Endpoint::new("v1/agents/completions", HttpMethod::Post);
        Ok(self
            .request_builder
            .request_json(&endpoint, Some(request))
            .await?)
    }
}
