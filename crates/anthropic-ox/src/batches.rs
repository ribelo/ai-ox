// This file will contain the implementation for the Message Batches API.
use crate::error::ErrorInfo;
use crate::request::ChatRequest;
use crate::response::ChatResponse;
use serde::{Deserialize, Serialize};

/// A request to create a new message batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBatchRequest {
    /// A list of requests to include in the batch.
    pub requests: Vec<BatchMessageRequest>,
}

/// A single request within a message batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMessageRequest {
    /// A unique identifier for the request within the batch.
    pub custom_id: String,
    /// The parameters for the individual message creation request.
    pub params: ChatRequest,
}

/// The status of a message batch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    InProgress,
    Canceling,
    Ended,
}

/// The counts of requests in a batch, categorized by status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestCounts {
    /// The number of requests that were canceled.
    pub canceled: u32,
    /// The number of requests that timed out.
    pub errored: u32,
    /// The number of requests that expired.
    pub expired: u32,
    /// The number of requests that are currently processing.
    pub processing: u32,
    /// The number of requests that succeeded.
    pub succeeded: u32,
}

/// Represents a message batch object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageBatch {
    /// The unique identifier for the message batch.
    pub id: String,
    /// The type of the object, which is always "message_batch".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The timestamp of when the batch was archived.
    pub archived_at: Option<String>,
    /// The timestamp of when the batch cancellation was initiated.
    pub cancel_initiated_at: Option<String>,
    /// The timestamp of when the batch was created.
    pub created_at: String,
    /// The timestamp of when the batch ended processing.
    pub ended_at: Option<String>,
    /// The timestamp of when the batch will expire.
    pub expires_at: String,
    /// The current processing status of the batch.
    pub processing_status: BatchStatus,
    /// The counts of requests in the batch, categorized by status.
    pub request_counts: RequestCounts,
    /// The URL where the results of the batch can be downloaded.
    pub results_url: Option<String>,
}

/// A response containing a list of message batches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BatchListResponse {
    /// The list of message batches.
    pub data: Vec<MessageBatch>,
    /// Indicates if there are more batches to fetch.
    pub has_more: bool,
    /// The ID of the first batch in the list.
    pub first_id: Option<String>,
    /// The ID of the last batch in the list.
    pub last_id: Option<String>,
}

/// The result of a single request in a message batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// The unique identifier for the request.
    pub custom_id: String,
    /// The response for the request, which can be a success or an error.
    #[serde(flatten)]
    pub response: BatchResultResponse,
}

/// The response for a single batch result, which can be a success or an error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BatchResultResponse {
    /// A successful response.
    Success(ChatResponse),
    /// An error response.
    Error { error: ErrorInfo },
}
