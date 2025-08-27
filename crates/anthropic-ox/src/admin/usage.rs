use serde::{Deserialize, Serialize};

/// The time granularity for usage and cost reports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimeGranularity {
    #[serde(rename = "1m")]
    Minute,
    #[serde(rename = "1h")]
    Hour,
    #[serde(rename = "1d")]
    Day,
}

/// The dimensions to group usage reports by.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UsageDimension {
    ApiKeyId,
    Model,
    ServiceTier,
    WorkspaceId,
}

/// The dimensions to group cost reports by.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CostDimension {
    WorkspaceId,
    Description,
}

/// The token counts for a usage report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageTokens {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

/// The dimensions for a usage report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageDimensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
}

/// A single usage report entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageReport {
    pub time_bucket: String,
    pub dimensions: UsageDimensions,
    pub tokens: UsageTokens,
}

/// A response containing a usage report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageReportResponse {
    pub data: Vec<UsageReport>,
    pub has_more: bool,
    pub next_page: Option<String>,
}

/// The cost breakdown for a cost report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Cost {
    pub input_tokens_cost: String,
    pub output_tokens_cost: String,
    pub web_search_cost: String,
    pub code_execution_cost: String,
}

/// The dimensions for a cost report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CostDimensions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A single cost report entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CostReport {
    pub time_bucket: String,
    pub dimensions: CostDimensions,
    pub cost: Cost,
}

/// A response containing a cost report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CostReportResponse {
    pub data: Vec<CostReport>,
    pub has_more: bool,
    pub next_page: Option<String>,
}
