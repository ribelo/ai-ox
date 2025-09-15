use std::collections::HashMap;
use serde_json::Value;

/// Trait for services that can upload binary data and return a URI
pub trait Uploader: Send + Sync + std::fmt::Debug {
    /// Upload binary data and return a URI that can be used to reference it
    fn upload(&self, data: Vec<u8>, mime_type: String, name: Option<String>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, UploadError>> + Send>>;
}

#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("Upload failed: {0}")]
    Failed(String),
    #[error("Upload service unavailable")]
    Unavailable,
    #[error("File too large: {size} bytes exceeds limit")]
    TooLarge { size: usize },
}

/// Policy for handling content that a provider doesn't support
#[derive(Debug)]
pub enum ConversionPolicy {
    /// Fail with an error if any content can't be represented exactly (default)
    Strict,
    
    /// Allow uploading base64 data to get URIs when provider doesn't support base64
    UploadAllowed {
        uploader: std::sync::Arc<MockUploader>,
    },
    
    /// Allow storing original content in metadata when provider can't handle it
    ShadowAllowed,
    
    /// Both upload and shadow are allowed
    Combined {
        uploader: std::sync::Arc<MockUploader>,
    },
}

impl Default for ConversionPolicy {
    fn default() -> Self {
        Self::Strict
    }
}

/// Describes a transformation needed during conversion
#[derive(Debug, Clone)]
pub enum TransformAction {
    /// Content can be passed through as-is
    PassThrough,
    
    /// Base64 data needs to be uploaded to get a URI
    UploadBase64 {
        original_size: usize,
        mime_type: String,
    },
    
    /// Content needs to be stored in shadow metadata
    Shadow {
        original_type: String,
        simplified_to: String,
    },
    
    /// Content must be omitted (only in non-strict mode with explicit policy)
    Omit {
        reason: String,
    },
}

/// Plan for converting a message to a specific provider
#[derive(Default)]
pub struct ConversionPlan {
    /// Provider this plan is for
    pub provider_name: String,
    
    /// Policy being applied
    pub policy_name: String,
    
    /// Actions for each part (indexed by position)
    pub part_actions: Vec<TransformAction>,
    
    /// Any errors that would prevent conversion
    pub errors: Vec<ConversionError>,
    
    /// Warnings about potential data loss
    pub warnings: Vec<String>,
    
    /// Metadata to be added for roundtrip preservation
    pub shadow_metadata: HashMap<String, Value>,
}

impl ConversionPlan {
    pub fn new(provider_name: impl Into<String>, policy: &ConversionPolicy) -> Self {
        let policy_name = match policy {
            ConversionPolicy::Strict => "Strict",
            ConversionPolicy::UploadAllowed { .. } => "UploadAllowed",
            ConversionPolicy::ShadowAllowed => "ShadowAllowed",
            ConversionPolicy::Combined { .. } => "Combined",
        };
        
        Self {
            provider_name: provider_name.into(),
            policy_name: policy_name.to_string(),
            ..Default::default()
        }
    }
    
    /// Check if this plan can be executed without data loss
    pub fn is_lossless(&self) -> bool {
        self.errors.is_empty() && 
        !self.part_actions.iter().any(|a| matches!(a, TransformAction::Omit { .. }))
    }
    
    /// Check if this plan has any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    /// Add an error to this plan
    pub fn add_error(&mut self, error: ConversionError) {
        self.errors.push(error);
    }
    
    /// Add a warning to this plan
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    
    /// Plan an action for a specific part
    pub fn add_action(&mut self, action: TransformAction) {
        self.part_actions.push(action);
    }
}

/// Error that occurs during conversion planning or execution
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConversionError {
    #[error("Part at index {part_index} ({part_type}) is not supported by {provider}: {reason}")]
    UnsupportedContent {
        part_index: usize,
        part_type: String,
        provider: String,
        reason: String,
    },
    
    #[error("MIME type '{mime_type}' is not supported by {provider}")]
    UnsupportedMimeType {
        mime_type: String,
        provider: String,
    },
    
    #[error("Base64 data too large ({size} bytes) for {provider} (max: {max_size} bytes)")]
    Base64TooLarge {
        size: usize,
        max_size: usize,
        provider: String,
    },
    
    #[error("Provider {provider} requires {required_feature} but it's not available")]
    MissingRequiredFeature {
        provider: String,
        required_feature: String,
    },
    
    #[error("Upload required but no uploader provided in policy")]
    NoUploaderAvailable,
    
    #[error("Shadow metadata required but provider doesn't support metadata passthrough")]
    NoShadowSupport,
}

/// Mock uploader for testing
#[derive(Debug, Clone)]
pub struct MockUploader {
    pub base_url: String,
}

impl MockUploader {
    pub fn new() -> Self {
        Self {
            base_url: "https://mock-storage.example.com".to_string(),
        }
    }
}

#[cfg(test)]
impl Uploader for MockUploader {
    fn upload(&self, data: Vec<u8>, mime_type: String, name: Option<String>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, UploadError>> + Send + 'static>> {
        // Generate a fake URL based on content
        let hash = format!("{:x}", md5::compute(&data));
        let extension = mime_type.split('/').nth(1).unwrap_or("bin").to_string();
        let filename = name.unwrap_or(hash);
        let base_url = self.base_url.clone();
        Box::pin(async move {
            Ok(format!("{}/files/{}.{}", base_url, filename, extension))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conversion_plan_lossless() {
        let mut plan = ConversionPlan::new("test", &ConversionPolicy::Strict);
        plan.add_action(TransformAction::PassThrough);
        plan.add_action(TransformAction::UploadBase64 {
            original_size: 1024,
            mime_type: "image/png".to_string(),
        });
        
        assert!(plan.is_lossless());
        assert!(!plan.has_errors());
        
        // Add an omit action - no longer lossless
        plan.add_action(TransformAction::Omit {
            reason: "Not supported".to_string(),
        });
        assert!(!plan.is_lossless());
    }
    
    #[test]
    fn test_policy_default_is_strict() {
        let policy = ConversionPolicy::default();
        matches!(policy, ConversionPolicy::Strict);
    }
}