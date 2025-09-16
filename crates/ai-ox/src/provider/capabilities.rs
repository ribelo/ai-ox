use std::collections::HashSet;

/// Describes what content types and features a provider supports
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    /// Provider name for debugging
    pub provider_name: String,

    /// Can accept base64-encoded binary data in requests
    pub supports_base64_blob_input: bool,

    /// Can accept URI references to external files
    pub supports_blob_uri_input: bool,

    /// Supports images in messages
    pub supports_images: bool,

    /// Supports audio in messages
    pub supports_audio: bool,

    /// Supports file attachments
    pub supports_files: bool,

    /// Can handle tool calls
    pub supports_tool_use: bool,

    /// Can handle tool results with multiple parts
    pub supports_tool_result_parts: bool,

    /// Can preserve metadata through roundtrip
    pub supports_metadata_passthrough: bool,

    /// MIME types this provider accepts
    pub allowed_mime_inputs: HashSet<String>,

    /// Maximum size for base64 data (bytes), None = no limit
    pub max_base64_size: Option<usize>,
}

impl Capabilities {
    pub fn new(provider_name: impl Into<String>) -> Self {
        Self {
            provider_name: provider_name.into(),
            ..Default::default()
        }
    }

    /// Anthropic Claude capabilities
    pub fn anthropic() -> Self {
        let mut caps = Self::new("anthropic");
        caps.supports_base64_blob_input = true;
        caps.supports_images = true;
        caps.supports_tool_use = true;
        caps.supports_tool_result_parts = false; // Only supports text/image in tool results
        caps.allowed_mime_inputs = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        caps.max_base64_size = Some(5 * 1024 * 1024); // 5MB limit
        caps
    }

    /// OpenAI GPT capabilities
    pub fn openai() -> Self {
        let mut caps = Self::new("openai");
        caps.supports_base64_blob_input = true;
        caps.supports_blob_uri_input = true;
        caps.supports_images = true;
        caps.supports_audio = true; // GPT-4-audio models
        caps.supports_tool_use = true;
        caps.supports_tool_result_parts = false;
        caps.allowed_mime_inputs = [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "audio/wav",
            "audio/mp3",
            "audio/ogg",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        caps
    }

    /// Google Gemini capabilities
    pub fn gemini() -> Self {
        let mut caps = Self::new("gemini");
        caps.supports_base64_blob_input = true;
        caps.supports_blob_uri_input = true;
        caps.supports_images = true;
        caps.supports_audio = true;
        caps.supports_files = true;
        caps.supports_tool_use = true;
        caps.supports_tool_result_parts = true; // Gemini has rich tool results
        caps.allowed_mime_inputs = ["image/*", "audio/*", "video/*", "application/pdf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        caps
    }

    /// Mistral capabilities
    pub fn mistral() -> Self {
        let mut caps = Self::new("mistral");
        caps.supports_base64_blob_input = false; // Mistral doesn't handle base64 well
        caps.supports_blob_uri_input = true;
        caps.supports_images = true; // Via Pixtral models
        caps.supports_tool_use = true;
        caps.supports_tool_result_parts = false;
        caps.allowed_mime_inputs = ["image/jpeg", "image/png"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        caps
    }

    /// OpenRouter capabilities (depends on underlying model)
    pub fn openrouter() -> Self {
        let mut caps = Self::new("openrouter");
        // OpenRouter is a proxy, capabilities depend on the specific model
        // This is a conservative default
        caps.supports_base64_blob_input = true;
        caps.supports_images = true;
        caps.supports_tool_use = true;
        caps.supports_tool_result_parts = false;
        caps.allowed_mime_inputs = ["image/jpeg", "image/png", "image/webp"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        caps
    }

    /// Check if a specific MIME type is supported
    pub fn supports_mime(&self, mime_type: &str) -> bool {
        // Check exact match
        if self.allowed_mime_inputs.contains(mime_type) {
            return true;
        }

        // Check wildcard patterns like "image/*"
        for allowed in &self.allowed_mime_inputs {
            if allowed.ends_with("/*") {
                let prefix = &allowed[..allowed.len() - 2];
                if mime_type.starts_with(prefix) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if base64 data of given size is supported
    pub fn can_accept_base64(&self, size: usize) -> bool {
        if !self.supports_base64_blob_input {
            return false;
        }

        match self.max_base64_size {
            Some(max) => size <= max,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_capabilities() {
        let caps = Capabilities::anthropic();

        assert_eq!(caps.provider_name, "anthropic");
        assert!(caps.supports_base64_blob_input);
        assert!(caps.supports_images);
        assert!(caps.supports_tool_use);
        assert!(!caps.supports_tool_result_parts);
        assert!(caps.supports_mime("image/jpeg"));
        assert!(caps.supports_mime("image/png"));
        assert!(!caps.supports_mime("audio/wav"));
        assert!(caps.can_accept_base64(1024));
        assert!(!caps.can_accept_base64(10 * 1024 * 1024)); // Over 5MB limit
    }

    #[test]
    fn test_openai_capabilities() {
        let caps = Capabilities::openai();

        assert_eq!(caps.provider_name, "openai");
        assert!(caps.supports_base64_blob_input);
        assert!(caps.supports_blob_uri_input);
        assert!(caps.supports_images);
        assert!(caps.supports_audio);
        assert!(caps.supports_tool_use);
        assert!(!caps.supports_tool_result_parts);
        assert!(caps.supports_mime("image/jpeg"));
        assert!(caps.supports_mime("audio/wav"));
        assert!(caps.can_accept_base64(1024)); // No size limit
    }

    #[test]
    fn test_gemini_capabilities() {
        let caps = Capabilities::gemini();

        assert_eq!(caps.provider_name, "gemini");
        assert!(caps.supports_base64_blob_input);
        assert!(caps.supports_blob_uri_input);
        assert!(caps.supports_images);
        assert!(caps.supports_audio);
        assert!(caps.supports_files);
        assert!(caps.supports_tool_use);
        assert!(caps.supports_tool_result_parts); // Gemini has rich tool results
        assert!(caps.supports_mime("image/jpeg"));
        assert!(caps.supports_mime("video/mp4")); // Wildcard match
        assert!(caps.supports_mime("application/pdf"));
    }

    #[test]
    fn test_mistral_capabilities() {
        let caps = Capabilities::mistral();

        assert_eq!(caps.provider_name, "mistral");
        assert!(!caps.supports_base64_blob_input); // Mistral doesn't handle base64 well
        assert!(caps.supports_blob_uri_input);
        assert!(caps.supports_images);
        assert!(caps.supports_tool_use);
        assert!(!caps.supports_tool_result_parts);
        assert!(caps.supports_mime("image/jpeg"));
        assert!(!caps.can_accept_base64(1024)); // Doesn't support base64
    }

    #[test]
    fn test_openrouter_capabilities() {
        let caps = Capabilities::openrouter();

        assert_eq!(caps.provider_name, "openrouter");
        assert!(caps.supports_base64_blob_input);
        assert!(caps.supports_images);
        assert!(caps.supports_tool_use);
        assert!(!caps.supports_tool_result_parts);
        assert!(caps.supports_mime("image/jpeg"));
        assert!(caps.supports_mime("image/webp"));
    }

    #[test]
    fn test_mime_wildcard_matching() {
        let mut caps = Capabilities::new("test");
        caps.allowed_mime_inputs.insert("image/*".to_string());
        caps.allowed_mime_inputs.insert("audio/wav".to_string());

        assert!(caps.supports_mime("image/jpeg"));
        assert!(caps.supports_mime("image/png"));
        assert!(caps.supports_mime("audio/wav"));
        assert!(!caps.supports_mime("video/mp4"));
        assert!(!caps.supports_mime("text/plain"));
    }

    #[test]
    fn test_base64_size_limits() {
        let mut caps = Capabilities::new("test");
        caps.supports_base64_blob_input = true;
        caps.max_base64_size = Some(1024);

        assert!(caps.can_accept_base64(512));
        assert!(caps.can_accept_base64(1024));
        assert!(!caps.can_accept_base64(2048));

        // Test no limit
        caps.max_base64_size = None;
        assert!(caps.can_accept_base64(10 * 1024 * 1024));
    }
}
