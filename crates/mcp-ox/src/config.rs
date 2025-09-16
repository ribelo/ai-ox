/// Configuration for MCP conversions
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Whether to use strict mode (default true)
    /// In strict mode: Error if missing required fields (ID, name)
    /// In lenient mode: Generate defaults but log warning
    pub strict: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self { strict: true }
    }
}
