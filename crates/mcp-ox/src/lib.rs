pub mod config;
pub mod content;
pub mod error;
pub mod tool;

pub use config::*;
pub use error::*;

/// Local traits for MCP conversions (NOT std From/TryFrom)
pub trait ToMcp<T> {
    fn to_mcp(&self) -> Result<T, McpConversionError>;
}

pub trait FromMcp<T> {
    fn from_mcp(value: T) -> Result<Self, McpConversionError>
    where
        Self: Sized;

    fn from_mcp_with_config(
        value: T,
        _config: &ConversionConfig,
    ) -> Result<Self, McpConversionError>
    where
        Self: Sized,
    {
        Self::from_mcp(value)
    }
}
