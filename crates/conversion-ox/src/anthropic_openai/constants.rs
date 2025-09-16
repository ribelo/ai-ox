/// Constants for Anthropic-OpenAI conversions

// Default token limits
pub const DEFAULT_MAX_TOKENS: u32 = 4096;
pub const DEFAULT_THINKING_BUDGET: u32 = 8192;

// Reasoning configuration
pub const REASONING_EFFORT_HIGH: &str = "high";
pub const REASONING_SUMMARY_AUTO: &str = "auto";

// Response status
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_IN_PROGRESS: &str = "in_progress";

// Roles
pub const ROLE_ASSISTANT: &str = "assistant";
pub const ROLE_USER: &str = "user";

// Message types
pub const MESSAGE_TYPE: &str = "message";
