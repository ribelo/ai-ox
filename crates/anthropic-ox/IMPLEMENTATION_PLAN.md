# Anthropic-ox API Coverage Implementation Plan

## Executive Summary

The current `anthropic-ox` crate provides a solid foundation for Anthropic API integration, implementing core messaging functionality with modern Rust patterns. However, it covers only **~15%** of the full Anthropic API surface area, focusing primarily on the Messages API. This plan outlines the roadmap to expand coverage to **~80%** of common use cases.

## Current Implementation Status

### ✅ **Implemented Features** (Well-Covered)
1. **Messages API** (POST /v1/messages) - **Complete**
   - Request/response structures
   - Streaming support 
   - Tool use integration
   - Vision support (image content)
   - Thinking content (recently added)
   - Error handling
   - OAuth and API key authentication

2. **Core Data Types** - **Complete**
   - Message structures with role-based content
   - Content blocks (Text, Image, ToolUse, ToolResult, Thinking)
   - Tool definitions and choice strategies
   - Usage tracking structures
   - Streaming event types

3. **Client Infrastructure** - **Good**
   - HTTP client with proper authentication
   - Rate limiting support (feature-gated)
   - Custom headers support
   - Environment variable loading

## 📋 **Implementation Roadmap**

### Phase 1: Essential Missing Features (High Priority)

#### 1.1 Token Counting API
**Endpoints**: `POST /v1/messages/count-tokens`
**Business Impact**: High (cost optimization)
**Implementation Effort**: Low

**Tasks**:
- [ ] Add `TokenCountRequest` struct
- [ ] Add `TokenCountResponse` struct  
- [ ] Implement `count_tokens()` method on `Anthropic` client
- [ ] Add cost estimation utilities
- [ ] Write comprehensive tests

**Implementation**:
```rust
// src/token_counting.rs
pub struct TokenCountRequest {
    pub model: String,
    pub messages: Messages,
    // ... other fields from ChatRequest
}

pub struct TokenCountResponse {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
}

impl Anthropic {
    pub async fn count_tokens(&self, request: &TokenCountRequest) 
        -> Result<TokenCountResponse, AnthropicRequestError> {
        // Implementation
    }
}
```

#### 1.2 Models API  
**Endpoints**: `GET /v1/models`, `GET /v1/models/{model_id}`
**Business Impact**: Medium (dynamic model discovery)
**Implementation Effort**: Low

**Tasks**:
- [ ] Add `ModelsListResponse` struct
- [ ] Add `ModelInfo` struct with capabilities
- [ ] Implement `list_models()` method
- [ ] Implement `get_model()` method
- [ ] Add dynamic model string support alongside enum
- [ ] Write comprehensive tests

**Implementation**:
```rust
// src/models.rs
pub struct ModelInfo {
    pub id: String,
    pub display_name: String,
    pub created_at: String,
    pub max_tokens: Option<u32>,
    pub input_cost_per_mtok: Option<f64>,
    pub output_cost_per_mtok: Option<f64>,
}

pub struct ModelsListResponse {
    pub data: Vec<ModelInfo>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
}

impl Anthropic {
    pub async fn list_models(&self) -> Result<ModelsListResponse, AnthropicRequestError>;
    pub async fn get_model(&self, model_id: &str) -> Result<ModelInfo, AnthropicRequestError>;
}
```

#### 1.3 Enhanced Streaming Support
**Features**: Fine-grained tool streaming, Interleaved thinking
**Business Impact**: Medium (performance improvement)
**Implementation Effort**: Low

**Tasks**:
- [ ] Add `BetaFeatures` configuration struct
- [ ] Add beta header management
- [ ] Enhance streaming for fine-grained tool parameters
- [ ] Add interleaved thinking support
- [ ] Update streaming event types
- [ ] Write comprehensive tests

**Implementation**:
```rust
// src/client.rs
pub struct BetaFeatures {
    pub fine_grained_tool_streaming: bool,
    pub interleaved_thinking: bool,
    pub search_results: bool,
}

impl Anthropic {
    pub fn with_beta_features(mut self, features: BetaFeatures) -> Self {
        if features.fine_grained_tool_streaming {
            self.headers.insert(
                "anthropic-beta".to_string(),
                "fine-grained-tool-streaming-2025-05-14".to_string()
            );
        }
        // ... other beta features
        self
    }
}
```

### Phase 2: Batch Processing (Medium Priority)

#### 2.1 Message Batches API
**Endpoints**: All batch processing endpoints
**Business Impact**: High (50% cost savings)
**Implementation Effort**: Medium

**Tasks**:
- [ ] Add `MessageBatch` structs for all operations
- [ ] Add `BatchRequest` and `BatchResponse` types
- [ ] Implement all batch management methods
- [ ] Add async polling utilities
- [ ] Add pagination support for listing
- [ ] Handle JSONL result streaming
- [ ] Write comprehensive tests including async scenarios

**Implementation**:
```rust
// src/batches.rs
pub struct MessageBatchRequest {
    pub requests: Vec<BatchMessageRequest>,
    pub completion_window: String, // "24h"
}

pub struct MessageBatch {
    pub id: String,
    pub object: String, // "message_batch"
    pub processing_status: BatchStatus,
    pub request_counts: RequestCounts,
    pub ended_at: Option<String>,
    pub created_at: String,
    pub expires_at: String,
    pub cancel_initiated_at: Option<String>,
}

impl Anthropic {
    pub async fn create_message_batch(&self, request: &MessageBatchRequest) -> Result<MessageBatch, AnthropicRequestError>;
    pub async fn list_message_batches(&self, limit: Option<u32>, before_id: Option<&str>) -> Result<BatchListResponse, AnthropicRequestError>;
    pub async fn get_message_batch(&self, batch_id: &str) -> Result<MessageBatch, AnthropicRequestError>;
    pub async fn get_message_batch_results(&self, batch_id: &str) -> Result<BatchResultsStream, AnthropicRequestError>;
    pub async fn cancel_message_batch(&self, batch_id: &str) -> Result<MessageBatch, AnthropicRequestError>;
}
```

#### 2.2 Files API (Beta)
**Endpoints**: All file management endpoints
**Business Impact**: Medium (asset management)
**Implementation Effort**: Medium

**Tasks**:
- [ ] Add multipart form data support
- [ ] Add `FileUpload` and `FileInfo` structs
- [ ] Implement all file management methods
- [ ] Add file streaming for downloads
- [ ] Handle MIME type detection
- [ ] Add file lifecycle management utilities
- [ ] Write comprehensive tests including file I/O

**Implementation**:
```rust
// src/files.rs
pub struct FileUpload {
    pub file: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
}

pub struct FileInfo {
    pub id: String,
    pub object: String, // "file"
    pub size_bytes: u64,
    pub filename: String,
    pub mime_type: String,
    pub created_at: String,
}

impl Anthropic {
    pub async fn upload_file(&self, file_upload: FileUpload) -> Result<FileInfo, AnthropicRequestError>;
    pub async fn list_files(&self, limit: Option<u32>, before_id: Option<&str>) -> Result<FileListResponse, AnthropicRequestError>;
    pub async fn get_file(&self, file_id: &str) -> Result<FileInfo, AnthropicRequestError>;
    pub async fn download_file(&self, file_id: &str) -> Result<Vec<u8>, AnthropicRequestError>;
    pub async fn delete_file(&self, file_id: &str) -> Result<(), AnthropicRequestError>;
}
```

### Phase 3: Advanced Features (Lower Priority)

#### 3.1 Admin APIs
**Endpoints**: Organization and workspace management
**Business Impact**: Low (specialized use case)
**Implementation Effort**: Medium

**Tasks**:
- [ ] Add admin authentication support (`sk-ant-admin-...`)
- [ ] Add organization management structs
- [ ] Add workspace management structs
- [ ] Add usage reporting structs
- [ ] Implement all admin endpoints
- [ ] Add proper authorization handling
- [ ] Write comprehensive tests

#### 3.2 Experimental Features
**Features**: Computer Use, Search Results, Prompt Tools
**Business Impact**: Low (experimental/beta)
**Implementation Effort**: High

**Tasks**:
- [ ] Add Computer Use API support
- [ ] Add Search Results content blocks
- [ ] Add Prompt Tools (when available)
- [ ] Handle experimental API changes gracefully
- [ ] Write comprehensive tests

## 🛠 **Technical Architecture Changes**

### Modular Structure Reorganization
```
src/
├── lib.rs              // Main exports and client
├── client.rs           // Core client (extracted from lib.rs)
├── auth.rs             // Authentication handling
├── error.rs            // Error types (existing)
├── messages/           // Messages API (existing functionality)
│   ├── mod.rs
│   ├── request.rs
│   ├── response.rs
│   └── streaming.rs
├── tokens.rs           // Token counting API
├── models.rs           // Models API
├── batches.rs          // Batch processing
├── files.rs            // File management
├── admin/              // Admin APIs
│   ├── mod.rs
│   ├── organizations.rs
│   ├── workspaces.rs
│   └── usage.rs
└── experimental/       // Beta/experimental features
    ├── mod.rs
    ├── computer_use.rs
    ├── search_results.rs
    └── prompt_tools.rs
```

### Feature Flags
```toml
[features]
default = ["messages", "tokens", "models"]
messages = []           # Core messaging (always included)
tokens = []             # Token counting API
models = []             # Models API
batches = []            # Batch processing
files = []              # File management
admin = []              # Admin APIs
experimental = []       # Beta/experimental features
full = ["tokens", "models", "batches", "files", "admin", "experimental"]
```

### Error Handling Enhancement
```rust
// src/error.rs - Enhanced error types
#[derive(Debug, thiserror::Error)]
pub enum AnthropicRequestError {
    // Existing errors...
    
    // New API-specific errors
    #[error("Batch processing error: {0}")]
    BatchError(String),
    
    #[error("File operation error: {0}")]
    FileError(String),
    
    #[error("Admin API error: {0}")]
    AdminError(String),
    
    #[error("Experimental API error: {0}")]
    ExperimentalError(String),
}
```

## 📊 **Implementation Priority Matrix**

| Feature | Business Impact | Implementation Effort | Priority | Timeline |
|---------|----------------|----------------------|----------|----------|
| Token Counting | High (cost optimization) | Low | **P0** | Week 1-2 |
| Models API | Medium (discovery) | Low | **P0** | Week 2-3 |
| Enhanced Streaming | Medium (performance) | Low | **P1** | Week 3-4 |
| Batch Processing | High (50% cost savings) | Medium | **P1** | Week 4-6 |
| Files API | Medium (asset management) | Medium | **P1** | Week 6-8 |
| Admin APIs | Low (specialized) | Medium | **P2** | Week 9-11 |
| Computer Use | Low (beta) | High | **P3** | Week 12+ |

## 🎯 **Success Metrics**

### Coverage Goals
- **Phase 1 Completion**: ~60% API coverage (vs current ~15%)
- **Phase 2 Completion**: ~80% API coverage  
- **Phase 3 Completion**: ~95% API coverage

### Quality Metrics
- [ ] 100% test coverage for new endpoints
- [ ] Comprehensive documentation for all new features
- [ ] Performance benchmarks for batch processing
- [ ] Memory usage optimization for file handling
- [ ] Backward compatibility maintenance

### Community Impact
- [ ] Significant cost savings through batch processing
- [ ] Improved developer experience with model discovery
- [ ] Better cost estimation through token counting
- [ ] Enhanced streaming performance

## 🔄 **Migration Strategy**

### Backward Compatibility
- All existing APIs remain unchanged
- New features added as additional methods
- Feature flags allow gradual adoption
- Clear migration guides for enhanced features

### Breaking Changes (Future v2.0)
- Consolidate streaming interfaces
- Optimize authentication patterns  
- Streamline error hierarchy
- Improve async patterns

## 📚 **Documentation Plan**

### API Documentation
- [ ] Comprehensive rustdoc for all new types
- [ ] Usage examples for each endpoint
- [ ] Integration guides for common patterns
- [ ] Performance and cost optimization guides

### Community Resources
- [ ] Migration guide from basic to full API usage
- [ ] Best practices for batch processing
- [ ] Cost optimization strategies
- [ ] Troubleshooting guide

This implementation plan transforms `anthropic-ox` from a basic client to a comprehensive Anthropic API SDK, providing significant value through cost optimization, enhanced functionality, and improved developer experience.