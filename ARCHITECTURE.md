# ai-ox Architecture

<!--
TODO(Jules): This document has been updated to reflect the current architecture.
It's crucial to keep this document in sync with any future architectural
changes to ensure it remains a useful resource for developers.
-->

This document describes the architectural design and patterns used in the ai-ox Rust workspace.

## Overview

ai-ox is designed as a provider-agnostic AI integration framework with a unified abstraction layer. The architecture emphasizes trait-based abstractions, composability, and extensibility while maintaining type safety and performance.

## Workspace Structure

The workspace is organized into a core crate (`ai-ox`), several provider-specific client crates (`*-ox`), and a support crate for macros.

```
ai-ox/
├── crates/
│   ├── ai-ox/           # Core abstractions, unified API, and provider adapters
│   ├── ai-ox-macros/    # Procedural macros (e.g., #[toolbox])
│   ├── anthropic-ox/    # Standalone client for Anthropic
│   ├── gemini-ox/       # Standalone client for Google Gemini
│   ├── groq-ox/         # Standalone client for Groq
│   ├── mistral-ox/      # Standalone client for Mistral
│   └── openrouter-ox/   # Standalone client for OpenRouter
```

### Crate Dependencies and Integration Pattern

The architecture uses an **Adapter Pattern**. The `ai-ox` crate acts as the central orchestrator and defines the core abstractions, primarily the `Model` trait.

- **Provider Crates (`*-ox`)**: These are completely standalone SDKs. They do **not** depend on `ai-ox` and have no knowledge of its traits or abstractions. Their responsibility is to provide a pure, provider-specific Rust API.
- **`ai-ox` Crate**: This crate conditionally depends on the provider crates via feature flags (e.g., `features = ["anthropic"]`). For each supported provider, `ai-ox` contains a private "adapter" module (e.g., `ai-ox/src/model/anthropic/`).
- **Adapter Modules**: This is where the integration happens. Each adapter module defines a wrapper struct (e.g., `AnthropicModel`) that contains an instance of the standalone provider client. This wrapper struct then implements the `ai-ox::model::Model` trait, "adapting" the provider-specific client to the unified `ai-ox` interface.

## Core Architecture Patterns

### 1. Trait-Based Abstraction

The architecture uses traits to define common interfaces across providers:

```rust
// Core model abstraction. Note: uses BoxFuture for object safety, not async fn.
pub trait Model: Send + Sync {
    fn request(&self, request: ModelRequest) -> BoxFuture<'_, Result<ModelResponse>>;
    fn request_stream(&self, request: ModelRequest) -> BoxStream<'_, Result<StreamEvent>>;
}

// The Agent is a concrete struct, not a trait.
// It is detailed in the "Agent System Architecture" section.
```

### 2. Builder Pattern

Consistent use of builder pattern with `bon` crate for complex object construction:

```rust
#[derive(Builder)]
pub struct ModelRequest {
    messages: Vec<Message>,
    tools: Option<ToolSet>,
    temperature: Option<f32>,
    // ...
}
```

### 3. Provider Abstraction Layer

The `ai-ox` core abstracts away provider-specific details using the `Model` trait and the adapter pattern. The dependency flow is from the core to the provider clients, with the integration logic residing within the core itself.

```
                                 ┌──────────────────────────┐
                                 │        ai-ox Core        │
                                 │ (Defines `Model` Trait)  │
                                 └───────────┬──────────────┘
                                             │
                                ┌────────────▼────────────┐
                                │   Provider Adapters     │
                                │ (Inside ai-ox::model::*)│
                                └────────────┬────────────┘
     ┌───────────────────────────────────────┤
     │                  │                    │
┌────▼─────┐     ┌──────▼───────┐     ┌──────▼───────┐
│ anthropic-ox │     │  gemini-ox   │     │  mistral-ox  │
│ (Standalone) │     │ (Standalone) │     │ (Standalone) │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Agent System Architecture

### Core Components

1. **Agent Struct** - The `Agent` is a concrete struct that encapsulates a model, a toolset, and configuration. It is the primary entry point for orchestrating conversations. It is not a trait.
2. **Tool Integration** - Seamless tool calling with macro-generated bindings
3. **Conversation Management** - Multi-turn conversation handling with context
4. **Event Streaming** - Real-time agent execution events

### Agent Flow

```
Input Message → Agent → Model → Tools → Response
     ↑                                     ↓
     └─── Tool Results ← Tool Execution ←──┘
```

### Key Features

- **Tool Execution Loop** - Automatic tool calling and result integration
- **Context Management** - Conversation history and tool state tracking
- **Error Recovery** - Graceful handling of tool failures and model errors
- **Streaming Support** - Real-time progress updates and partial results

## Model Abstraction Layer

### Unified Model Interface

The `Model` trait provides a consistent interface across all AI providers:

```rust
pub trait Model: Send + Sync + 'static {
    /// Returns the model name/identifier.
    fn model(&self) -> &str;

    /// Sends a single, non-streaming request to the model.
    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>>;

    /// Initiates a streaming request to the model.
    fn request_stream(
        &self,
        request: ModelRequest,
    ) -> BoxStream<'_, Result<StreamEvent, GenerateContentError>>;

    /// Internal, object-safe method for structured JSON responses.
    fn request_structured_internal(
        &self,
        request: ModelRequest,
        schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>>;
}
```

### Provider Implementations

The `Model` trait is implemented on **wrapper structs** inside the `ai-ox` crate's provider modules (e.g., `ai_ox::model::anthropic::AnthropicModel`). These implementations use a conversion layer to map between the generic `ai-ox` types and the provider-specific types from the standalone client crates.

- **AnthropicModel** - Adapter for the `anthropic-ox` client.
- **GeminiModel** - Adapter for the `gemini-ox` client.
- **GroqModel** - Adapter for the `groq-ox` client.
- **MistralModel** - Adapter for the `mistral-ox` client.
- **OpenRouterModel** - Adapter for the `openrouter-ox` client.

### Request/Response Pattern

```rust
pub struct ModelRequest {
    pub messages: Vec<Message>,
    pub tools: Option<ToolSet>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    // Provider-agnostic parameters
}

pub struct ModelResponse {
    pub message: Message,
    pub usage: Option<Usage>,
    pub finish_reason: FinishReason,
}
```

## Content & Message System

### Message Structure

Messages follow a standardized structure for cross-provider compatibility:

```rust
pub struct Message {
    pub role: MessageRole,      // User, Assistant
    pub content: Vec<Part>,     // Content parts
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    File(FileData),
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    ToolResult {
        call_id: String,
        name: String,
        content: Value,
    },
}
```

### Content Handling

- **Multimodal Support** - Text, image, audio, video content types
- **Streaming Events** - Incremental content updates for real-time streaming
- **Tool Integration** - Seamless tool calls and results within message flow
- **Metadata Preservation** - Provider-specific metadata without losing information

## Tool Calling Architecture

### ToolBox Macro System

Tools are defined using the `#[toolbox]` macro for automatic binding generation:

```rust
#[toolbox]
impl MyTools {
    /// Search the web for information
    pub async fn web_search(&self, query: String) -> Result<String> {
        // Implementation
    }
    
    /// Get current weather
    pub async fn get_weather(&self, location: String) -> Result<Weather> {
        // Implementation
    }
}
```

### Tool Execution Flow

1. **Definition** - Tools defined with `#[toolbox]` macro
2. **Registration** - Tools added to `ToolSet` for agent use
3. **Calling** - Model requests tool execution with structured parameters
4. **Execution** - Tool method called with validated parameters
5. **Result** - Tool result integrated back into conversation

### ToolBox Trait and ToolSet

The `ToolBox` trait is the core of the tool system. It allows any struct to expose a set of tools and a way to invoke them. The `ToolSet` then acts as a container for multiple `ToolBox` instances, providing a single point of entry for the agent to discover and execute tools.

```rust
pub trait ToolBox: Send + Sync + 'static {
    fn tools(&self) -> Vec<Tool>;
    fn invoke(&self, call: ToolCall) -> BoxFuture<Result<ToolResult, ToolError>>;
}

#[derive(Clone, Default)]
pub struct ToolSet {
    toolboxes: Vec<Arc<dyn ToolBox>>,
}

impl ToolSet {
    pub fn add_toolbox(&mut self, toolbox: impl ToolBox + Send + Sync + 'static);
    pub async fn invoke(&self, call: ToolCall) -> Result<ToolResult, ToolError>;
}
```

## Workflow System Architecture

### Node Trait and Workflow Execution

The workflow system is designed as a stateful, dynamically-linked graph. Each `Node` represents a single step. Instead of a predefined graph structure with transitions, each node's `run` method dynamically determines the `NextNode` to execute, which can be another node (`Continue`) or the end of the workflow (`End`). This allows for highly flexible and data-driven workflow paths.

```rust
// A node determines the next step in a workflow.
pub trait Node<S, O>: Send + Sync + DynClone {
    fn run(&self, context: RunContext<S>) -> BoxFuture<Result<NextNode<S, O>, WorkflowError>>;
}

// The result of a node's execution.
pub enum NextNode<S, O> {
    Continue(Box<dyn Node<S, O>>),
    End(O),
}

// A workflow is defined by its starting node and initial state.
pub struct Workflow<S, O> {
    initial_node: Box<dyn Node<S, O>>,
    context: RunContext<S>,
}

impl<S, O> Workflow<S, O> {
    pub fn new(initial_node: impl Node<S, O> + 'static, initial_state: S) -> Self;
    pub async fn run(&self) -> Result<O, WorkflowError>;
}
```

### Workflow Execution

The `Workflow::run` method orchestrates the execution:

```
Workflow::run() → initial_node.run() → NextNode::Continue(node_b) → node_b.run() → NextNode::End(output) → Result<O>
```

### Key Features

- **State Management** - A `RunContext` preserves shared state across all nodes.
- **Dynamic Routing** - Each node is responsible for determining the next step, allowing for complex, data-driven branching.
- **Type Safety** - The workflow is generic over a state type `S` and an output type `O`.
- **Composability** - Nodes are self-contained and can be reused across different workflows.

## Provider Integration Patterns

### Adapter and Conversion Layer Pattern

The core integration pattern is the **Adapter Pattern**. For each provider, `ai-ox` provides an adapter module that is responsible for bridging the gap between the generic `ai-ox` interfaces and the specific provider's SDK.

A key part of this adapter is the **Conversion Layer**. Each adapter module (`ai-ox/src/model/{provider}/`) contains a `conversion.rs` file. This file is responsible for mapping data structures between the `ai-ox` core types (e.g., `ModelRequest`) and the provider-specific types (e.g., `anthropic_ox::ChatRequest`).

```rust
// In ai-ox/src/model/anthropic/conversion.rs
use crate::model::ModelRequest;
use anthropic_ox::request::ChatRequest as AnthropicRequest;

pub fn convert_request_to_anthropic(request: ModelRequest) -> AnthropicRequest {
    // ... conversion logic ...
}
```

### Provider-Specific Optimizations

- **Gemini** - Multimodal content optimization, live session support
- **OpenRouter** - Multi-model routing, provider preferences, fallback handling

### Extensibility

The steps to add a new provider have been refined based on this adapter pattern. The goal is to keep the provider-specific crate as a simple, standalone SDK and place all the integration logic within `ai-ox`.

## Error Handling Architecture

### Structured Error Hierarchy

```rust
#[derive(thiserror::Error, Debug)]
pub enum AiOxError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),
    
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),
    
    #[error("Workflow error: {0}")]
    Workflow(#[from] WorkflowError),
}
```

### Error Propagation

- **Graceful Degradation** - Continue operation when possible
- **Context Preservation** - Error context maintained through call stack
- **Recovery Strategies** - Automatic retry and fallback mechanisms

## Streaming Architecture

### Real-time Response Streaming

The architecture supports real-time streaming for immediate user feedback:

```rust
pub enum StreamEvent {
    TextDelta(String),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Usage(Usage),
}

pub enum AgentEvent {
    Delta(StreamEvent),
    ToolExecution(ToolCall),
    Completed(Message),
    Error(AiOxError),
}
```

### Streaming Flow

```
Model Stream → Stream Events → Agent Events → User Interface
     ↓              ↓              ↓
Tool Calls → Tool Execution → Tool Results
```

## Design Principles

### 1. Provider Agnostic
- Common abstractions allow switching between providers
- Provider-specific optimizations remain available
- Unified API reduces integration complexity

### 2. Type Safety
- Strong typing throughout the system
- Compile-time validation of tool schemas
- Structured error handling

### 3. Composability
- Modular components can be combined flexibly
- Workflows compose agents and tools
- Tools can be mixed and matched per use case

### 4. Performance
- Async/await throughout for non-blocking operations
- Streaming support for real-time responsiveness
- Efficient memory usage with zero-copy where possible

### 5. Extensibility
- New providers via trait implementation
- Custom tools via macro system
- Workflow customization through Node trait

## Extension Points

### Adding New Providers

1.  **Create a Standalone Provider Crate**: Create a new crate (e.g., `my-provider-ox`). This crate should handle all API communication for the target provider but should **not** depend on `ai-ox`. It should define its own request/response structs.
2.  **Add Crate to Workspace**: Add the new crate to the `Cargo.toml` workspace members.
3.  **Create Adapter Module in `ai-ox`**: Inside `crates/ai-ox/src/model/`, create a new module for the provider (e.g., `my_provider`).
4.  **Implement the Adapter**: Within this new module:
    a.  Create a wrapper struct (e.g., `MyProviderModel`) that contains the client from `my-provider-ox`.
    b.  Create a `conversion.rs` submodule to map between `ai-ox` types and `my-provider-ox` types.
    c.  Implement the `ai_ox::model::Model` trait for your `MyProviderModel` wrapper struct, using the conversion functions.
5.  **Add Feature Flag**: Add a new feature flag for the provider in `crates/ai-ox/Cargo.toml` and make the dependency on `my-provider-ox` optional.
6.  **Update `ai-ox`**: Wire up the new module in `crates/ai-ox/src/model/mod.rs` under the new feature flag.
7.  **Add Integration Tests**: Add tests to verify the integration works as expected.

### Custom Tool Development

1. Create struct with tool methods
2. Apply `#[toolbox]` macro
3. Implement tool logic with proper error handling
4. Register with `ToolSet` for agent use

### Workflow Customization

1. Implement `Node` trait for custom workflow steps
2. Define transitions and execution logic
3. Compose into `WorkflowGraph`
4. Execute with `RunContext` for state management

This architecture provides a solid foundation for AI integration while maintaining flexibility for future enhancements and provider additions.