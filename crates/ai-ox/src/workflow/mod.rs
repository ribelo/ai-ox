//! Finite State Machine (FSM) workflow system for orchestrating complex, multi-step processes.
//!
//! This module provides a robust, type-safe FSM implementation that enables the execution
//! of workflows defined as a graph of nodes. Each node performs a specific task and
//! determines the next step in the workflow based on its execution results.
//!
//! # Core Components
//!
//! - [`Workflow`]: The main workflow runner that manages execution
//! - [`Node`]: Trait for defining workflow steps with custom business logic
//! - [`NextNode`]: Enum that determines the next step in execution
//! - [`RunContext`]: Shared state container accessible to all nodes
//! - [`WorkflowError`]: Error type for workflow execution failures
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_ox::workflow::{Workflow, Node, NextNode, RunContext, WorkflowError};
//! use std::future::Future;
//! use std::pin::Pin;
//!
//! #[derive(Debug, Clone)]
//! struct SimpleState {
//!     counter: i32,
//! }
//!
//! #[derive(Debug, Clone)]
//! struct IncrementNode;
//!
//! impl Node<SimpleState, String> for IncrementNode {
//!     fn run(&self, context: RunContext<SimpleState>)
//!         -> Pin<Box<dyn Future<Output = Result<NextNode<SimpleState, String>, WorkflowError>> + Send + '_>>
//!     {
//!         Box::pin(async move {
//!             let mut state = context.state.lock().await;
//!             state.counter += 1;
//!             Ok(NextNode::End(format!("Counter: {}", state.counter)))
//!         })
//!     }
//! }
//!
//! // Usage in an async context:
//! async fn example() -> Result<(), WorkflowError> {
//!     let workflow = Workflow::new(IncrementNode, SimpleState { counter: 0 });
//!     let result = workflow.run().await?;
//!     println!("Result: {}", result); // Prints: "Result: Counter: 1"
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod graph;
pub mod node;
pub mod run_context;

#[cfg(test)]
pub mod tests;

pub use error::WorkflowError;
pub use graph::Workflow;
pub use node::{NextNode, Node};
pub use run_context::RunContext;
