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
//! - [`Next`]: Enum that determines the next step in execution
//! - [`RunContext`]: Shared state container accessible to all nodes
//! - [`WorkflowError`]: Error type for workflow execution failures
//!
//! # Example
//!
//! ```rust,no_run
//! use ai_ox::workflow::{Workflow, Node, Next, RunContext, WorkflowError};
//! use async_trait::async_trait;
//! use dyn_clone::DynClone;
//!
//! #[derive(Debug, Clone)]
//! struct SimpleState {
//!     counter: i32,
//! }
//!
//! #[derive(Debug, Clone)]
//! struct EmptyDeps;
//!
//! #[derive(Debug, Clone)]
//! struct IncrementNode;
//!
//! #[async_trait]
//! impl Node<SimpleState, EmptyDeps, String> for IncrementNode {
//!     async fn run(&self, context: &RunContext<SimpleState, EmptyDeps>) -> Result<Next<SimpleState, EmptyDeps, String>, WorkflowError> {
//!         let mut state = context.state.lock().await;
//!         state.counter += 1;
//!         Ok(Next::End(format!("Counter: {}", state.counter)))
//!     }
//! }
//!
//! dyn_clone::clone_trait_object!(IncrementNode);
//!
//! // Usage in an async context:
//! async fn example() -> Result<(), WorkflowError> {
//!     let workflow = Workflow::new(IncrementNode, SimpleState { counter: 0 }, EmptyDeps);
//!     let result = workflow.run().await?;
//!     println!("Result: {}", result); // Prints: "Result: Counter: 1"
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod graph;
pub mod node;
pub mod run_context;

pub use error::WorkflowError;
pub use graph::Workflow;
pub use node::{Next, Node};
pub use run_context::RunContext;
