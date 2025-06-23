use super::error::WorkflowError;
use super::run_context::RunContext;
use dyn_clone::DynClone;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;

/// Represents the next step in a workflow.
pub enum NextNode<S, O> {
    /// Continue to the next node instance.
    Continue(Box<dyn Node<S, O>>),
    /// End the workflow with a final output.
    End(O),
}

/// A node in the workflow graph.
///
/// Nodes are the fundamental units of execution in a workflow. Each node
/// performs a specific task and determines the next node to execute.
pub trait Node<S, O>: Send + Sync + DynClone {
    /// Executes the business logic of the node.
    ///
    /// This async function takes the shared run context and returns the next
    /// step in the workflow, or an error if the node's execution fails.
    fn run(&self, context: RunContext<S>) -> BoxFuture<Result<NextNode<S, O>, WorkflowError>>;
}

dyn_clone::clone_trait_object!(<S, O> Node<S, O>);
