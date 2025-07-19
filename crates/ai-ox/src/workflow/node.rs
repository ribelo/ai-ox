use super::error::WorkflowError;
use super::run_context::RunContext;
use async_trait::async_trait;
use dyn_clone::DynClone;

/// Represents the next step in a workflow.
pub enum Next<State, Deps, Output> {
    /// Continue to the next node instance.
    Continue(Box<dyn Node<State, Deps, Output>>),
    /// End the workflow with a final output.
    End(Output),
}

/// A node in the workflow graph.
///
/// Nodes are the fundamental units of execution in a workflow. Each node
/// performs a specific task and determines the next node to execute.
#[async_trait]
pub trait Node<State, Deps, Output>: Send + Sync + DynClone {
    /// Executes the business logic of the node.
    ///
    /// This async function takes the shared run context and returns the next
    /// step in the workflow, or an error if the node's execution fails.
    async fn run(
        &self,
        context: &RunContext<State, Deps>,
    ) -> Result<Next<State, Deps, Output>, WorkflowError>;
}

impl<State, Deps, Output, N> From<N> for Next<State, Deps, Output>
where
    N: Node<State, Deps, Output> + 'static,
{
    fn from(node: N) -> Self {
        Next::Continue(Box::new(node))
    }
}

dyn_clone::clone_trait_object!(<State, Deps, Output> Node<State, Deps, Output>);
