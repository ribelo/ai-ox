use super::error::WorkflowError;
use super::node::{Next, Node};
use super::run_context::RunContext;

/// Represents a workflow that can be executed.
#[derive(Clone)]
pub struct Workflow<State, Deps, Output> {
    initial_node: Box<dyn Node<State, Deps, Output>>,
    context: RunContext<State, Deps>,
}

impl<State, Deps, Output> Workflow<State, Deps, Output>
where
    State: Send + Sync,
    Deps: Send + Sync,
    Output: Send + Sync,
{
    /// Creates a new workflow with an initial node, state, and dependencies.
    pub fn new(
        initial_node: impl Node<State, Deps, Output> + 'static,
        initial_state: State,
        deps: Deps,
    ) -> Self {
        Self {
            initial_node: Box::new(initial_node),
            context: RunContext::new(initial_state, deps),
        }
    }

    /// Runs the workflow to completion.
    ///
    /// It starts with the initial node and executes subsequent nodes until
    /// one of them returns `NextNode::End` or an error occurs.
    pub async fn run(&self) -> Result<Output, WorkflowError> {
        let mut current_node = dyn_clone::clone_box(&*self.initial_node);

        loop {
            // The '?' operator will automatically propagate the error if the node fails.
            let next = current_node.run(&self.context).await?;
            match next {
                Next::Continue(next_node) => {
                    current_node = next_node;
                }
                Next::End(output) => {
                    // If the workflow finishes successfully, wrap the output in Ok().
                    return Ok(output);
                }
            }
        }
    }
}
