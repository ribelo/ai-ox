use super::error::WorkflowError;
use super::node::{NextNode, Node};
use super::run_context::RunContext;

/// Represents a workflow that can be executed.
#[derive(Clone)]
pub struct Workflow<S, O> {
    initial_node: Box<dyn Node<S, O>>,
    context: RunContext<S>,
}

impl<S, O> Workflow<S, O>
where
    S: Send + Sync,
    O: Send + Sync,
{
    /// Creates a new workflow with an initial node and state.
    pub fn new(initial_node: impl Node<S, O> + 'static, initial_state: S) -> Self {
        Self {
            initial_node: Box::new(initial_node),
            context: RunContext::new(initial_state),
        }
    }

    /// Runs the workflow to completion.
    ///
    /// It starts with the initial node and executes subsequent nodes until
    /// one of them returns `NextNode::End` or an error occurs.
    pub async fn run(&self) -> Result<O, WorkflowError> {
        let mut current_node = dyn_clone::clone_box(&*self.initial_node);

        loop {
            // The '?' operator will automatically propagate the error if the node fails.
            let next = current_node.run(self.context.clone()).await?;
            match next {
                NextNode::Continue(next_node) => {
                    current_node = next_node;
                }
                NextNode::End(output) => {
                    // If the workflow finishes successfully, wrap the output in Ok().
                    return Ok(output);
                }
            }
        }
    }
}
