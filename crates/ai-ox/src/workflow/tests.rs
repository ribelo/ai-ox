//! Integration tests for the workflow system.
//!
//! These tests demonstrate the workflow system's ability to:
//! - Execute workflows with multiple nodes
//! - Handle errors gracefully
//! - Manage state across nodes

use super::{
    error::WorkflowError, graph::Workflow, node::NextNode, node::Node, run_context::RunContext,
};
use std::future::Future;
use std::pin::Pin;
use thiserror::Error;

/// Test state that holds a counter value.
#[derive(Debug, Clone)]
pub struct CounterState {
    pub counter: i32,
}

impl CounterState {
    pub fn new(initial_value: i32) -> Self {
        Self {
            counter: initial_value,
        }
    }
}

/// Custom error type for testing error handling in workflows.
#[derive(Debug, Error)]
pub enum CounterError {
    #[error("Counter value is too high: {value}")]
    ValueTooHigh { value: i32 },
    #[error("Counter value is negative: {value}")]
    NegativeValue { value: i32 },
}

/// A node that increments the counter by a specified amount.
#[derive(Clone)]
pub struct AddNode {
    pub increment: i32,
    pub next_node: Option<Box<dyn Node<CounterState, String>>>,
}

impl AddNode {
    pub fn new(increment: i32) -> Self {
        Self {
            increment,
            next_node: None,
        }
    }

    pub fn with_next(mut self, next_node: impl Node<CounterState, String> + 'static) -> Self {
        self.next_node = Some(Box::new(next_node));
        self
    }
}

impl Node<CounterState, String> for AddNode {
    fn run(
        &self,
        context: RunContext<CounterState>,
    ) -> Pin<
        Box<dyn Future<Output = Result<NextNode<CounterState, String>, WorkflowError>> + Send + '_>,
    > {
        Box::pin(async move {
            let mut state = context.state.lock().await;
            state.counter += self.increment;

            match &self.next_node {
                Some(next) => Ok(NextNode::Continue(dyn_clone::clone_box(&**next))),
                None => Ok(NextNode::End(format!(
                    "Final counter value: {}",
                    state.counter
                ))),
            }
        })
    }
}

/// A node that checks the counter value and either continues or ends the workflow.
#[derive(Clone)]
pub struct CheckNode {
    pub max_value: i32,
    pub continue_node: Option<Box<dyn Node<CounterState, String>>>,
    pub should_error_on_exceed: bool,
}

impl CheckNode {
    pub fn new(max_value: i32) -> Self {
        Self {
            max_value,
            continue_node: None,
            should_error_on_exceed: false,
        }
    }

    pub fn with_continue(
        mut self,
        continue_node: impl Node<CounterState, String> + 'static,
    ) -> Self {
        self.continue_node = Some(Box::new(continue_node));
        self
    }

    pub fn with_error_on_exceed(mut self) -> Self {
        self.should_error_on_exceed = true;
        self
    }
}

impl Node<CounterState, String> for CheckNode {
    fn run(
        &self,
        context: RunContext<CounterState>,
    ) -> Pin<
        Box<dyn Future<Output = Result<NextNode<CounterState, String>, WorkflowError>> + Send + '_>,
    > {
        Box::pin(async move {
            let state = context.state.lock().await;
            let current_value = state.counter;

            if current_value > self.max_value {
                if self.should_error_on_exceed {
                    return Err(WorkflowError::node_execution_failed(
                        CounterError::ValueTooHigh {
                            value: current_value,
                        },
                    ));
                } else {
                    return Ok(NextNode::End(format!(
                        "Counter exceeded maximum ({}): {}",
                        self.max_value, current_value
                    )));
                }
            }

            if current_value < 0 {
                return Err(WorkflowError::node_execution_failed(
                    CounterError::NegativeValue {
                        value: current_value,
                    },
                ));
            }

            match &self.continue_node {
                Some(next) => Ok(NextNode::Continue(dyn_clone::clone_box(&**next))),
                None => Ok(NextNode::End(format!(
                    "Counter is within bounds: {current_value}"
                ))),
            }
        })
    }
}

/// A node that conditionally branches based on counter value.
#[derive(Clone)]
pub struct ConditionalNode {
    pub threshold: i32,
    pub if_true: Option<Box<dyn Node<CounterState, String>>>,
    pub if_false: Option<Box<dyn Node<CounterState, String>>>,
}

impl ConditionalNode {
    pub fn new(threshold: i32) -> Self {
        Self {
            threshold,
            if_true: None,
            if_false: None,
        }
    }

    pub fn with_true_branch(mut self, node: impl Node<CounterState, String> + 'static) -> Self {
        self.if_true = Some(Box::new(node));
        self
    }

    pub fn with_false_branch(mut self, node: impl Node<CounterState, String> + 'static) -> Self {
        self.if_false = Some(Box::new(node));
        self
    }
}

impl Node<CounterState, String> for ConditionalNode {
    fn run(
        &self,
        context: RunContext<CounterState>,
    ) -> Pin<
        Box<dyn Future<Output = Result<NextNode<CounterState, String>, WorkflowError>> + Send + '_>,
    > {
        Box::pin(async move {
            let state = context.state.lock().await;
            let current_value = state.counter;

            if current_value >= self.threshold {
                match &self.if_true {
                    Some(node) => Ok(NextNode::Continue(dyn_clone::clone_box(&**node))),
                    None => Ok(NextNode::End(format!(
                        "Counter {} >= threshold {}",
                        current_value, self.threshold
                    ))),
                }
            } else {
                match &self.if_false {
                    Some(node) => Ok(NextNode::Continue(dyn_clone::clone_box(&**node))),
                    None => Ok(NextNode::End(format!(
                        "Counter {} < threshold {}",
                        current_value, self.threshold
                    ))),
                }
            }
        })
    }
}

#[cfg(test)]
mod workflow_tests {
    use super::*;

    #[tokio::test]
    async fn test_successful_workflow_execution_with_multiple_nodes() {
        // Create a workflow with multiple nodes that increment a counter
        let add_node1 = AddNode::new(5).with_next(
            AddNode::new(3).with_next(CheckNode::new(20).with_continue(AddNode::new(2))),
        );

        let workflow = Workflow::new(add_node1, CounterState::new(0));

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Final counter value: 10"); // 0 + 5 + 3 + 2 = 10
    }

    #[tokio::test]
    async fn test_workflow_state_management_across_nodes() {
        // Test that state changes persist across multiple node executions
        let workflow = Workflow::new(
            AddNode::new(1)
                .with_next(AddNode::new(2).with_next(AddNode::new(3).with_next(AddNode::new(4)))),
            CounterState::new(10),
        );

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Final counter value: 20"); // 10 + 1 + 2 + 3 + 4 = 20
    }

    #[tokio::test]
    async fn test_workflow_with_conditional_branching() {
        // Test workflow with conditional logic
        let true_branch = AddNode::new(100);
        let false_branch = AddNode::new(1);

        let conditional_node = ConditionalNode::new(5)
            .with_true_branch(true_branch)
            .with_false_branch(false_branch);

        let workflow = Workflow::new(
            AddNode::new(3).with_next(conditional_node),
            CounterState::new(0),
        );

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Final counter value: 4"); // 0 + 3 = 3, 3 < 5, so add 1: 3 + 1 = 4
    }

    #[tokio::test]
    async fn test_workflow_handles_check_node_early_termination() {
        // Test that CheckNode can terminate workflow early
        let workflow = Workflow::new(
            AddNode::new(15).with_next(CheckNode::new(10)), // This should terminate early since 15 > 10
            CounterState::new(0),
        );

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Counter exceeded maximum (10): 15");
    }

    #[tokio::test]
    async fn test_workflow_error_handling_with_counter_too_high() {
        // Test error handling when counter exceeds maximum with error flag
        let workflow = Workflow::new(
            AddNode::new(25).with_next(CheckNode::new(10).with_error_on_exceed()),
            CounterState::new(0),
        );

        let result = workflow.run().await;

        assert!(result.is_err());
        let error = result.unwrap_err();

        match error {
            WorkflowError::NodeExecutionFailed { source } => {
                let counter_error = source.downcast_ref::<CounterError>().unwrap();
                match counter_error {
                    CounterError::ValueTooHigh { value } => {
                        assert_eq!(*value, 25);
                    }
                    _ => panic!("Expected ValueTooHigh error"),
                }
            }
        }
    }

    #[tokio::test]
    async fn test_workflow_error_handling_with_negative_counter() {
        // Test error handling when counter becomes negative
        let workflow = Workflow::new(
            AddNode::new(-10).with_next(CheckNode::new(10)),
            CounterState::new(5),
        );

        let result = workflow.run().await;

        assert!(result.is_err());
        let error = result.unwrap_err();

        match error {
            WorkflowError::NodeExecutionFailed { source } => {
                let counter_error = source.downcast_ref::<CounterError>().unwrap();
                match counter_error {
                    CounterError::NegativeValue { value } => {
                        assert_eq!(*value, -5); // 5 + (-10) = -5
                    }
                    _ => panic!("Expected NegativeValue error"),
                }
            }
        }
    }

    #[tokio::test]
    async fn test_workflow_with_complex_state_transitions() {
        // Test a more complex workflow with multiple decision points
        let workflow = Workflow::new(
            AddNode::new(3).with_next(
                CheckNode::new(50).with_continue(
                    AddNode::new(7).with_next(
                        ConditionalNode::new(15)
                            .with_true_branch(AddNode::new(5))
                            .with_false_branch(AddNode::new(20)),
                    ),
                ),
            ),
            CounterState::new(2),
        );

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        // 2 + 3 = 5, check passes, 5 + 7 = 12, 12 < 15, so add 20: 12 + 20 = 32
        assert_eq!(output, "Final counter value: 32");
    }

    #[tokio::test]
    async fn test_single_node_workflow() {
        // Test the simplest possible workflow with just one node
        let workflow = Workflow::new(AddNode::new(42), CounterState::new(0));

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Final counter value: 42");
    }

    #[tokio::test]
    async fn test_workflow_preserves_state_consistency() {
        // Test that concurrent access to state is handled correctly
        let workflow =
            Workflow::new(
                AddNode::new(1).with_next(AddNode::new(1).with_next(
                    AddNode::new(1).with_next(AddNode::new(1).with_next(AddNode::new(1))),
                )),
                CounterState::new(0),
            );

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output, "Final counter value: 5");
    }

    #[tokio::test]
    async fn test_conditional_node_threshold_boundary() {
        // Test conditional node behavior at threshold boundary
        let workflow = Workflow::new(
            AddNode::new(10).with_next(
                ConditionalNode::new(10) // exactly at threshold
                    .with_true_branch(AddNode::new(100))
                    .with_false_branch(AddNode::new(1)),
            ),
            CounterState::new(0),
        );

        let result = workflow.run().await;

        assert!(result.is_ok());
        let output = result.unwrap();
        // 0 + 10 = 10, 10 >= 10 (threshold), so take true branch: 10 + 100 = 110
        assert_eq!(output, "Final counter value: 110");
    }
}
