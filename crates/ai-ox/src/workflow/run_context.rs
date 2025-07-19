use std::sync::Arc;
use tokio::sync::Mutex;

/// The context for the graph run.
///
/// This holds the state of the graph and shared dependencies, and is passed to nodes when they're run.
#[derive(Debug)]
pub struct RunContext<State, Deps> {
    /// The current state of the workflow, protected by a Mutex for safe concurrent access.
    pub state: Arc<Mutex<State>>,
    /// Shared dependencies that nodes can access.
    pub deps: Arc<Deps>,
}

impl<State, Deps> RunContext<State, Deps> {
    /// Creates a new `RunContext` with the given initial state and dependencies.
    pub fn new(state: State, deps: Deps) -> Self {
        Self {
            state: Arc::new(Mutex::new(state)),
            deps: Arc::new(deps),
        }
    }
}

impl<State, Deps> Clone for RunContext<State, Deps> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            deps: Arc::clone(&self.deps),
        }
    }
}
