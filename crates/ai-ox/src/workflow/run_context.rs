use std::sync::Arc;
use tokio::sync::Mutex;

/// The context for the graph run.
///
/// This holds the state of the graph and is passed to nodes when they're run.
#[derive(Debug)]
pub struct RunContext<S> {
    /// The current state of the workflow, protected by a Mutex for safe concurrent access.
    pub state: Arc<Mutex<S>>,
}

impl<S> RunContext<S> {
    /// Creates a new `RunContext` with the given initial state.
    pub fn new(state: S) -> Self {
        Self {
            state: Arc::new(Mutex::new(state)),
        }
    }
}

impl<S> Clone for RunContext<S> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}
