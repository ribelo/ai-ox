use super::{Tool, ToolBox, ToolError, ToolHooks, ToolUse};
use futures_util::future::BoxFuture;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A container that holds multiple toolboxes and provides a unified interface
/// for tool discovery and invocation.
#[derive(Clone, Default)]
pub struct ToolSet {
    toolboxes: Vec<Arc<dyn ToolBox>>,
}

impl std::fmt::Debug for ToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSet")
            .field("toolboxes_count", &self.toolboxes.len())
            .field("tools", &self.get_all_tools())
            .finish()
    }
}

impl ToolSet {
    /// Creates a new empty ToolSet.
    pub fn new() -> Self {
        Self {
            toolboxes: Vec::new(),
        }
    }

    /// Adds a toolbox to this set.
    ///
    /// The provided toolbox will be wrapped in an `Arc` internally, so the caller
    /// does not need to manage the `Arc` themselves. If you need to share a toolbox
    /// instance across multiple sets, wrap it in an `Arc` before adding it.
    pub fn add_toolbox(&mut self, toolbox: impl ToolBox + 'static) {
        self.toolboxes.push(Arc::new(toolbox));
    }

    /// Adds a toolbox to this set using a builder pattern.
    pub fn with_toolbox(mut self, toolbox: impl ToolBox + 'static) -> Self {
        self.add_toolbox(toolbox);
        self
    }

    /// Returns all tools from all toolboxes in this set.
    pub fn get_all_tools(&self) -> Vec<Tool> {
        let mut all_tools = Vec::new();

        for toolbox in &self.toolboxes {
            all_tools.extend(toolbox.tools());
        }

        all_tools
    }

    /// Finds the toolbox that contains the function with the given name.
    fn find_toolbox_for_function(&self, name: &str) -> Option<&Arc<dyn ToolBox>> {
        self.toolboxes
            .iter()
            .find(|toolbox| toolbox.has_function(name))
    }

    /// Checks if any toolbox in this set has a function with the given name.
    pub fn has_function(&self, name: &str) -> bool {
        self.find_toolbox_for_function(name).is_some()
    }

    /// Invokes a tool function by finding the appropriate toolbox and
    /// delegating the call to it.
    pub async fn invoke(&self, call: ToolUse) -> Result<crate::content::Part, ToolError> {
        let toolbox = self
            .find_toolbox_for_function(&call.name)
            .ok_or_else(|| ToolError::not_found(&call.name))?;

        toolbox.invoke(call).await
    }

    /// Invokes a tool function with hooks for dangerous operations.
    pub async fn invoke_with_hooks(
        &self,
        call: ToolUse,
        hooks: ToolHooks,
    ) -> Result<crate::content::Part, ToolError> {
        let toolbox = self
            .find_toolbox_for_function(&call.name)
            .ok_or_else(|| ToolError::not_found(&call.name))?;

        toolbox.invoke_with_hooks(call, hooks).await
    }

    /// Checks if the given function name is considered dangerous by any toolbox.
    pub fn is_dangerous_function(&self, name: &str) -> bool {
        self.toolboxes
            .iter()
            .any(|toolbox| toolbox.dangerous_functions().contains(&name))
    }

    /// Returns all dangerous function names from all toolboxes.
    ///
    /// This aggregates dangerous functions across all toolboxes in this set,
    /// which can be useful for UI, logging, or batch approval scenarios.
    pub fn get_all_dangerous_functions(&self) -> Vec<&str> {
        self.toolboxes
            .iter()
            .flat_map(|toolbox| toolbox.dangerous_functions().iter().copied())
            .collect()
    }
}

impl ToolBox for ToolSet {
    fn tools(&self) -> Vec<Tool> {
        self.get_all_tools()
    }

    fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<crate::content::Part, ToolError>> {
        Box::pin(async move { ToolSet::invoke(self, call).await })
    }

    fn invoke_with_hooks(
        &self,
        call: ToolUse,
        hooks: ToolHooks,
    ) -> BoxFuture<'_, Result<crate::content::Part, ToolError>> {
        Box::pin(async move { ToolSet::invoke_with_hooks(self, call, hooks).await })
    }

    fn dangerous_functions(&self) -> &[&str] {
        // ToolSet doesn't have its own dangerous functions -
        // it delegates to individual toolboxes
        &[]
    }

    fn has_function(&self, name: &str) -> bool {
        ToolSet::has_function(self, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{FunctionMetadata, Tool};
    use futures_util::FutureExt;
    use serde_json::json;
    use std::sync::Arc;

    // Mock toolbox for testing
    #[derive(Debug)]
    struct MockToolBox {
        function_name: String,
    }

    impl MockToolBox {
        fn new(function_name: impl Into<String>) -> Self {
            Self {
                function_name: function_name.into(),
            }
        }
    }

    impl ToolBox for MockToolBox {
        fn tools(&self) -> Vec<Tool> {
            vec![Tool::FunctionDeclarations(vec![FunctionMetadata {
                name: self.function_name.clone(),
                description: Some(format!("Mock function {}", self.function_name)),
                parameters: json!({"type": "object", "properties": {}}),
            }])]
        }

        fn invoke(&self, call: ToolUse) -> BoxFuture<'_, Result<crate::content::Part, ToolError>> {
            let function_name = self.function_name.clone();
            async move {
                if call.name == function_name {
                    Ok(crate::content::Part::ToolResult {
                        id: call.id,
                        name: call.name,
                        parts: vec![],
                        ext: BTreeMap::new(),
                    })
                } else {
                    Err(ToolError::not_found(call.name))
                }
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn test_empty_toolset() {
        let toolset = ToolSet::new();
        assert!(!toolset.has_function("any_function"));
        assert!(toolset.get_all_tools().is_empty());

        let call = ToolUse::new("1", "missing_function", json!({}));
        let result = toolset.invoke(call).await;
        assert!(matches!(result, Err(ToolError::NotFound { .. })));
    }

    #[tokio::test]
    async fn test_single_toolbox() {
        let mut toolset = ToolSet::new();
        toolset.add_toolbox(MockToolBox::new("test_function"));

        assert!(toolset.has_function("test_function"));
        assert!(!toolset.has_function("missing_function"));

        let tools = toolset.get_all_tools();
        assert_eq!(tools.len(), 1);

        // Test successful invocation
        let call = ToolUse::new("1", "test_function", json!({}));
        let result = toolset.invoke(call).await;
        assert!(result.is_ok());

        // Test missing function
        let call = ToolUse::new("2", "missing_function", json!({}));
        let result = toolset.invoke(call).await;
        assert!(matches!(result, Err(ToolError::NotFound { .. })));
    }

    #[tokio::test]
    async fn test_multiple_toolboxes() {
        let toolset = ToolSet::new()
            .with_toolbox(MockToolBox::new("function_a"))
            .with_toolbox(MockToolBox::new("function_b"));

        assert!(toolset.has_function("function_a"));
        assert!(toolset.has_function("function_b"));
        assert!(!toolset.has_function("function_c"));

        let tools = toolset.get_all_tools();
        assert_eq!(tools.len(), 2);

        // Test invocation of function from first toolbox
        let call = ToolUse::new("1", "function_a", json!({}));
        let result = toolset.invoke(call).await;
        assert!(result.is_ok());

        // Test invocation of function from second toolbox
        let call = ToolUse::new("2", "function_b", json!({}));
        let result = toolset.invoke(call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_shared_toolbox_with_arc() {
        let shared_toolbox = Arc::new(MockToolBox::new("shared_function"));
        let mut set1 = ToolSet::new();
        set1.add_toolbox(shared_toolbox.clone());

        let mut set2 = ToolSet::new();
        set2.add_toolbox(shared_toolbox); // Move the last Arc

        assert!(set1.has_function("shared_function"));
        assert!(set2.has_function("shared_function"));

        // Both toolsets should be able to invoke the shared function
        let call1 = ToolUse::new("1", "shared_function", json!({}));
        assert!(set1.invoke(call1).await.is_ok());

        let call2 = ToolUse::new("2", "shared_function", json!({}));
        assert!(set2.invoke(call2).await.is_ok());
    }
}
