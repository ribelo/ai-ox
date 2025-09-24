use serde::{Deserialize, Serialize};

use crate::tool::{ToolBox, error::FunctionCallError};

use super::{PromptFeedback, ResponseCandidate, usage::UsageMetadata};
use crate::content::{Content, FunctionCall, Part, Role};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<ResponseCandidate>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

impl GenerateContentResponse {
    #[must_use]
    pub fn content(&self) -> Vec<&Content> {
        self.candidates.iter().map(|c| &c.content).collect()
    }

    pub fn content_owned(&self) -> Vec<Content> {
        self.candidates.iter().map(|c| c.content.clone()).collect()
    }

    #[must_use]
    pub fn last_content(&self) -> Option<&Content> {
        self.candidates.first().map(|c| &c.content)
    }

    #[must_use]
    pub fn last_content_owned(&self) -> Option<Content> {
        self.candidates.first().map(|c| &c.content).cloned()
    }

    pub fn function_calls(&self) -> impl Iterator<Item = &FunctionCall> + '_ {
        self.last_content()
            .into_iter()
            .flat_map(|c| c.parts().iter().filter_map(|p| p.as_function_call()))
    }

    // Changed signature to use Arc<dyn ToolBox>
    pub async fn invoke_functions(
        &self,
        tools: impl ToolBox + Clone,
    ) -> Result<Option<Content>, FunctionCallError> {
        // Get function calls to execute
        let function_calls: Vec<FunctionCall> = self.function_calls().cloned().collect();

        if function_calls.is_empty() {
            return Ok(None); // No functions to call
        }

        // Use a JoinSet to manage concurrent tasks
        let mut join_set = tokio::task::JoinSet::new();

        // Spawn tasks for each function call invocation
        // Assuming ToolBox::invoke takes an owned FunctionCall returns Result<Content, Error>
        // and ToolBox is Send + Sync + 'static (or wrapped appropriately, now using Arc)
        for fc in function_calls {
            let tools_clone = tools.clone(); // Clone the Arc, not the ToolBox
            join_set.spawn(async move {
                tools_clone.invoke(fc).await // Execute the tool invocation
            });
        }

        // Collect results from completed tasks
        let mut parts: Vec<Part> = Vec::new();
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(Ok(fn_response)) => {
                    // Task completed successfully, and tool invocation succeeded
                    // Extract owned Parts from the resulting Content.
                    parts.push(fn_response.into());
                }
                Ok(Err(tool_error)) => return Err(tool_error),
                Err(join_error) => {
                    return Err(FunctionCallError::ExecutionFailed(join_error.to_string()));
                }
            }
        }

        // If no parts were successfully generated, return None
        // Otherwise, build the final Content object with role User
        if parts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Content::new(Role::User, parts)))
        }
    }
}
