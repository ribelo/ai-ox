use serde::{Deserialize, Serialize};

/// Configuration for specifying tool use in the request.
///
/// Currently, this only supports configuring function calling.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)] // Added PartialEq, Eq
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Optional. Function calling configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

impl ToolConfig {
    /// Creates a new, empty `ToolConfig`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the function calling configuration.
    #[must_use]
    pub fn function_calling_config(
        mut self,
        function_calling_config: FunctionCallingConfig,
    ) -> Self {
        self.function_calling_config = Some(function_calling_config);
        self
    }

    /// A convenience method to set the function calling mode directly.
    ///
    /// This will create a default `FunctionCallingConfig` if one doesn't exist.
    #[must_use]
    pub fn mode(mut self, mode: Mode) -> Self {
        let mut fcc = self.function_calling_config.unwrap_or_default();
        fcc.mode = Some(mode); // Use direct field access for clarity inside impl
        self.function_calling_config = Some(fcc);
        self
    }

    /// A convenience method to set the allowed function names directly.
    ///
    /// This requires the `Mode` to be `ANY`.
    /// This will create a default `FunctionCallingConfig` if one doesn't exist.
    ///
    /// # Panics
    ///
    /// While this method doesn't panic itself, using allowed names without setting
    /// the mode to `ANY` might lead to unexpected behavior or API errors.
    #[must_use]
    pub fn allowed_function_names(
        mut self,
        allowed_function_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let mut fcc = self.function_calling_config.unwrap_or_default();
        fcc.allowed_function_names =
            Some(allowed_function_names.into_iter().map(Into::into).collect());
        self.function_calling_config = Some(fcc);
        self
    }
}

/// Configuration for specifying function calling behavior.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)] // Added PartialEq, Eq
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// Optional. Specifies the mode in which function calling should execute.
    /// If unspecified, the default value will be set to AUTO by the API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<Mode>, // Made fields pub for direct access if needed, builders preferred
    /// Optional. A set of function names that, when provided, limits the functions the model
    /// will call.
    ///
    /// This should only be set when the Mode is `ANY`. Function names should match
    /// `FunctionDeclaration.name`. With mode set to `ANY`, the model will predict a
    /// function call from the set of function names provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>, // Made fields pub
}

impl FunctionCallingConfig {
    /// Creates a new `FunctionCallingConfig` with default values (all fields `None`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the function calling mode (e.g., `AUTO`, `ANY`, `NONE`).
    #[must_use]
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Sets the allowed function names, restricting the model to only call functions
    /// from this list when the mode is `ANY`.
    #[must_use]
    pub fn allowed_function_names(
        mut self,
        allowed_function_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_function_names =
            Some(allowed_function_names.into_iter().map(Into::into).collect());
        self
    }
}

/// Defines the execution behavior for function calling.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Mode {
    /// Unspecified function calling mode. The API will default to `AUTO`.
    /// It's generally recommended to explicitly set the mode rather than using this.
    ModeUnspecified, // Note: Consider if this variant is truly needed or if Option<Mode> is better. Kept for parity with schema.
    /// Default model behavior. The model decides whether to predict a function call
    /// or a natural language response.
    #[default] // Auto is the typical default behavior
    Auto,
    /// Constrains the model to always predict a function call.
    /// If `allowed_function_names` is set in `FunctionCallingConfig`, the prediction
    /// is limited to one of those functions. Otherwise, it can be any function
    /// provided in the tool's `function_declarations`.
    Any,
    /// Disables function calling. The model will not predict any function calls,
    /// behaving as if no function declarations were provided.
    None,
}
