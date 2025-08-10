use serde::{Deserialize, Serialize};

/// A tool that can be called by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The type of tool (usually "function")
    pub r#type: String,
    
    /// Function definition
    pub function: Function,
}

/// Function definition for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Name of the function
    pub name: String,
    
    /// Description of what the function does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    
    /// JSON schema for the function parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    
    /// Whether the function is strict (exact parameter matching)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl Tool {
    /// Create a new function tool
    pub fn function(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            r#type: "function".to_string(),
            function: Function {
                name: name.into(),
                description: Some(description.into()),
                parameters: None,
                strict: None,
            },
        }
    }
    
    /// Create a function tool with parameters schema
    pub fn function_with_params(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            r#type: "function".to_string(),
            function: Function {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
                strict: None,
            },
        }
    }
    
    /// Set whether the function parameters should be strictly validated
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.function.strict = Some(strict);
        self
    }
}

impl Function {
    /// Create a new function
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters: None,
            strict: None,
        }
    }
    
    /// Set the parameters schema
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }
    
    /// Set strict parameter validation
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

#[cfg(feature = "schema")]
impl Tool {
    /// Create a tool from a type that implements JsonSchema
    pub fn from_schema<T: schemars::JsonSchema>() -> Self {
        let schema = schemars::schema_for!(T);
        Self {
            r#type: "function".to_string(),
            function: Function {
                name: std::any::type_name::<T>().split("::").last().unwrap_or("unknown").to_string(),
                description: schema.schema.metadata.as_ref().and_then(|m| m.description.clone()),
                parameters: Some(serde_json::to_value(schema).unwrap()),
                strict: None,
            },
        }
    }
}

/// Tool choice options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// No tool calling
    None,
    /// Automatically choose whether to call tools
    Auto,
    /// Force a specific tool to be called
    Required,
    /// Specific function to call
    Function { function: FunctionChoice },
}

/// Specific function choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    /// Name of the function to call
    pub name: String,
}

impl ToolChoice {
    /// Don't call any tools
    pub fn none() -> Self {
        Self::None
    }
    
    /// Automatically decide whether to call tools
    pub fn auto() -> Self {
        Self::Auto
    }
    
    /// Require that a tool is called
    pub fn required() -> Self {
        Self::Required
    }
    
    /// Force a specific function to be called
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function {
            function: FunctionChoice {
                name: name.into(),
            },
        }
    }
}