use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tool {
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Builder)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ToolFunction {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: None,
            strict: None,
        }
    }

    pub fn with_parameters(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Some(parameters),
            strict: None,
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

impl Tool {
    pub fn function(function: ToolFunction) -> Self {
        Self {
            r#type: "function".to_string(),
            function,
        }
    }

    pub fn function_with_params(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self::function(ToolFunction::with_parameters(name, description, parameters))
    }

    #[cfg(feature = "schema")]
    pub fn from_schema<T: schemars::JsonSchema>() -> Self {
        let schema = schemars::schema_for!(T);
        Self::function_with_params(
            std::any::type_name::<T>().split("::").last().unwrap_or("unknown").to_string(),
            schema.schema.metadata.as_ref().and_then(|m| m.description.clone()).unwrap_or_default(),
            serde_json::to_value(schema).unwrap(),
        )
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.function.strict = Some(strict);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    #[serde(untagged)]
    Function { r#type: String, function: ToolChoiceFunction },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

impl ToolChoice {
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function {
            r#type: "function".to_string(),
            function: ToolChoiceFunction {
                name: name.into(),
            },
        }
    }
}