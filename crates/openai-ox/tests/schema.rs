//! Schema-related tests
//!
//! These tests require the "schema" feature to be enabled.

#[cfg(feature = "schema")]
#[cfg(test)]
mod tests {
    use openai_ox::{Tool, Function};
    use schemars::{JsonSchema, schema_for};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct WeatherQuery {
        /// The city and state, e.g. San Francisco, CA
        location: String,
        /// Temperature unit (celsius or fahrenheit)
        #[serde(default = "default_unit")]
        unit: String,
    }

    fn default_unit() -> String {
        "celsius".to_string()
    }

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct CalculatorInput {
        /// The mathematical expression to evaluate
        expression: String,
        /// Number of decimal places in the result
        #[serde(default)]
        precision: u8,
    }

    #[test]
    fn test_schema_generation() {
        let schema = schema_for!(WeatherQuery);

        assert_eq!(schema.schema.object().properties.len(), 2);
        assert!(schema.schema.object().properties.contains_key("location"));
        assert!(schema.schema.object().properties.contains_key("unit"));

        // Location should be required
        let required = &schema.schema.object().required;
        assert!(required.contains(&"location".to_string()));
        assert!(!required.contains(&"unit".to_string())); // unit has default
    }

    #[test]
    fn test_tool_from_schema() {
        let tool = Tool::from_schema::<WeatherQuery>();

        assert_eq!(tool.r#type, "function");
        assert_eq!(tool.function.name, "WeatherQuery");
        assert!(tool.function.parameters.is_some());

        let params = tool.function.parameters.unwrap();
        assert!(params.is_object());

        let obj = params.as_object().unwrap();
        assert!(obj.contains_key("type"));
        assert!(obj.contains_key("properties"));
        assert_eq!(obj["type"], "object");
    }

    #[test]
    fn test_function_with_schema() {
        let schema = schema_for!(CalculatorInput);
        let schema_value = serde_json::to_value(schema).unwrap();

        let function = Function::new("calculate", "Evaluate mathematical expressions")
            .with_parameters(schema_value.clone());

        assert_eq!(function.name, "calculate");
        assert_eq!(function.description, Some("Evaluate mathematical expressions".to_string()));
        assert!(function.parameters.is_some());
        assert_eq!(function.parameters.unwrap(), schema_value);
    }

    #[test]
    fn test_tool_with_strict_schema() {
        let tool = Tool::from_schema::<WeatherQuery>().with_strict(true);

        assert_eq!(tool.function.strict, Some(true));
    }

    #[test]
    fn test_manual_schema_creation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["query"]
        });

        let tool = Tool::function_with_params(
            "search",
            "Search for information",
            schema
        );

        assert_eq!(tool.function.name, "search");
        assert!(tool.function.parameters.is_some());

        let params = tool.function.parameters.unwrap();
        assert_eq!(params["type"], "object");
        assert!(params["properties"].is_object());
        assert_eq!(params["required"], json!(["query"]));
    }

    #[test]
    fn test_schema_validation_structure() {
        let schema = schema_for!(WeatherQuery);
        let schema_value = serde_json::to_value(schema).unwrap();

        // Verify the schema has the expected OpenAI function format
        assert!(schema_value.is_object());
        let obj = schema_value.as_object().unwrap();

        // Should have schema metadata
        assert!(obj.contains_key("$schema"));
        assert!(obj.contains_key("title"));

        // The actual schema should be properly structured
        if let Some(definitions) = obj.get("definitions") {
            assert!(definitions.is_object());
        }
    }

    #[test]
    fn test_nested_schema() {
        #[derive(Debug, Serialize, Deserialize, JsonSchema)]
        struct Address {
            street: String,
            city: String,
            country: String,
        }

        #[derive(Debug, Serialize, Deserialize, JsonSchema)]
        struct Person {
            name: String,
            age: u32,
            address: Address,
        }

        let schema = schema_for!(Person);
        let schema_value = serde_json::to_value(schema).unwrap();

        let tool = Tool::function_with_params(
            "create_person",
            "Create a person record",
            schema_value
        );

        assert_eq!(tool.function.name, "create_person");
        assert!(tool.function.parameters.is_some());

        // Verify nested structure exists
        let params = tool.function.parameters.unwrap();
        if let Some(definitions) = params.get("definitions") {
            assert!(definitions.is_object());
            let defs = definitions.as_object().unwrap();
            assert!(defs.contains_key("Address"));
            assert!(defs.contains_key("Person"));
        }
    }
}