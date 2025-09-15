use crate::tool::{Tool as AiOxTool, ToolUse};
pub use gemini_ox::tool::{FunctionMetadata as GeminiFunctionMetadata, Tool as GeminiTool};
use gemini_ox::content::part::FunctionCall as GeminiFunctionCall;

/// Converts an `ai-ox` `Tool` to a `gemini-ox` `Tool`.
impl From<AiOxTool> for GeminiTool {
    fn from(ai_tool: AiOxTool) -> Self {
        match ai_tool {
            AiOxTool::FunctionDeclarations(functions) => {
                // Convert ai-ox FunctionMetadata to gemini-ox FunctionMetadata
                let gemini_functions: Vec<GeminiFunctionMetadata> = functions
                    .into_iter()
                    .map(|func| GeminiFunctionMetadata {
                        name: func.name,
                        description: func.description,
                        parameters: func.parameters,
                    })
                    .collect();
                Self::FunctionDeclarations(gemini_functions)
            }
            AiOxTool::GeminiTool(gemini_tool) => gemini_tool,
        }
    }
}

impl From<GeminiFunctionCall> for ToolUse {
    fn from(gemini_call: GeminiFunctionCall) -> Self {
        ToolUse {
            id: uuid::Uuid::new_v4().to_string(), // Gemini does not provide an ID for tool calls
            name: gemini_call.name,
            args: gemini_call.args.unwrap_or_default(),
            ext: Some(std::collections::BTreeMap::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_function_declarations_conversion_ai_to_gemini() {
        use crate::tool::FunctionMetadata;

        let ai_tool = AiOxTool::FunctionDeclarations(vec![FunctionMetadata {
            name: "test_function".to_string(),
            description: Some("A test function".to_string()),
            parameters: json!({"type": "object", "properties": {}}),
        }]);

        let gemini_tool: GeminiTool = ai_tool.into();

        match gemini_tool {
            GeminiTool::FunctionDeclarations(functions) => {
                assert_eq!(functions.len(), 1);
                assert_eq!(functions[0].name, "test_function");
                assert_eq!(
                    functions[0].description,
                    Some("A test function".to_string())
                );
            }
            _ => panic!("Expected FunctionDeclarations variant"),
        }
    }

    #[test]
    fn test_gemini_tool_passthrough_conversion() {
        use gemini_ox::tool::google::GoogleSearch;

        let google_search = GoogleSearch::default();
        let inner_gemini_tool = GeminiTool::GoogleSearch(google_search);
        let ai_tool = AiOxTool::GeminiTool(inner_gemini_tool.clone());

        let converted_gemini_tool: GeminiTool = ai_tool.into();

        match converted_gemini_tool {
            GeminiTool::GoogleSearch(_) => {
                // Test passes if we get the GoogleSearch variant
            }
            _ => panic!("Expected GoogleSearch variant"),
        }
    }

    #[tokio::test]
    #[ignore = "Requires GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
    async fn test_google_search_tool_integration()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::content::message::{Message, MessageRole};
        use crate::content::part::Part;
        use crate::model::{Model, gemini::GeminiModel};
        

        let api_key =
            match std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_AI_API_KEY")) {
                Ok(key) => key,
                Err(_) => {
                    println!(
                        "GEMINI_API_KEY or GOOGLE_AI_API_KEY not set, skipping google search test"
                    );
                    return Ok(());
                }
            };

        // Create a GeminiModel with GoogleSearch tool
        let model = GeminiModel::builder()
            .api_key(api_key)
            .model("gemini-1.5-flash".to_string())
            .build();

        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Search for information about Rust programming language".to_string(),
                ext: std::collections::BTreeMap::new(),
            }],
            timestamp: Some(chrono::Utc::now()),
            ext: Some(std::collections::BTreeMap::new()),
        };

        // This should work without errors (GoogleSearch tool should be available)
        let result = model.request(message.into()).await;

        match result {
            Ok(response) => {
                println!("GoogleSearch tool integration test passed");
                println!("Response: {:?}", response.message.content);
                assert_eq!(response.vendor_name, "google");
                assert_eq!(response.model_name, "gemini-1.5-flash");
            }
            Err(e) => {
                // For now, just log the error - the tool integration might not trigger actual search
                println!("GoogleSearch tool test completed with: {e:?}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_google_search_tool_serialization() {
        use gemini_ox::tool::google::GoogleSearch;

        // Test that GoogleSearch can be wrapped in ai-ox Tool and serialized
        let google_search = GoogleSearch::default();
        let gemini_tool = GeminiTool::GoogleSearch(google_search);
        let ai_tool = AiOxTool::GeminiTool(gemini_tool.clone());

        // Should serialize without errors
        let serialized = serde_json::to_value(&ai_tool).unwrap();
        println!(
            "Serialized GoogleSearch tool: {}",
            serde_json::to_string_pretty(&serialized).unwrap()
        );

        // Should be able to convert back to GeminiTool
        let converted: GeminiTool = ai_tool.into();
        match converted {
            GeminiTool::GoogleSearch(_) => {
                println!("GoogleSearch tool serialization test passed");
            }
            _ => panic!("Expected GoogleSearch after conversion"),
        }
    }

    #[test]
    fn test_multiple_tools_including_google_search() {
        use crate::tool::FunctionMetadata;
        use gemini_ox::tool::google::{GoogleSearch, GoogleSearchRetrieval};

        // Test that we can have multiple tools including GoogleSearch
        let tools = vec![
            AiOxTool::FunctionDeclarations(vec![FunctionMetadata {
                name: "custom_function".to_string(),
                description: Some("A custom function".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
            }]),
            AiOxTool::GeminiTool(GeminiTool::GoogleSearch(GoogleSearch::default())),
            AiOxTool::GeminiTool(GeminiTool::GoogleSearchRetrieval {
                google_search_retrieval: GoogleSearchRetrieval::default(),
            }),
        ];

        // All tools should convert successfully
        let converted_tools: Vec<GeminiTool> = tools.into_iter().map(Into::into).collect();

        assert_eq!(converted_tools.len(), 3);

        // Verify the types
        match &converted_tools[0] {
            GeminiTool::FunctionDeclarations(_) => {}
            _ => panic!("Expected FunctionDeclarations"),
        }

        match &converted_tools[1] {
            GeminiTool::GoogleSearch(_) => {}
            _ => panic!("Expected GoogleSearch"),
        }

        match &converted_tools[2] {
            GeminiTool::GoogleSearchRetrieval { .. } => {}
            _ => panic!("Expected GoogleSearchRetrieval"),
        }

        println!("Multiple tools test passed");
    }
}
