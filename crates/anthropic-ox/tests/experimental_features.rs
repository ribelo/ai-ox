#![cfg(feature = "experimental")]

use anthropic_ox::{
    message::{Citations, Content, SearchResult, Text},
    tool::{ComputerTool, Tool},
};

#[test]
fn test_computer_tool_serialization() {
    let tool = Tool::Computer(ComputerTool {
        object_type: "computer_20250124".to_string(),
        name: "computer".to_string(),
        display_width_px: 1920,
        display_height_px: 1080,
        display_number: None,
    });

    let json = serde_json::to_string(&tool).unwrap();
    let expected = r#"{"type":"computer_20250124","name":"computer","display_width_px":1920,"display_height_px":1080}"#;
    assert_eq!(json, expected);
}

#[test]
fn test_search_result_content_serialization() {
    let content = Content::SearchResult(SearchResult {
        source: "https://example.com".to_string(),
        title: "Example".to_string(),
        content: vec![Text {
            text: "This is a test".to_string(),
            cache_control: None,
        }],
        citations: Some(Citations { enabled: true }),
        cache_control: None,
    });

    let json = serde_json::to_string(&content).unwrap();
    let expected = r#"{"type":"search_result","source":"https://example.com","title":"Example","content":[{"text":"This is a test"}],"citations":{"enabled":true}}"#;
    assert_eq!(json, expected);
}
