use anthropic_ox::tool::{ToolResult, ToolResultContent};

#[test]
fn test_tool_result_deserialize_string_content() {
    // Test the Claude Code format where content is a string
    let json_str = r#"{
        "tool_use_id": "toolu_01A09q90qw90lkasdjl",
        "content": "Cargo.lock\nCargo.toml\nCLAUDE.md\nsrc/"
    }"#;

    let result: ToolResult = serde_json::from_str(json_str).expect("Failed to deserialize");

    assert_eq!(result.tool_use_id, "toolu_01A09q90qw90lkasdjl");
    assert_eq!(result.content.len(), 1);

    match &result.content[0] {
        ToolResultContent::Text { text } => {
            assert_eq!(text, "Cargo.lock\nCargo.toml\nCLAUDE.md\nsrc/");
        }
        _ => panic!("Expected Text content"),
    }
}

#[test]
fn test_tool_result_deserialize_array_content() {
    // Test the standard Anthropic format where content is an array
    let json_str = r#"{
        "tool_use_id": "toolu_01A09q90qw90lkasdjl",
        "content": [
            {
                "type": "text",
                "text": "Here are the files:"
            }
        ]
    }"#;

    let result: ToolResult = serde_json::from_str(json_str).expect("Failed to deserialize");

    assert_eq!(result.tool_use_id, "toolu_01A09q90qw90lkasdjl");
    assert_eq!(result.content.len(), 1);

    match &result.content[0] {
        ToolResultContent::Text { text } => {
            assert_eq!(text, "Here are the files:");
        }
        _ => panic!("Expected Text content"),
    }
}

#[test]
fn test_tool_result_deserialize_with_cache_control() {
    // Test tool result with cache_control field
    let json_str = r#"{
        "tool_use_id": "toolu_01A09q90qw90lkasdjl",
        "content": "Files listed successfully",
        "cache_control": {
            "type": "ephemeral"
        }
    }"#;

    let result: ToolResult = serde_json::from_str(json_str).expect("Failed to deserialize");

    assert_eq!(result.tool_use_id, "toolu_01A09q90qw90lkasdjl");
    assert_eq!(result.content.len(), 1);
    assert!(result.cache_control.is_some());

    let cache_control = result.cache_control.unwrap();
    assert_eq!(cache_control.cache_type, "ephemeral");
}

#[test]
fn test_tool_result_deserialize_with_is_error() {
    // Test tool result with is_error field
    let json_str = r#"{
        "tool_use_id": "toolu_01A09q90qw90lkasdjl",
        "content": "Error: File not found",
        "is_error": true
    }"#;

    let result: ToolResult = serde_json::from_str(json_str).expect("Failed to deserialize");

    assert_eq!(result.tool_use_id, "toolu_01A09q90qw90lkasdjl");
    assert_eq!(result.content.len(), 1);
    assert_eq!(result.is_error, Some(true));

    match &result.content[0] {
        ToolResultContent::Text { text } => {
            assert_eq!(text, "Error: File not found");
        }
        _ => panic!("Expected Text content"),
    }
}
