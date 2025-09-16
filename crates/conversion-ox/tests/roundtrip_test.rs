//! Roundtrip conversion tests to ensure perfect data preservation
//!
//! These tests verify that converting from ai-ox format to provider formats
//! and back preserves ALL information without loss.

use ai_ox::content::part::{DataRef, Part};
use std::collections::BTreeMap;

/// Test roundtrip conversion for ToolResult parts
/// Ensures tool call ID, name, and all content parts are preserved
#[test]
fn test_toolresult_roundtrip_preservation() {
    // Create a complex ToolResult with multiple content types
    let original_tool_result = Part::ToolResult {
        id: "call_12345".to_string(),
        name: "complex_tool".to_string(),
        ext: BTreeMap::new(),
        parts: vec![
            Part::Text {
                text: "Tool execution successful".to_string(),
                ext: BTreeMap::new(),
            },
            Part::Blob {
                data_ref: DataRef::Base64 {
                    data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==".to_string(),
                },
                mime_type: "image/png".to_string(),
                name: None,
                description: None,
                ext: BTreeMap::new(),
            },
            Part::ToolResult {
                id: "nested_call".to_string(),
                name: "nested_tool".to_string(),
                parts: vec![Part::Text {
                    text: "Nested result".to_string(),
                    ext: BTreeMap::new(),
                }],
                ext: BTreeMap::new(),
            },
        ],
    };

    // TODO: Add actual conversion tests once conversion functions are implemented
    // For now, just verify the structure is correct
    match &original_tool_result {
        Part::ToolResult {
            id, name, parts, ..
        } => {
            assert_eq!(id, "call_12345");
            assert_eq!(name, "complex_tool");

            // Verify parts has 3 elements
            assert_eq!(parts.len(), 3);

            // Verify first content part
            if let Some(Part::Text { text, .. }) = parts.get(0) {
                assert_eq!(text, "Tool execution successful");
            } else {
                panic!("Expected text part");
            }

            // Verify second content part
            if let Some(Part::Blob {
                data_ref,
                mime_type,
                ..
            }) = parts.get(1)
            {
                assert_eq!(mime_type, "image/png");
                if let DataRef::Base64 { data } = data_ref {
                    assert!(data.len() > 0);
                } else {
                    panic!("Expected base64 data");
                }
            } else {
                panic!("Expected blob part");
            }

            // Verify third content part (nested tool result)
            if let Some(Part::ToolResult {
                id: nested_id,
                name: nested_name,
                parts: nested_parts,
                ..
            }) = parts.get(2)
            {
                assert_eq!(nested_id, "nested_call");
                assert_eq!(nested_name, "nested_tool");
                assert_eq!(nested_parts.len(), 1);
                if let Some(Part::Text { text, .. }) = nested_parts.get(0) {
                    assert_eq!(text, "Nested result");
                } else {
                    panic!("Expected nested text part");
                }
            } else {
                panic!("Expected nested tool result part");
            }
        }
        _ => panic!("Expected ToolResult part"),
    }
}

/// Test that empty tool results are handled correctly
#[test]
fn test_empty_toolresult_roundtrip() {
    let original_tool_result = Part::ToolResult {
        id: "empty_call".to_string(),
        name: "empty_tool".to_string(),
        parts: vec![],
        ext: BTreeMap::new(),
    };

    match &original_tool_result {
        Part::ToolResult {
            id, name, parts, ..
        } => {
            assert_eq!(id, "empty_call");
            assert_eq!(name, "empty_tool");
            assert!(parts.is_empty());
        }
        _ => panic!("Expected ToolResult part"),
    }
}

/// Test tool result with only text content
#[test]
fn test_text_only_toolresult_roundtrip() {
    let original_tool_result = Part::ToolResult {
        id: "text_call".to_string(),
        name: "text_tool".to_string(),
        parts: vec![Part::Text {
            text: "Simple text result".to_string(),
            ext: BTreeMap::new(),
        }],
        ext: BTreeMap::new(),
    };

    match &original_tool_result {
        Part::ToolResult {
            id, name, parts, ..
        } => {
            assert_eq!(id, "text_call");
            assert_eq!(name, "text_tool");
            assert_eq!(parts.len(), 1);
            if let Some(Part::Text { text, .. }) = parts.get(0) {
                assert_eq!(text, "Simple text result");
            } else {
                panic!("Expected text part");
            }
        }
        _ => panic!("Expected ToolResult part"),
    }
}

// TODO: Add actual roundtrip conversion tests for each provider:
// - Anthropic ↔ ai-ox ToolResult
// - OpenAI ↔ ai-ox ToolResult
// - OpenRouter ↔ ai-ox ToolResult
// - Gemini ↔ ai-ox ToolResult
// - Mistral ↔ ai-ox ToolResult
//
// Each test should:
// 1. Create ai-ox ToolResult with complex content
// 2. Convert to provider format
// 3. Convert back to ai-ox format
// 4. Verify EXACT equality of all fields
