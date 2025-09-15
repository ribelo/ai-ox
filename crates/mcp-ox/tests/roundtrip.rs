
use ai_ox::content::part::Part;
use ai_ox::tool::ToolUse;
use mcp_sdk::types::{CallToolRequest, CallToolResponse, ToolResponseContent};
use mcp_ox::{FromMcp, ToMcp};
use serde_json::json;

#[test]
fn test_basic_tool_call_roundtrip() {
    // Create ai-ox ToolUse
    let original_call = ToolUse::new(
        "call_123",
        "test_function",
        json!({"arg1": "value1", "arg2": 42})
    );

    // Convert to MCP CallToolRequest
    let mcp_request = original_call.to_mcp().unwrap();

    // Verify ID is preserved in arguments
    assert_eq!(mcp_request.name, "test_function");
    let args = mcp_request.arguments.as_ref().unwrap().as_object().unwrap();
    assert_eq!(args.get("x_ai_ox_tool_call_id").unwrap().as_str().unwrap(), "call_123");
    assert_eq!(args.get("arg1").unwrap().as_str().unwrap(), "value1");
    assert_eq!(args.get("arg2").unwrap().as_i64().unwrap(), 42);

    // Convert back to ToolUse
    let back_call = ToolUse::from_mcp(mcp_request).unwrap();

    // Verify the call
    assert_eq!(back_call.id, "call_123");
    assert_eq!(back_call.name, "test_function");
    let back_args = back_call.args.as_object().unwrap();
    assert_eq!(back_args.get("arg1").unwrap().as_str().unwrap(), "value1");
    assert_eq!(back_args.get("arg2").unwrap().as_i64().unwrap(), 42);
}

#[test]
fn test_tool_result_roundtrip() {
    // Create ToolResult with text content
    let original_result = Part::ToolResult {
        id: "call_456".to_string(),
        name: "text_tool".to_string(),
        parts: vec![Part::Text { text: "Hello world".to_string(), ext: std::collections::BTreeMap::new() }],
        ext: std::collections::BTreeMap::new(),
    };

    // Convert to MCP CallToolResponse
    let mcp_response = original_result.to_mcp().unwrap();

    // Verify content has the response
    assert_eq!(mcp_response.content.len(), 1);
    if let ToolResponseContent::Text { text } = &mcp_response.content[0] {
        assert_eq!(text, "Hello world");
    } else {
        panic!("Expected text content");
    }

    // Verify meta is set correctly
    assert!(mcp_response.meta.is_some());
    let meta = mcp_response.meta.as_ref().unwrap();
    if let Some(ai_ox) = meta.get("ai_ox") {
        if let Some(obj) = ai_ox.as_object() {
            assert_eq!(obj.get("call_id").unwrap(), "call_456");
            assert_eq!(obj.get("name").unwrap(), "text_tool");
        } else {
            panic!("Invalid ai_ox meta structure");
        }
    } else {
        panic!("Missing ai_ox in meta");
    }

    // Convert back
    let back_result = Part::from_mcp(mcp_response).unwrap();

    // Verify exact preservation
    if let Part::ToolResult { id, name, parts, ext } = back_result {
        assert_eq!(id, "call_456");
        assert_eq!(name, "text_tool");
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            Part::Text { text, .. } => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text part"),
        }
        assert!(ext.is_empty());
    } else {
        panic!("Expected ToolResult part");
    }
}

#[test]
fn test_error_cases() {
    // Test missing ID scenarios
    let mcp_request = CallToolRequest {
        name: "test".to_string(),
        arguments: Some(json!({})),
        meta: None,
    };

    let call = ToolUse::from_mcp(mcp_request).unwrap();
    assert!(call.id.is_empty()); // No ID in arguments
}

#[test]
fn test_id_preservation() {
    // Create ToolUse with ID
    let call = ToolUse::new("unique_id", "func", json!({}));

    // Convert to MCP and back
    let mcp_req = call.to_mcp().unwrap();
    let back_call = ToolUse::from_mcp(mcp_req).unwrap();

    // Verify ID preserved
    assert_eq!(back_call.id, "unique_id");

    // Test ToolResult
    let result = Part::ToolResult {
        id: "result_id".to_string(),
        name: "func".to_string(),
        parts: vec![],
        ext: std::collections::BTreeMap::new(),
    };
    let mcp_resp = result.to_mcp().unwrap();
    let back_result = Part::from_mcp(mcp_resp).unwrap();
    if let Part::ToolResult { id, name, .. } = back_result {
        assert_eq!(id, "result_id");
        assert_eq!(name, "func");
    } else {
        panic!("Expected ToolResult part");
    }
}

#[test]
fn test_image_content_roundtrip() {
    use ai_ox::content::part::{DataRef, Part};

    // Create ToolResult with image content
    let original_result = Part::ToolResult {
        id: "call_img".to_string(),
        name: "image_tool".to_string(),
        parts: vec![Part::Blob {
            data_ref: DataRef::Base64 {
                data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==".to_string(),
            },
            mime_type: "image/png".to_string(),
            name: None,
            description: None,
            ext: std::collections::BTreeMap::new(),
        }],
        ext: std::collections::BTreeMap::new(),
    };

    // Convert to MCP CallToolResponse
    let mcp_response = original_result.to_mcp().unwrap();

    // Convert back
    let back_result = Part::from_mcp(mcp_response).unwrap();

    // Verify exact preservation
    if let Part::ToolResult { id, name, parts, .. } = back_result {
        assert_eq!(id, "call_img");
        assert_eq!(name, "image_tool");
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            Part::Blob { data_ref, mime_type, .. } => {
                assert_eq!(mime_type, "image/png");
                match data_ref {
                    DataRef::Base64 { data } => {
                        assert_eq!(data, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==");
                    }
                    DataRef::Uri { .. } => panic!("Unexpected URI source"),
                }
            }
            _ => panic!("Expected blob part"),
        }
    } else {
        panic!("Expected ToolResult part");
    }
}

#[test]
fn test_file_content_roundtrip() {
    use ai_ox::content::part::{DataRef, Part};

    // Create ToolResult with file content
    let original_result = Part::ToolResult {
        id: "call_file".to_string(),
        name: "file_tool".to_string(),
        parts: vec![Part::Blob {
            data_ref: DataRef::Uri {
                uri: "file:///tmp/test.txt".to_string(),
            },
            mime_type: "text/plain".to_string(),
            name: None,
            description: None,
            ext: std::collections::BTreeMap::new(),
        }],
        ext: std::collections::BTreeMap::new(),
    };

    // Convert to MCP CallToolResponse
    let mcp_response = original_result.to_mcp().unwrap();

    // Convert back
    let back_result = Part::from_mcp(mcp_response).unwrap();

    // Verify exact preservation
    if let Part::ToolResult { id, name, parts, .. } = back_result {
        assert_eq!(id, "call_file");
        assert_eq!(name, "file_tool");
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            Part::Blob { data_ref, mime_type, name, description, ext } => {
                assert_eq!(mime_type, "text/plain");
                match data_ref {
                    DataRef::Uri { uri } => assert_eq!(uri, "file:///tmp/test.txt"),
                    _ => panic!("Expected URI data_ref"),
                }
                assert_eq!(name.as_deref(), Some("resource"));
                assert!(description.is_none());
                assert!(ext.is_empty());
            }
            _ => panic!("Expected blob part"),
        }
    } else {
        panic!("Expected ToolResult part");
    }
}

#[test]
fn test_mixed_content_roundtrip() {
    use ai_ox::content::part::{DataRef, Part};

    // Create ToolResult with mixed content
    let original_result = Part::ToolResult {
        id: "call_mixed".to_string(),
        name: "mixed_tool".to_string(),
        parts: vec![
            Part::Text { text: "Here's an image:".to_string(), ext: std::collections::BTreeMap::new() },
            Part::Blob {
                data_ref: DataRef::Base64 {
                    data: "base64imagedata".to_string(),
                },
                mime_type: "image/jpeg".to_string(),
                name: None,
                description: None,
                ext: std::collections::BTreeMap::new(),
            },
            Part::Text { text: "And a file:".to_string(), ext: std::collections::BTreeMap::new() },
            Part::Blob {
                data_ref: DataRef::Uri {
                    uri: "file:///tmp/doc.pdf".to_string(),
                },
                mime_type: "application/pdf".to_string(),
                name: None,
                description: None,
                ext: std::collections::BTreeMap::new(),
            },
        ],
        ext: std::collections::BTreeMap::new(),
    };

    // Convert to MCP CallToolResponse
    let mcp_response = original_result.to_mcp().unwrap();

    // Convert back
    let back_result = Part::from_mcp(mcp_response).unwrap();

    // Verify exact preservation
    if let Part::ToolResult { id, name, parts, .. } = back_result {
        assert_eq!(id, "call_mixed");
        assert_eq!(name, "mixed_tool");
        assert_eq!(parts.len(), 4);

        match &parts[0] {
            Part::Text { text, .. } => assert_eq!(text, "Here's an image:"),
            _ => panic!("Expected text part"),
        }
        match &parts[1] {
            Part::Blob { data_ref, mime_type, .. } => {
                assert_eq!(mime_type, "image/jpeg");
                match data_ref {
                    DataRef::Base64 { data } => {
                        assert_eq!(data, "base64imagedata");
                    }
                    _ => panic!("Unexpected data_ref type"),
                }
            }
            _ => panic!("Expected blob part"),
        }
        match &parts[2] {
            Part::Text { text, .. } => assert_eq!(text, "And a file:"),
            _ => panic!("Expected text part"),
        }
        match &parts[3] {
            Part::Blob { data_ref, mime_type, name, description, ext } => {
                assert_eq!(mime_type, "application/pdf");
                match data_ref {
                    DataRef::Uri { uri } => assert_eq!(uri, "file:///tmp/doc.pdf"),
                    _ => panic!("Expected URI data_ref"),
                }
                assert_eq!(name.as_deref(), Some("resource"));
                assert!(description.is_none());
                assert!(ext.is_empty());
            }
            _ => panic!("Expected blob part"),
        }
    } else {
        panic!("Expected ToolResult part");
    }
}

#[test]
fn test_is_error_flag_propagation() {
    use std::collections::BTreeMap;

    // Test MCP CallToolResponse with is_error -> ToolResult with ext
    let mcp_response = CallToolResponse {
        content: vec![ToolResponseContent::Text {
            text: "Error occurred".to_string(),
        }],
        meta: Some(json!({
            "ai_ox": {
                "call_id": "error_call",
                "name": "error_tool"
            }
        })),
        is_error: Some(true),
    };

    let tool_result = Part::from_mcp(mcp_response).unwrap();
    if let Part::ToolResult { id, name, ext, .. } = tool_result {
        assert_eq!(id, "error_call");
        assert_eq!(name, "error_tool");
        assert!(!ext.is_empty());
        assert_eq!(ext.get("mcp.is_error").unwrap(), &json!(true));
    } else {
        panic!("Expected ToolResult part");
    }

    // Test ToolResult with ext -> MCP CallToolResponse with is_error
    let mut ext_map = BTreeMap::new();
    ext_map.insert("mcp.is_error".to_string(), json!(true));
    let original_result = Part::ToolResult {
        id: "error_call".to_string(),
        name: "error_tool".to_string(),
        parts: vec![],
        ext: ext_map,
    };

    let back_mcp = original_result.to_mcp().unwrap();
    assert_eq!(back_mcp.is_error, Some(true));

    // Test roundtrip without error
    let no_error_result = Part::ToolResult {
        id: "normal_call".to_string(),
        name: "normal_tool".to_string(),
        parts: vec![],
        ext: std::collections::BTreeMap::new(),
    };
    let mcp_no_error = no_error_result.to_mcp().unwrap();
    assert_eq!(mcp_no_error.is_error, None);

    let back_no_error = Part::from_mcp(mcp_no_error).unwrap();
    if let Part::ToolResult { ext, .. } = back_no_error {
        assert!(ext.is_empty());
    } else {
        panic!("Expected ToolResult part");
    }
}

#[test]
fn test_strict_vs_lenient_modes() {
    use mcp_ox::ConversionConfig;

    // Test lenient mode with missing meta
    let mcp_response_no_meta = CallToolResponse {
        content: vec![ToolResponseContent::Text {
            text: "No meta".to_string(),
        }],
        meta: None,
        is_error: None,
    };

    let lenient_config = ConversionConfig { strict: false };
    let result_lenient = Part::from_mcp_with_config(mcp_response_no_meta.clone(), &lenient_config).unwrap();
    if let Part::ToolResult { id, name, .. } = result_lenient {
        assert_eq!(id, "");
        assert_eq!(name, "");
    } else {
        panic!("Expected ToolResult part");
    }

    // Test strict mode with missing meta should fail
    let strict_config = ConversionConfig { strict: true };
    let result_strict = Part::from_mcp_with_config(mcp_response_no_meta, &strict_config);
    assert!(result_strict.is_err());

    // Test lenient mode with incomplete meta
    let mcp_response_incomplete_meta = CallToolResponse {
        content: vec![ToolResponseContent::Text {
            text: "Incomplete meta".to_string(),
        }],
        meta: Some(json!({
            "ai_ox": {
                "call_id": "test_id"
                // missing name
            }
        })),
        is_error: None,
    };

    let result_incomplete_lenient = Part::from_mcp_with_config(mcp_response_incomplete_meta.clone(), &lenient_config).unwrap();
    if let Part::ToolResult { id, name, .. } = result_incomplete_lenient {
        assert_eq!(id, "test_id");
        assert_eq!(name, "");
    } else {
        panic!("Expected ToolResult part");
    }

    // Test strict mode with incomplete meta should fail
    let result_incomplete_strict = Part::from_mcp_with_config(mcp_response_incomplete_meta, &strict_config);
    assert!(result_incomplete_strict.is_err());
}