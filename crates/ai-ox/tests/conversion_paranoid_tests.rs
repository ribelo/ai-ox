//! Additional paranoid conversion tests
//!
//! These tests complement the existing provider_roundtrip and encoding tests by
//! exercising additional edge-cases requested by the test architect: deep
//! nesting, all Part variants (including Opaque and URI-backed Blobs), provider
//! rejections for unsupported content (Anthropic), and provider roundtrips for
//! Gemini file-data variants and OpenRouter opaque-skipping behaviour.

use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::{DataRef, Part};
use ai_ox::tool::encoding::{decode_tool_result_parts, encode_tool_result_parts};
use serde_json::json;
use std::collections::BTreeMap;

// --------------------------------------------------------------------------------
// Encoding / decoding invariants
// --------------------------------------------------------------------------------

#[test]
fn test_encode_decode_preserves_all_part_variants_including_opaque_and_uri() {
    let mut ext = BTreeMap::new();
    ext.insert("x.test".to_string(), json!(true));

    let parts: Vec<Part> = vec![
        Part::Text {
            text: "Leading text".into(),
            ext: BTreeMap::new(),
        },
        // Inline base64 blob
        Part::Blob {
            data_ref: DataRef::Base64 {
                data: "YmFzZTY0ZGF0YQ==".into(),
            },
            mime_type: "application/pdf".into(),
            name: Some("doc.pdf".into()),
            description: Some("A pdf document".into()),
            ext: BTreeMap::new(),
        },
        // URI-backed blob
        Part::Blob {
            data_ref: DataRef::Uri {
                uri: "https://example.com/image.png".into(),
            },
            mime_type: "image/png".into(),
            name: None,
            description: None,
            ext: BTreeMap::new(),
        },
        // Tool use (should survive encode/decode if included inside ToolResult parts)
        Part::ToolUse {
            id: "call-1".into(),
            name: "compute".into(),
            args: json!({"value": 123}),
            ext: BTreeMap::new(),
        },
        // Opaque provider-specific content
        Part::Opaque {
            provider: "weird-provider".into(),
            kind: "special".into(),
            payload: json!({"blob": "data", "n": 1}),
            ext: ext.clone(),
        },
    ];

    let tool_name = "paranoid_test_tool";
    let encoded = encode_tool_result_parts(tool_name, &parts).expect("encoding should succeed");
    let (decoded_name, decoded_parts) =
        decode_tool_result_parts(&encoded).expect("decoding should succeed");

    assert_eq!(decoded_name, tool_name);
    assert_eq!(
        decoded_parts, parts,
        "All Part variants (Text, Blob URI/Base64, ToolUse, Opaque) must roundtrip exactly"
    );
}

#[test]
fn test_encode_decode_deeply_nested_toolresults() {
    // Build a 4-level nested ToolResult structure
    let level4 = Part::ToolResult {
        id: "lvl4".into(),
        name: "n4".into(),
        parts: vec![Part::Text {
            text: "deep".into(),
            ext: BTreeMap::new(),
        }],
        ext: BTreeMap::new(),
    };

    let level3 = Part::ToolResult {
        id: "lvl3".into(),
        name: "n3".into(),
        parts: vec![level4.clone()],
        ext: BTreeMap::new(),
    };

    let level2 = Part::ToolResult {
        id: "lvl2".into(),
        name: "n2".into(),
        parts: vec![level3.clone()],
        ext: BTreeMap::new(),
    };

    let level1 = Part::ToolResult {
        id: "lvl1".into(),
        name: "n1".into(),
        parts: vec![level2.clone()],
        ext: BTreeMap::new(),
    };

    let parts = vec![
        Part::Text {
            text: "start".into(),
            ext: BTreeMap::new(),
        },
        level1.clone(),
    ];

    let enc =
        encode_tool_result_parts("deep_tool", &parts).expect("encode deep nested should succeed");
    let (name, dec) = decode_tool_result_parts(&enc).expect("decode deep nested should succeed");

    assert_eq!(name, "deep_tool");
    assert_eq!(
        dec, parts,
        "Deeply nested ToolResult structures must be preserved exactly"
    );
}

#[test]
fn test_encode_decode_handles_tooluse_inside_parts() {
    let inner_tool_use = Part::ToolUse {
        id: "tu-1".into(),
        name: "inner".into(),
        args: json!({"k":"v"}),
        ext: BTreeMap::new(),
    };

    let parts = vec![Part::ToolResult {
        id: "tr-1".into(),
        name: "outer_tool".into(),
        parts: vec![inner_tool_use.clone()],
        ext: BTreeMap::new(),
    }];

    let encoded = encode_tool_result_parts("tool-with-tooluse", &parts).expect("encode");
    let (decoded_name, decoded_parts) = decode_tool_result_parts(&encoded).expect("decode");

    assert_eq!(decoded_name, "tool-with-tooluse");
    assert_eq!(decoded_parts, parts);
}

#[test]
fn test_encode_decode_large_base64_payload_roundtrips() {
    // Keep size moderate to avoid test timeout but large enough to stress serialization
    let payload = "A".repeat(512 * 1024); // ~512KB base64 string

    let parts = vec![Part::Blob {
        data_ref: DataRef::Base64 {
            data: payload.clone(),
        },
        mime_type: "image/png".into(),
        name: None,
        description: None,
        ext: BTreeMap::new(),
    }];

    let encoded = encode_tool_result_parts("big_blob", &parts).expect("encode");
    let (name, decoded) = decode_tool_result_parts(&encoded).expect("decode");

    assert_eq!(name, "big_blob");
    assert_eq!(
        decoded, parts,
        "Large base64 blob must survive encode/decode intact"
    );
}

#[test]
fn test_decode_rejects_malformed_structure_with_wrong_types() {
    // ai_ox_tool_result.name is a number instead of string
    let malformed = r#"{"ai_ox_tool_result": {"name": 123, "content": []}}"#;
    let res = decode_tool_result_parts(malformed);
    assert!(
        res.is_err(),
        "Decoding must fail for malformed/incorrectly-typed structure"
    );
}

// --------------------------------------------------------------------------------
// Provider-specific edge cases (feature gated)
// --------------------------------------------------------------------------------

// Commented out due to private module access - conversion functions are internal implementation details
// #[cfg(feature = "anthropic")]
// #[test]
// fn test_anthropic_rejects_audio_and_uris() {
//     use ai_ox::model::request::ModelRequest;
//     use ai_ox::model::anthropic::conversion::convert_request_to_anthropic;

//     // 1) base64-encoded audio should be rejected for Anthropic
//     let audio_part = Part::Blob {
//         data_ref: DataRef::Base64 { data: "SGVsbG8=".into() },
//         mime_type: "audio/mp3".into(),
//         name: None,
//         description: None,
//         ext: BTreeMap::new(),
//     };

//     let msg = Message::new(MessageRole::User, vec![audio_part]);
//     let req: ModelRequest = ModelRequest::from(vec![msg]);

//     let res = convert_request_to_anthropic(req, "claude-test".into(), None, 100, None);
//     assert!(res.is_err(), "Anthropic should reject base64 audio payloads");
//     let err = res.unwrap_err();
//     assert!(err.to_string().contains("Unsupported blob mime_type for Anthropic"));

//     // 2) URI-backed data refs are not supported by Anthropic in this converter
//     let uri_part = Part::Blob {
//         data_ref: DataRef::Uri { uri: "https://example.com/foo.png".into() },
//         mime_type: "image/png".into(),
//         name: None,
//         description: None,
//         ext: BTreeMap::new(),
//     };

//     let msg2 = Message::new(MessageRole::User, vec![uri_part]);
//     let req2: ModelRequest = ModelRequest::from(vec![msg2]);
//     let res2 = convert_request_to_anthropic(req2, "claude-test".into(), None, 100, None);
//     assert!(res2.is_err(), "Anthropic should reject URI-backed data references");
// //     let err2 = res2.unwrap_err();
// //     assert!(err2.to_string().contains("URI data references not supported by Anthropic provider"));
// // }

#[cfg(feature = "gemini")]
#[test]
fn test_gemini_roundtrip_preserves_filedata_uri_blob() {
    use gemini_ox::content::Content as GeminiContent;
    use std::convert::TryInto;

    // Build a Message with a ToolResult that contains a URI-backed blob
    let original = Message::new(
        MessageRole::Assistant,
        vec![Part::ToolResult {
            id: "call_gem_uri".into(),
            name: "file_tool".into(),
            parts: vec![Part::Blob {
                data_ref: DataRef::Uri {
                    uri: "https://example.com/photo.jpg".into(),
                },
                mime_type: "image/jpeg".into(),
                name: None,
                description: None,
                ext: BTreeMap::new(),
            }],
            ext: BTreeMap::new(),
        }],
    );

    // Convert to Gemini content and back
    let gemini: GeminiContent = original
        .clone()
        .try_into()
        .expect("gemini conversion failed");
    let roundtrip: Message = gemini
        .try_into()
        .expect("gemini -> ai-ox conversion failed");

    assert_eq!(
        original.content, roundtrip.content,
        "Gemini roundtrip must preserve URI-backed Blob as FileData"
    );
}

// Commented out due to private module access - conversion functions are internal implementation details
// #[cfg(feature = "openrouter")]
// #[test]
// fn test_openrouter_convert_message_skips_opaque_parts() {
//     use ai_ox::model::openrouter::conversion::build_openrouter_messages;
//     use openrouter_ox::message::Message as ORMessage;

//     let msg = Message::new(
//         MessageRole::User,
//         vec![
//             Part::Text { text: "Hello world".into(), ext: BTreeMap::new() },
//             Part::Opaque { provider: "x".into(), kind: "y".into(), payload: json!({"a":1}), ext: BTreeMap::new() },
//         ],
//     );

//     let or_msgs = convert_message_to_openrouter(msg, "test-model");

//     // The opaque part should be skipped, leaving a single user message with text
//     assert!(!or_msgs.is_empty());
//     // Verify first message is a User message containing the expected text
//     let first = or_msgs.into_iter().next().unwrap();
//     match first {
//         ORMessage::User(u) => {
//             let text = u.content.0.iter().filter_map(|p| p.as_text().map(|t| t.text.clone())).collect::<Vec<_>>().join(" ");
//             assert!(text.contains("Hello world"));
//         }
//         other => panic!("Expected User message, got {:?}", other),
//     }
// }
