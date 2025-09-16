//! Paranoid provider roundtrip tests
//!
//! These integration tests verify that provider adapters and the standardized
//! tool-result encoding preserve ai-ox Parts perfectly across conversions.
//!
//! Tests are feature-gated per provider so they only run when the corresponding
//! provider feature is enabled for the `ai-ox` crate. The tests are deliberately
//! exhaustive: text-only, multi-part (text + image), nested tool results, and
//! edge cases (empty content, Unicode, large payloads).

use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::{DataRef, Part};
use ai_ox::tool::encoding::{decode_tool_result_parts, encode_tool_result_parts};
use std::collections::BTreeMap;

// Small helper: flatten a sequence of ai-ox Messages into a single Vec<Part>
fn flatten_messages_parts(msgs: impl IntoIterator<Item = Message>) -> Vec<Part> {
    let mut parts = Vec::new();
    for m in msgs {
        parts.extend(m.content);
    }
    parts
}

// Common test data used across providers
fn sample_base64_png() -> &'static str {
    // 1x1 transparent PNG base64 (short and deterministic)
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
}

// ------------------------- Gemini roundtrips ------------------------------

#[cfg(feature = "gemini")]
mod gemini_roundtrip {
    use super::*;
    use gemini_ox::content::Content as GeminiContent;
    use std::convert::{TryFrom, TryInto};

    #[test]
    fn gemini_roundtrip_text_only_tool_result_preserves_everything() {
        let original = Message::new(
            MessageRole::Assistant,
            vec![Part::ToolResult {
                id: "call_gemini_1".to_string(),
                name: "compute_answer".to_string(),
                parts: vec![Part::Text {
                    text: "42".to_string(),
                    ext: BTreeMap::new(),
                }],
                ext: BTreeMap::new(),
            }],
        );

        // Convert to Gemini content
        let gemini: GeminiContent = original
            .clone()
            .try_into()
            .expect("gemini conversion failed");

        // Convert back to ai-ox Message
        let roundtrip: Message = gemini
            .try_into()
            .expect("gemini -> ai-ox conversion failed");

        assert_eq!(
            original.content, roundtrip.content,
            "Gemini roundtrip must preserve exact Parts (text-only)"
        );
    }

    #[test]
    fn gemini_roundtrip_multipart_and_nested_tool_results() {
        let nested = Part::ToolResult {
            id: "nested_call".to_string(),
            name: "nested_tool".to_string(),
            parts: vec![Part::Text {
                text: "inner".to_string(),
                ext: BTreeMap::new(),
            }],
            ext: BTreeMap::new(),
        };

        let original = Message::new(
            MessageRole::Assistant,
            vec![Part::ToolResult {
                id: "call_gemini_2".to_string(),
                name: "complex_tool".to_string(),
                parts: vec![
                    Part::Text {
                        text: "Result with image:".to_string(),
                        ext: BTreeMap::new(),
                    },
                    Part::Blob {
                        data_ref: DataRef::Base64 {
                            data: sample_base64_png().to_string(),
                        },
                        mime_type: "image/png".to_string(),
                        name: None,
                        description: None,
                        ext: BTreeMap::new(),
                    },
                    nested.clone(),
                ],
                ext: BTreeMap::new(),
            }],
        );

        let gemini: GeminiContent = original
            .clone()
            .try_into()
            .expect("gemini conversion failed");
        let roundtrip: Message = gemini
            .try_into()
            .expect("gemini -> ai-ox conversion failed");

        assert_eq!(
            original.content, roundtrip.content,
            "Gemini roundtrip must preserve multipart and nested tool results exactly"
        );
    }

    #[test]
    fn gemini_roundtrip_edge_cases_empty_unicode_large() {
        // empty content
        let empty = Message::new(
            MessageRole::Assistant,
            vec![Part::ToolResult {
                id: "empty_call".to_string(),
                name: "empty_tool".to_string(),
                parts: vec![],
                ext: BTreeMap::new(),
            }],
        );

        let gemini_empty: GeminiContent = empty
            .clone()
            .try_into()
            .expect("gemini empty conversion failed");
        let roundtrip_empty: Message = gemini_empty
            .try_into()
            .expect("gemini empty -> ai-ox failed");
        assert_eq!(
            empty.content, roundtrip_empty.content,
            "Empty ToolResult must roundtrip without loss"
        );

        // unicode
        let unicode_text =
            "Emoji: 🧪 — RTL: \u{05D0}\u{05D1}\u{05D2} — Combining: a\u{0301}".to_string();
        let unicode = Message::new(
            MessageRole::Assistant,
            vec![Part::ToolResult {
                id: "u1".to_string(),
                name: "u_tool".to_string(),
                parts: vec![Part::Text {
                    text: unicode_text.clone(),
                    ext: BTreeMap::new(),
                }],
                ext: BTreeMap::new(),
            }],
        );
        let gemini_unicode: GeminiContent = unicode
            .clone()
            .try_into()
            .expect("gemini unicode conversion failed");
        let roundtrip_unicode: Message = gemini_unicode
            .try_into()
            .expect("gemini unicode -> ai-ox failed");
        assert_eq!(
            unicode.content, roundtrip_unicode.content,
            "Unicode must survive roundtrip intact"
        );

        // large data (moderate size to keep test reasonable)
        let large = "X".repeat(20_000);
        let large_msg = Message::new(
            MessageRole::Assistant,
            vec![Part::ToolResult {
                id: "big".to_string(),
                name: "big_tool".to_string(),
                parts: vec![Part::Text {
                    text: large.clone(),
                    ext: BTreeMap::new(),
                }],
                ext: BTreeMap::new(),
            }],
        );
        let gemini_large: GeminiContent = large_msg
            .clone()
            .try_into()
            .expect("gemini large conversion failed");
        let roundtrip_large: Message = gemini_large
            .try_into()
            .expect("gemini large -> ai-ox failed");
        assert_eq!(
            large_msg.content, roundtrip_large.content,
            "Large payload must be preserved exactly"
        );
    }
}

// ----------------------- OpenRouter roundtrips -----------------------------

#[cfg(feature = "openrouter")]
mod openrouter_roundtrip {
    use super::*;
    use openrouter_ox::message::{
        AssistantMessage, Message as ORMessage, ToolMessage, UserMessage,
    };

    fn simulate_openrouter_messages_from_ai(original: &Message) -> Vec<ORMessage> {
        let mut msgs = Vec::new();

        for part in &original.content {
            match part {
                Part::Text { text, .. } => {
                    // user/assistant text maps to a user message for roundtrip testing
                    msgs.push(ORMessage::User(UserMessage::text(text.clone())));
                }
                Part::ToolResult {
                    id, name, parts, ..
                } => {
                    // encode the tool result parts into the standardized JSON string
                    let encoded = encode_tool_result_parts(name, parts).expect("encoding failed");
                    // OpenRouter Tool messages carry name -> use with_name so conversion back can retrieve it
                    msgs.push(ORMessage::Tool(ToolMessage::with_name(
                        id.clone(),
                        encoded,
                        name.clone(),
                    )));
                }
                other => {
                    // Fallback: serialize unknown parts to text and send as user message
                    let serialized = serde_json::to_string(&other)
                        .unwrap_or_else(|_| "<serialization error>".to_string());
                    msgs.push(ORMessage::User(UserMessage::text(serialized)));
                }
            }
        }

        msgs
    }

    #[test]
    fn openrouter_roundtrip_text_only_tool_result() {
        let original = Message::new(
            MessageRole::User,
            vec![
                Part::Text {
                    text: "Here is a result:".to_string(),
                    ext: BTreeMap::new(),
                },
                Part::ToolResult {
                    id: "call_or_1".to_string(),
                    name: "search".to_string(),
                    parts: vec![Part::Text {
                        text: "found".to_string(),
                        ext: BTreeMap::new(),
                    }],
                    ext: BTreeMap::new(),
                },
            ],
        );

        let or_msgs = simulate_openrouter_messages_from_ai(&original);

        // Convert provider messages back into ai-ox messages using From impl
        let ai_msgs: Vec<Message> = or_msgs.into_iter().map(|m| m.into()).collect();
        let final_parts = flatten_messages_parts(ai_msgs);

        assert_eq!(
            original.content, final_parts,
            "OpenRouter roundtrip (simulated) must preserve exact Parts"
        );
    }

    #[test]
    fn openrouter_roundtrip_multipart_and_edge_cases() {
        let original = Message::new(
            MessageRole::Assistant,
            vec![Part::ToolResult {
                id: "call_or_2".to_string(),
                name: "complex_tool".to_string(),
                parts: vec![
                    Part::Text {
                        text: "Line 1".to_string(),
                        ext: BTreeMap::new(),
                    },
                    Part::Blob {
                        data_ref: DataRef::Base64 {
                            data: sample_base64_png().to_string(),
                        },
                        mime_type: "image/png".to_string(),
                        name: None,
                        description: None,
                        ext: BTreeMap::new(),
                    },
                    Part::Text {
                        text: "Unicode: ✅ Ω こんにちは".to_string(),
                        ext: BTreeMap::new(),
                    },
                ],
                ext: BTreeMap::new(),
            }],
        );

        let or_msgs = simulate_openrouter_messages_from_ai(&original);
        let ai_msgs: Vec<Message> = or_msgs.into_iter().map(|m| m.into()).collect();
        let final_parts = flatten_messages_parts(ai_msgs);

        assert_eq!(
            original.content, final_parts,
            "OpenRouter multipart & unicode must roundtrip exactly"
        );

        // empty content case
        let empty = Message::new(
            MessageRole::User,
            vec![Part::ToolResult {
                id: "empty_or".to_string(),
                name: "empty".to_string(),
                parts: vec![],
                ext: BTreeMap::new(),
            }],
        );
        let or_empty = simulate_openrouter_messages_from_ai(&empty);
        let ai_empty: Vec<Message> = or_empty.into_iter().map(|m| m.into()).collect();
        let final_empty = flatten_messages_parts(ai_empty);
        assert_eq!(
            empty.content, final_empty,
            "OpenRouter empty ToolResult must roundtrip"
        );
    }
}

// ------------------------- Encoding invariants ----------------------------
// Tests that all providers rely on: the standardized encoding/decoding of Vec<Part>
// This is the single-source-of-truth for transporting complex tool results
// through providers that only accept strings.

#[test]
fn encode_decode_roundtrip_text_and_image() {
    let name = "test_tool";
    let parts = vec![
        Part::Text {
            text: "Leading text".to_string(),
            ext: BTreeMap::new(),
        },
        Part::Blob {
            data_ref: DataRef::Base64 {
                data: sample_base64_png().to_string(),
            },
            mime_type: "image/png".to_string(),
            name: None,
            description: None,
            ext: BTreeMap::new(),
        },
        Part::Text {
            text: "Trailing text".to_string(),
            ext: BTreeMap::new(),
        },
    ];

    let encoded = encode_tool_result_parts(name, &parts).expect("encode failed");
    let (decoded_name, decoded_parts) = decode_tool_result_parts(&encoded).expect("decode failed");

    assert_eq!(name, decoded_name, "tool name should roundtrip");
    assert_eq!(
        parts, decoded_parts,
        "encode/decode should be a perfect roundtrip for mixed text+image parts"
    );
}

#[test]
fn encode_decode_roundtrip_complex_nested() {
    let name = "complex_tool";
    let nested = Part::ToolResult {
        id: "nested_call".to_string(),
        name: "nested".to_string(),
        parts: vec![Part::Text {
            text: "inner".to_string(),
            ext: BTreeMap::new(),
        }],
        ext: BTreeMap::new(),
    };
    let parts = vec![
        Part::Text {
            text: "outer".to_string(),
            ext: BTreeMap::new(),
        },
        nested,
    ];

    let encoded = encode_tool_result_parts(name, &parts).expect("encode failed");
    let (decoded_name, decoded_parts) = decode_tool_result_parts(&encoded).expect("decode failed");

    assert_eq!(name, decoded_name, "tool name should roundtrip");
    assert_eq!(
        parts, decoded_parts,
        "encode/decode must preserve nested tool result structure exactly"
    );
}

#[test]
fn encode_decode_edge_cases_empty_unicode_large() {
    let name = "edge_tool";

    // empty
    let parts_empty: Vec<Part> = vec![];
    let encoded_empty = encode_tool_result_parts(name, &parts_empty).expect("encode empty failed");
    let (decoded_name_empty, decoded_empty) =
        decode_tool_result_parts(&encoded_empty).expect("decode empty failed");
    assert_eq!(
        name, decoded_name_empty,
        "tool name should roundtrip for empty parts"
    );
    assert_eq!(parts_empty, decoded_empty, "empty parts must roundtrip");

    // unicode
    let u = "Δλ😊📚 — combining: a\u{0301}".to_string();
    let parts_uni = vec![Part::Text {
        text: u.clone(),
        ext: BTreeMap::new(),
    }];
    let encoded_uni = encode_tool_result_parts(name, &parts_uni).expect("encode unicode failed");
    let (decoded_name_uni, decoded_uni) =
        decode_tool_result_parts(&encoded_uni).expect("decode unicode failed");
    assert_eq!(
        name, decoded_name_uni,
        "tool name should roundtrip for unicode"
    );
    assert_eq!(parts_uni, decoded_uni, "unicode must roundtrip intact");

    // large payload
    let large_text = "L".repeat(50_000);
    let parts_large = vec![Part::Text {
        text: large_text.clone(),
        ext: BTreeMap::new(),
    }];
    let encoded_large = encode_tool_result_parts(name, &parts_large).expect("encode large failed");
    let (decoded_name_large, decoded_large) =
        decode_tool_result_parts(&encoded_large).expect("decode large failed");
    assert_eq!(
        name, decoded_name_large,
        "tool name should roundtrip for large payload"
    );
    assert_eq!(
        parts_large, decoded_large,
        "large payload must roundtrip exactly"
    );
}

// ------------------------- Mistral / Bedrock (encoding-based) -------------
// These providers rely on the standardized encoding. We test that the encoding
// preserves content in the presence of provider wrappers.

#[cfg(feature = "mistral")]
mod mistral_encoding_roundtrip {
    use super::*;
    use mistral_ox::message::{
        Message as MMessage, ToolMessage as MToolMessage, UserMessage as MUserMessage,
    };

    #[test]
    fn mistral_tool_message_preserves_encoded_content() {
        let tool_name = "mistral_tool";
        let original_content = vec![
            Part::Text {
                text: "Hello from tool".to_string(),
                ext: BTreeMap::new(),
            },
            Part::Blob {
                data_ref: DataRef::Base64 {
                    data: sample_base64_png().to_string(),
                },
                mime_type: "image/png".to_string(),
                name: None,
                description: None,
                ext: BTreeMap::new(),
            },
        ];

        let encoded =
            encode_tool_result_parts(tool_name, &original_content).expect("encode failed");
        let tool_msg = MToolMessage::new("call_m1", encoded.clone());

        // Simulate provider -> ai-ox: decode the tool content
        let (decoded_name, decoded_content) =
            decode_tool_result_parts(tool_msg.content()).expect("decode failed");

        assert_eq!(tool_name, decoded_name, "tool name must be preserved");
        assert_eq!(
            original_content, decoded_content,
            "Mistral ToolMessage content must decode to original Parts"
        );

        // also test empty tool message
        let empty_tool = MToolMessage::new(
            "call_empty",
            encode_tool_result_parts(tool_name, &Vec::<Part>::new()).unwrap(),
        );
        let (decoded_empty_name, decoded_empty) =
            decode_tool_result_parts(empty_tool.content()).expect("decode empty failed");
        assert_eq!(
            tool_name, decoded_empty_name,
            "tool name must be preserved for empty content"
        );
        assert!(
            decoded_empty.is_empty(),
            "Decoded empty tool message must be empty parts"
        );
    }
}

#[cfg(feature = "bedrock")]
mod bedrock_encoding_roundtrip {
    use super::*;
    use aws_sdk_bedrockruntime::types::{ToolResultBlock, ToolResultContentBlock};

    #[test]
    fn bedrock_toolresult_block_preserves_encoded_content() {
        let tool_name = "bedrock_tool";
        let original_content = vec![
            Part::Text {
                text: "Bedrock text".to_string(),
                ext: BTreeMap::new(),
            },
            Part::Blob {
                data_ref: DataRef::Base64 {
                    data: sample_base64_png().to_string(),
                },
                mime_type: "image/png".to_string(),
                name: None,
                description: None,
                ext: BTreeMap::new(),
            },
        ];

        let encoded =
            encode_tool_result_parts(tool_name, &original_content).expect("encode failed");

        // Simulate a ToolResultBlock containing a single Text content entry (what the converter uses)
        let tool_block = ToolResultBlock::builder()
            .tool_use_id("call_b1")
            .content(ToolResultContentBlock::Text(encoded.clone()))
            .build()
            .expect("failed to build ToolResultBlock");

        // In ai-ox bedrock conversion the Text content is decoded using decode_tool_result_parts
        let content_text = match tool_block.content() {
            [ToolResultContentBlock::Text(t)] => t.clone(),
            _ => panic!("unexpected tool result content structure"),
        };

        let (decoded_name, decoded_content) =
            decode_tool_result_parts(&content_text).expect("decode failed");
        assert_eq!(tool_name, decoded_name, "tool name must be preserved");
        assert_eq!(
            original_content, decoded_content,
            "Bedrock ToolResultBlock text must decode to original Parts"
        );
    }
}

// ------------------------- Notes ------------------------------------------
// These tests are intentionally conservative about which public APIs they call.
// They exercise:
// - The gemini TryFrom/TryInto conversions (full ai-ox <-> provider roundtrip)
// - The OpenRouter From<openrouter_ox::message::Message> impl (provider -> ai-ox)
//   together with the standardized encoding used to carry complex tool results
// - The standardized encode/decode functions directly (the single source of truth)
// - Simulated provider wrappers for Mistral and Bedrock that carry the encoded
//   tool-result payload as text, verifying decode produces identical Parts.
//
// If you enable the provider features and these tests fail, something in the
// conversion path (or the encoding) has broken the roundtrip guarantee. As I
// always say: "That's not a bug, that's a failure to test the space-time
// continuum of your logic." -- Sheldon
