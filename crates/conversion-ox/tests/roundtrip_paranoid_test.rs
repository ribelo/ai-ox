// Paranoid roundtrip tests: these are RED tests and are expected to FAIL
// The goal is to expose silent data-loss in provider conversions. If these
// ever pass without code changes, congratulations — you fixed the space-time
// continuum of your conversions.

#![allow(dead_code, unused_imports)]

use serde_json::json;

// We're intentionally splitting tests behind feature gates that mirror the
// conversion-ox features. Run `cargo test -p conversion-ox --features test`
// (or the specific feature) to exercise them. These tests assume an
// environment where provider crates are available; they are written to
// produce loud, obvious failures if any information is dropped silently.

// -----------------------------------------------------------------------------
// 1) Anthropic <-> Gemini
// -----------------------------------------------------------------------------

#[cfg(feature = "anthropic-gemini")]
#[test]
fn anthropic_gemini_roundtrip_toolresult_must_preserve_every_byte() {
    // Build a complex Anthropic ChatResponse containing a ToolResult with
    // multiple content parts (text, JSON-as-text, image). The conversion
    // pipeline MUST preserve every content part exactly on a roundtrip
    // Anthropic -> Gemini -> Anthropic, or else fail loudly.

    use anthropic_ox::message::Role as AnthropicRole;
    use anthropic_ox::message::{
        Content as AnthropicContent, ImageSource as AnthropicImageSource, Text as AnthropicText,
    };
    use anthropic_ox::response::ChatResponse as AnthropicResponse;
    use anthropic_ox::tool::{
        ToolResult as AnthropicToolResult, ToolResultContent as AnthropicToolResultContent,
    };
    use conversion_ox::anthropic_gemini::{
        anthropic_to_gemini_response, gemini_to_anthropic_response,
    };

    // Tiny 1x1 PNG base64 (same as used elsewhere in tests)
    let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==".to_string();

    let json_payload = json!({
        "temperature": 72,
        "humidity": 65,
        "condition": "sunny"
    });

    let original_tool_result = AnthropicToolResult {
        tool_use_id: "call_12345".to_string(),
        content: vec![
            AnthropicToolResultContent::Text {
                text: "Tool execution successful".to_string(),
            },
            // JSON we expect to survive as a textual block
            AnthropicToolResultContent::Text {
                text: serde_json::to_string_pretty(&json_payload).unwrap(),
            },
            // Image content (binary -> base64); this absolutely MUST survive
            AnthropicToolResultContent::Image {
                source: AnthropicImageSource::Base64 {
                    media_type: "image/png".to_string(),
                    data: png_b64.clone(),
                },
            },
        ],
        is_error: None,
        cache_control: None,
    };

    let original_response = AnthropicResponse {
        id: "resp-tool-1".to_string(),
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content: vec![AnthropicContent::ToolResult(original_tool_result.clone())],
        model: "claude-test-model".to_string(),
        stop_reason: None,
        stop_sequence: None,
        usage: anthropic_ox::response::Usage::default(),
    };

    // Do the conversion Anthropic -> Gemini -> Anthropic
    let gemini_resp = anthropic_to_gemini_response(original_response.clone()).unwrap();
    let roundtrip = gemini_to_anthropic_response(gemini_resp).unwrap();

    // Extract the ToolResult from the roundtrip response
    let round_tool_result = match roundtrip.content.get(0) {
        Some(AnthropicContent::ToolResult(tr)) => tr,
        other => panic!(
            "Expected ToolResult after Gemini -> Anthropic roundtrip, got: {:?}",
            other
        ),
    };

    // 1) tool_use_id must be preserved exactly
    assert_eq!(
        round_tool_result.tool_use_id, original_tool_result.tool_use_id,
        "Tool use ID must be preserved exactly on roundtrip"
    );

    // 2) The number of content items must be identical
    assert_eq!(
        round_tool_result.content.len(),
        original_tool_result.content.len(),
        "Number of ToolResult content parts changed on roundtrip. Original: {:#?}\nRoundtrip: {:#?}",
        original_tool_result.content,
        round_tool_result.content
    );

    // 3) Each item must be byte-for-byte identical in both value and variant
    for (idx, orig_item) in original_tool_result.content.iter().enumerate() {
        match (&orig_item, &round_tool_result.content[idx]) {
            (
                AnthropicToolResultContent::Text { text: orig_text },
                AnthropicToolResultContent::Text { text: round_text },
            ) => {
                assert_eq!(
                    round_text, orig_text,
                    "Text content at index {} was mangled during roundtrip\nORIGINAL:\n{}\nROUNDTRIP:\n{}",
                    idx, orig_text, round_text
                );
            }
            (
                AnthropicToolResultContent::Image { source: orig_src },
                AnthropicToolResultContent::Image { source: round_src },
            ) => {
                // Compare media type and base64 payload
                match (orig_src, round_src) {
                    (
                        AnthropicImageSource::Base64 {
                            media_type: orig_mt,
                            data: orig_data,
                        },
                        AnthropicImageSource::Base64 {
                            media_type: round_mt,
                            data: round_data,
                        },
                    ) => {
                        assert_eq!(orig_mt, round_mt, "Image media_type changed on roundtrip");
                        assert_eq!(
                            orig_data, round_data,
                            "Image base64 data changed on roundtrip (data lost or truncated)"
                        );
                    }
                }
            }
            (o, r) => {
                panic!(
                    "ToolResult content type changed at index {}: original={:?} roundtrip={:?}",
                    idx, o, r
                );
            }
        }
    }

    // If we reached here, we consider the roundtrip perfect. However we
    // expect this test to FAIL in the current codebase because the conversion
    // path intentionally drops images and/or collapses structured content.
}

// -----------------------------------------------------------------------------
// 2) Anthropic <-> OpenAI Responses API
// -----------------------------------------------------------------------------

#[cfg(feature = "anthropic-openai")]
#[test]
fn anthropic_openai_responses_roundtrip_toolresult_must_preserve_every_byte() {
    // Similar paranoid test for the Anthropic <-> OpenAI Responses conversion
    // path. The Responses API conversion often concatenates or skips content;
    // this test asserts that either the conversion fails explicitly OR the
    // content is preserved exactly.

    use anthropic_ox::message::Role as AnthropicRole;
    use anthropic_ox::message::{
        Content as AnthropicContent, ImageSource as AnthropicImageSource, Text as AnthropicText,
    };
    use anthropic_ox::response::ChatResponse as AnthropicResponse;
    use anthropic_ox::tool::{
        ToolResult as AnthropicToolResult, ToolResultContent as AnthropicToolResultContent,
    };
    use conversion_ox::anthropic_openai::{
        anthropic_to_openai_responses_response, openai_responses_to_anthropic_response,
    };

    let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==".to_string();
    let json_payload = json!({ "a": 1, "b": [1,2,3], "c": { "nested": true } });

    let original_tool_result = AnthropicToolResult {
        tool_use_id: "call_openai_42".to_string(),
        content: vec![
            AnthropicToolResultContent::Text {
                text: "Plain textual response".to_string(),
            },
            AnthropicToolResultContent::Text {
                text: serde_json::to_string(&json_payload).unwrap(),
            },
            AnthropicToolResultContent::Image {
                source: AnthropicImageSource::Base64 {
                    media_type: "image/png".to_string(),
                    data: png_b64.clone(),
                },
            },
        ],
        is_error: None,
        cache_control: None,
    };

    let original_response = AnthropicResponse {
        id: "resp-openai-1".to_string(),
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content: vec![AnthropicContent::ToolResult(original_tool_result.clone())],
        model: "claude-openai-proxy".to_string(),
        stop_reason: None,
        stop_sequence: None,
        usage: anthropic_ox::response::Usage::default(),
    };

    // Convert Anthropic -> OpenAI Responses -> Anthropic
    let responses_resp = anthropic_to_openai_responses_response(original_response.clone());

    // The conversion function returns Result<ResponsesResponse, ConversionError>,
    // but in practice it will likely succeed and silently drop unsupported
    // parts. We assert that the roundtrip either errors (explicitly) or the
    // content is identical.
    match responses_resp {
        Ok(resp) => {
            // Convert back
            match openai_responses_to_anthropic_response(resp) {
                Ok(roundtrip) => {
                    // Extract ToolResult
                    let round_tool = match roundtrip.content.get(0) {
                        Some(AnthropicContent::Thinking(_)) | Some(AnthropicContent::Text(_)) => {
                            // If the Responses conversion turned our tool result into
                            // plain text or thinking blocks, that's a silent loss.
                            panic!(
                                "OpenAI Responses roundtrip did NOT return a ToolResult; instead returned content: {:#?}",
                                roundtrip.content
                            );
                        }
                        Some(AnthropicContent::ToolResult(tr)) => tr,
                        other => panic!(
                            "Expected ToolResult after Responses -> Anthropic, got: {:?}",
                            other
                        ),
                    };

                    // Validate strict equality of content parts
                    assert_eq!(
                        round_tool.tool_use_id, original_tool_result.tool_use_id,
                        "tool_use_id changed"
                    );
                    assert_eq!(
                        round_tool.content.len(),
                        original_tool_result.content.len(),
                        "content length changed"
                    );

                    for (i, orig_c) in original_tool_result.content.iter().enumerate() {
                        match (orig_c, &round_tool.content[i]) {
                            (
                                AnthropicToolResultContent::Text { text: a },
                                AnthropicToolResultContent::Text { text: b },
                            ) => {
                                assert_eq!(a, b, "Text content at index {} was modified", i);
                            }
                            (
                                AnthropicToolResultContent::Image {
                                    source:
                                        AnthropicImageSource::Base64 {
                                            media_type: a_mt,
                                            data: a_data,
                                        },
                                },
                                AnthropicToolResultContent::Image {
                                    source:
                                        AnthropicImageSource::Base64 {
                                            media_type: b_mt,
                                            data: b_data,
                                        },
                                },
                            ) => {
                                assert_eq!(a_mt, b_mt, "Image mime-type changed at index {}", i);
                                assert_eq!(
                                    a_data, b_data,
                                    "Image base64 data changed at index {}",
                                    i
                                );
                            }
                            (o, r) => panic!(
                                "Content type changed during Responses roundtrip at index {}: original={:?} roundtrip={:?}",
                                i, o, r
                            ),
                        }
                    }
                }
                Err(e) => {
                    // Conversion returned an explicit error. That's acceptable
                    // per our "preserve or error" policy — but we still fail the
                    // test to highlight that provider conversion can't handle
                    // this case yet. Tests are RED on purpose.
                    panic!(
                        "openai_responses_to_anthropic_response returned error: {}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            // The forward conversion failed explicitly. Per policy this is
            // acceptable (better than silent loss), but we still want a RED
            // test to call attention to it.
            panic!(
                "anthropic_to_openai_responses_response returned error: {}",
                e
            );
        }
    }
}

// -----------------------------------------------------------------------------
// 3) Paranoid check for silent flattening (Gemini path)
// -----------------------------------------------------------------------------

#[cfg(feature = "anthropic-gemini")]
#[test]
fn gemini_roundtrip_must_not_stringify_structured_responses_silently() {
    // This test constructs an Anthropic ToolResult whose first text part is
    // machine-readable JSON. If the Gemini conversion silently stringifies,
    // flattens, or nests that JSON inside another JSON string, we should
    // detect it. The conversion path MUST either preserve the structure as
    // text EXACTLY, or fail loudly.

    use anthropic_ox::message::Content as AnthropicContent;
    use anthropic_ox::message::Role as AnthropicRole;
    use anthropic_ox::response::ChatResponse as AnthropicResponse;
    use anthropic_ox::tool::{
        ToolResult as AnthropicToolResult, ToolResultContent as AnthropicToolResultContent,
    };
    use conversion_ox::anthropic_gemini::{
        anthropic_to_gemini_response, gemini_to_anthropic_response,
    };

    // Structured JSON that could be accidentally double-serialized
    let structured: serde_json::Value = json!({
        "items": [
            {"id": 1, "value": "α"},
            {"id": 2, "value": "🔥"}
        ],
        "meta": {"timestamp": "2025-09-14T00:00:00Z"}
    });

    let orig_text = serde_json::to_string(&structured).expect("serialize should succeed");

    let original_tool_result = AnthropicToolResult {
        tool_use_id: "call_structured_1".to_string(),
        content: vec![AnthropicToolResultContent::Text {
            text: orig_text.clone(),
        }],
        is_error: None,
        cache_control: None,
    };

    let original_response = AnthropicResponse {
        id: "resp-struct-1".to_string(),
        r#type: "message".to_string(),
        role: AnthropicRole::Assistant,
        content: vec![AnthropicContent::ToolResult(original_tool_result.clone())],
        model: "claude-structured".to_string(),
        stop_reason: None,
        stop_sequence: None,
        usage: anthropic_ox::response::Usage::default(),
    };

    let gemini_resp = anthropic_to_gemini_response(original_response.clone()).unwrap();
    let roundtrip = gemini_to_anthropic_response(gemini_resp).unwrap();

    let round_tool = match roundtrip.content.get(0) {
        Some(AnthropicContent::ToolResult(tr)) => tr,
        other => panic!(
            "Expected ToolResult after Gemini roundtrip, got: {:?}",
            other
        ),
    };

    // Expect exactly the same single text block
    assert_eq!(
        round_tool.content.len(),
        1,
        "Expected exactly 1 content item in roundtrip"
    );

    match &round_tool.content[0] {
        AnthropicToolResultContent::Text { text } => {
            // If the implementation double-serialized the JSON it might look like
            // "{\"items\": ... }" (a JSON string) instead of the original
            // compact JSON. We require exact equality.
            assert_eq!(
                text, &orig_text,
                "Structured JSON text was altered or double-serialized during Gemini roundtrip.\nEXPECTED: {}\nGOT: {}",
                orig_text, text
            );
        }
        other => panic!("Expected text content after roundtrip, got: {:?}", other),
    }
}

// End of paranoid tests.
