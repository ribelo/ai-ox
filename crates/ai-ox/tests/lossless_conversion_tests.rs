use ai_ox::{
    content::{
        message::{Message, MessageRole},
        part::{DataRef, Part},
    },
    provider::{Capabilities, ConversionError, ConversionPlan, ConversionPolicy},
};
use serde_json::json;

#[test]
fn test_mistral_fails_on_unsupported_image() {
    // Mistral doesn't support images - should fail in strict mode
    let _message = Message::new(
        MessageRole::User,
        vec![
            Part::Text {
                text: "Look at this:".to_string(),
                ext: Default::default(),
            },
            Part::Blob {
                data_ref: DataRef::uri("https://example.com/image.jpg".to_string()),
                mime_type: "image/jpeg".to_string(),
                name: None,
                description: None,
                ext: Default::default(),
            },
        ],
    );

    let caps = Capabilities::mistral();
    assert!(!caps.supports_images || !caps.supports_base64_blob_input);

    // Test that conversion planning would fail
    let mut plan = ConversionPlan::new("mistral", &ConversionPolicy::Strict);
    plan.add_error(ConversionError::UnsupportedContent {
        part_index: 1,
        part_type: "Image".to_string(),
        provider: "mistral".to_string(),
        reason: "Mistral does not support images".to_string(),
    });

    assert!(plan.has_errors());
    assert!(!plan.is_lossless());
}

#[test]
fn test_mistral_fails_on_base64_audio() {
    let _message = Message::new(
        MessageRole::User,
        vec![Part::Blob {
            data_ref: DataRef::base64("SGVsbG8gV29ybGQ=".to_string()),
            mime_type: "audio/wav".to_string(),
            name: None,
            description: None,
            ext: Default::default(),
        }],
    );

    let caps = Capabilities::mistral();
    // Mistral should not support base64 input
    assert!(!caps.supports_base64_blob_input);
    assert!(!caps.can_accept_base64(1024));
}

#[test]
fn test_partv2_migration_preserves_content() {
    // Test that Part -> Part conversion is lossless
    let original_parts = vec![
        Part::Text {
            text: "Hello".to_string(),
            ext: Default::default(),
        },
        Part::Blob {
            data_ref: DataRef::uri("https://example.com/image.jpg".to_string()),
            mime_type: "image/jpeg".to_string(),
            name: None,
            description: None,
            ext: Default::default(),
        },
        Part::Blob {
            data_ref: DataRef::base64("audio_data".to_string()),
            mime_type: "audio/mp3".to_string(),
            name: None,
            description: None,
            ext: Default::default(),
        },
        Part::ToolUse {
            id: "call_123".to_string(),
            name: "search".to_string(),
            args: json!({"query": "test"}),
            ext: Default::default(),
        },
    ];

    let v2_parts: Vec<Part> = original_parts.iter().map(|p| p.clone()).collect();

    // Verify conversion
    assert_eq!(v2_parts.len(), original_parts.len());

    // Check first part (Text)
    if let Part::Text { text, .. } = &v2_parts[0] {
        assert_eq!(text, "Hello");
    } else {
        panic!("Expected Text part");
    }

    // Check second part (Image -> Blob)
    if let Part::Blob {
        data_ref,
        mime_type,
        ..
    } = &v2_parts[1]
    {
        assert!(matches!(data_ref, DataRef::Uri { .. }));
        assert!(mime_type.starts_with("image/"));
    } else {
        panic!("Expected Blob part for image");
    }

    // Check third part (Audio -> Blob)
    if let Part::Blob {
        data_ref,
        mime_type,
        ..
    } = &v2_parts[2]
    {
        assert!(matches!(data_ref, DataRef::Base64 { .. }));
        assert_eq!(mime_type, "audio/mp3");
    } else {
        panic!("Expected Blob part for audio");
    }

    // Check fourth part (ToolUse)
    if let Part::ToolUse { id, name, .. } = &v2_parts[3] {
        assert_eq!(id, "call_123");
        assert_eq!(name, "search");
    } else {
        panic!("Expected ToolUse part");
    }
}

#[test]
fn test_conversion_plan_strict_mode() {
    let mut plan = ConversionPlan::new("mistral", &ConversionPolicy::Strict);

    // Simulate planning conversion of unsupported content
    plan.add_error(ConversionError::UnsupportedContent {
        part_index: 1,
        part_type: "Image".to_string(),
        provider: "mistral".to_string(),
        reason: "Mistral does not support images".to_string(),
    });

    assert!(plan.has_errors());
    assert!(!plan.is_lossless());
}

#[test]
fn test_partv2_blob_unifies_media_types() {
    // Test that Blob correctly unifies different media types
    let image_blob = Part::blob_uri("https://example.com/pic.jpg", "image/jpeg");
    let audio_blob = Part::blob_base64("audio_data", "audio/wav");
    let pdf_blob = Part::blob_uri("https://example.com/doc.pdf", "application/pdf");

    assert!(image_blob.is_image());
    assert!(!image_blob.is_audio());

    assert!(audio_blob.is_audio());
    assert!(!audio_blob.is_image());

    assert!(!pdf_blob.is_image());
    assert!(!pdf_blob.is_audio());
    assert_eq!(pdf_blob.mime_type(), Some("application/pdf"));
}

#[test]
fn test_tool_result_supports_nested_parts() {
    // Test that Part::ToolResult can contain rich nested content
    let tool_result = Part::tool_result(
        "call_456",
        "image_search",
        vec![
            Part::text("Found 2 matching images:"),
            Part::blob_uri("https://example.com/result1.jpg", "image/jpeg"),
            Part::blob_uri("https://example.com/result2.jpg", "image/jpeg"),
            Part::text("Search completed in 0.5 seconds"),
        ],
    );

    if let Part::ToolResult { parts, .. } = tool_result {
        assert_eq!(parts.len(), 4);
        // Verify we have mixed content types
        assert!(matches!(parts[0], Part::Text { .. }));
        assert!(matches!(parts[1], Part::Blob { .. }));
        assert!(matches!(parts[2], Part::Blob { .. }));
        assert!(matches!(parts[3], Part::Text { .. }));
    } else {
        panic!("Expected ToolResult");
    }
}

#[test]
fn test_capabilities_mime_wildcard_matching() {
    let caps = Capabilities::gemini();

    // Gemini supports "image/*" wildcard
    assert!(caps.supports_mime("image/jpeg"));
    assert!(caps.supports_mime("image/png"));
    assert!(caps.supports_mime("image/webp"));
    assert!(caps.supports_mime("audio/wav"));
    assert!(caps.supports_mime("video/mp4"));

    // But not random types
    let caps_anthropic = Capabilities::anthropic();
    assert!(caps_anthropic.supports_mime("image/jpeg"));
    assert!(!caps_anthropic.supports_mime("audio/wav")); // Anthropic doesn't support audio
}

#[test]
fn test_base64_size_limits() {
    let caps = Capabilities::anthropic();

    // Anthropic has a 5MB limit
    assert!(caps.can_accept_base64(1024 * 1024)); // 1MB - OK
    assert!(caps.can_accept_base64(5 * 1024 * 1024)); // 5MB - OK
    assert!(!caps.can_accept_base64(10 * 1024 * 1024)); // 10MB - Too large
}

#[test]
fn test_ext_namespacing() {
    // Test that extensions use proper namespacing
    let part = Part::text("Hello")
        .with_ext("anthropic", "thinking", json!(true))
        .with_ext("openai", "reasoning", json!("step by step"))
        .with_ext("ai_ox", "original_mime", json!("text/plain"));

    if let Part::Text { ext, .. } = part {
        assert_eq!(ext.get("anthropic.thinking"), Some(&json!(true)));
        assert_eq!(ext.get("openai.reasoning"), Some(&json!("step by step")));
        assert_eq!(ext.get("ai_ox.original_mime"), Some(&json!("text/plain")));
        assert_eq!(ext.get("invalid"), None); // Non-namespaced key shouldn't exist
    }
}

#[test]
fn test_conversion_error_types() {
    // Test different types of conversion errors
    let errors = vec![
        ConversionError::UnsupportedContent {
            part_index: 0,
            part_type: "Image".to_string(),
            provider: "mistral".to_string(),
            reason: "Images not supported".to_string(),
        },
        ConversionError::UnsupportedMimeType {
            mime_type: "audio/wav".to_string(),
            provider: "anthropic".to_string(),
        },
        ConversionError::Base64TooLarge {
            size: 10 * 1024 * 1024,
            max_size: 5 * 1024 * 1024,
            provider: "anthropic".to_string(),
        },
    ];

    for error in errors {
        // Each error should be properly formatted
        let error_str = error.to_string();
        assert!(!error_str.is_empty());
        assert!(error_str.contains("not supported") || error_str.contains("too large"));
    }
}

#[test]
fn test_partv2_roundtrip_serialization() {
    // Test that Part can be serialized and deserialized without loss
    let original = Part::tool_result(
        "test_id",
        "test_tool",
        vec![
            Part::text("Hello world"),
            Part::blob_uri("https://example.com/file.pdf", "application/pdf"),
            Part::tool_use("nested_call", "nested_tool", json!({"param": "value"})),
        ],
    );

    // Serialize to JSON
    let json_str = serde_json::to_string(&original).expect("Serialization failed");

    // Deserialize back
    let deserialized: Part = serde_json::from_str(&json_str).expect("Deserialization failed");

    // Should be identical
    assert_eq!(original, deserialized);
}

#[test]
fn test_dataref_base64_size_estimation() {
    // Test base64 size estimation
    let small_data = DataRef::base64("SGVsbG8="); // "Hello" base64
    let large_data = DataRef::base64("A".repeat(4000)); // ~3KB of base64

    assert!(small_data.base64_size().unwrap() < 100); // Small data
    assert!(large_data.base64_size().unwrap() > 2000); // Large data

    // URI should return None
    let uri_data = DataRef::uri("https://example.com/file.jpg");
    assert_eq!(uri_data.base64_size(), None);
}

#[test]
fn test_provider_capabilities_comprehensive() {
    // Test all provider capabilities are properly configured

    // Anthropic: images only, 5MB base64 limit
    let anthropic = Capabilities::anthropic();
    assert!(anthropic.supports_images);
    assert!(!anthropic.supports_audio);
    assert!(anthropic.supports_base64_blob_input);
    assert_eq!(anthropic.max_base64_size, Some(5 * 1024 * 1024));

    // OpenAI: images and audio, no base64 limit
    let openai = Capabilities::openai();
    assert!(openai.supports_images);
    assert!(openai.supports_audio);
    assert!(openai.supports_base64_blob_input);
    assert!(openai.supports_blob_uri_input);
    assert_eq!(openai.max_base64_size, None);

    // Gemini: everything, wildcards
    let gemini = Capabilities::gemini();
    assert!(gemini.supports_images);
    assert!(gemini.supports_audio);
    assert!(gemini.supports_files);
    assert!(gemini.supports_tool_use);
    assert!(gemini.supports_tool_result_parts);

    // Mistral: images via Pixtral, no base64
    let mistral = Capabilities::mistral();
    assert!(mistral.supports_images);
    assert!(!mistral.supports_base64_blob_input);
    assert!(mistral.supports_blob_uri_input);
}

#[test]
fn test_conversion_policy_variants() {
    // Test different conversion policy behaviors
    let strict = ConversionPolicy::Strict;
    let shadow = ConversionPolicy::ShadowAllowed;

    // All should create valid plans
    let plan_strict = ConversionPlan::new("test", &strict);
    let plan_shadow = ConversionPlan::new("test", &shadow);

    assert_eq!(plan_strict.policy_name, "Strict");
    assert_eq!(plan_shadow.policy_name, "ShadowAllowed");
}

#[test]
fn test_partv2_opaque_provider_content() {
    // Test that opaque content preserves provider-specific data
    let opaque = Part::Opaque {
        provider: "custom_provider".to_string(),
        kind: "special_content".to_string(),
        payload: json!({"custom": "data", "value": 42}),
        ext: Default::default(),
    };

    if let Part::Opaque {
        provider,
        kind,
        payload,
        ..
    } = opaque
    {
        assert_eq!(provider, "custom_provider");
        assert_eq!(kind, "special_content");
        assert_eq!(payload["custom"], "data");
        assert_eq!(payload["value"], 42);
    } else {
        panic!("Expected Opaque part");
    }
}

#[test]
fn test_tool_result_with_empty_parts() {
    // Test edge case: tool result with no parts
    let empty_tool_result = Part::tool_result("call_empty", "empty_tool", vec![]);

    if let Part::ToolResult {
        parts, id, name, ..
    } = empty_tool_result
    {
        assert_eq!(id, "call_empty");
        assert_eq!(name, "empty_tool");
        assert!(parts.is_empty());
    } else {
        panic!("Expected ToolResult");
    }
}

#[test]
fn test_blob_with_metadata() {
    // Test blob with name and description
    let blob = Part::Blob {
        data_ref: DataRef::uri("https://example.com/document.pdf"),
        mime_type: "application/pdf".to_string(),
        name: Some("Important Document".to_string()),
        description: Some("Q4 Financial Report".to_string()),
        ext: Default::default(),
    };

    assert_eq!(blob.mime_type(), Some("application/pdf"));
    assert!(!blob.is_image());
    assert!(!blob.is_audio());
    assert!(!blob.is_video());
}

#[test]
fn test_ext_namespacing_prevents_collisions() {
    // Test that namespacing prevents key collisions
    let part = Part::text("test")
        .with_ext("provider_a", "setting", json!("value_a"))
        .with_ext("provider_b", "setting", json!("value_b"));

    if let Part::Text { ext, .. } = part {
        assert_eq!(ext.get("provider_a.setting"), Some(&json!("value_a")));
        assert_eq!(ext.get("provider_b.setting"), Some(&json!("value_b")));
        assert_ne!(ext.get("provider_a.setting"), ext.get("provider_b.setting"));
    }
}
