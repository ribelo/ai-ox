use anthropic_ox::{
    message::{Content, Message, Messages, Role, StringOrContents, Text},
    request::ChatRequest,
};
use serde_json;

#[test]
fn test_chat_request_with_string_content() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": "Hello world"
            }
        ]
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize ChatRequest: {:?}",
        result.err()
    );

    let request = result.unwrap();
    assert_eq!(request.model, "claude-3-5-sonnet-20241022");
    assert_eq!(request.max_tokens, 4096);
    assert_eq!(request.messages.len(), 1);

    let message = &request.messages[0];
    assert_eq!(message.role, Role::User);
    match &message.content {
        StringOrContents::String(s) => assert_eq!(s, "Hello world"),
        StringOrContents::Contents(_) => panic!("Expected String variant"),
    }
}

#[test]
fn test_chat_request_with_array_content() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022", 
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world"}
                ]
            }
        ]
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize ChatRequest: {:?}",
        result.err()
    );

    let request = result.unwrap();
    assert_eq!(request.messages.len(), 1);

    let message = &request.messages[0];
    match &message.content {
        StringOrContents::String(_) => panic!("Expected Contents variant"),
        StringOrContents::Contents(contents) => {
            assert_eq!(contents.len(), 1);
            match &contents[0] {
                Content::Text(text) => assert_eq!(text.text, "Hello world"),
                _ => panic!("Expected Text content"),
            }
        }
    }
}

#[test]
fn test_chat_request_multiple_messages_mixed_content() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": "Simple string message"
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": "Array response"}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"}
                ]
            }
        ]
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize ChatRequest: {:?}",
        result.err()
    );

    let request = result.unwrap();
    assert_eq!(request.messages.len(), 3);

    // First message: string content
    let msg1 = &request.messages[0];
    assert_eq!(msg1.role, Role::User);
    match &msg1.content {
        StringOrContents::String(s) => assert_eq!(s, "Simple string message"),
        StringOrContents::Contents(_) => panic!("Expected String variant for message 1"),
    }

    // Second message: array content
    let msg2 = &request.messages[1];
    assert_eq!(msg2.role, Role::Assistant);
    match &msg2.content {
        StringOrContents::String(_) => panic!("Expected Contents variant for message 2"),
        StringOrContents::Contents(contents) => {
            assert_eq!(contents.len(), 1);
            match &contents[0] {
                Content::Text(text) => assert_eq!(text.text, "Array response"),
                _ => panic!("Expected Text content"),
            }
        }
    }

    // Third message: multiple content items
    let msg3 = &request.messages[2];
    assert_eq!(msg3.role, Role::User);
    match &msg3.content {
        StringOrContents::String(_) => panic!("Expected Contents variant for message 3"),
        StringOrContents::Contents(contents) => {
            assert_eq!(contents.len(), 2);
            match &contents[0] {
                Content::Text(text) => assert_eq!(text.text, "First part"),
                _ => panic!("Expected Text content"),
            }
            match &contents[1] {
                Content::Text(text) => assert_eq!(text.text, "Second part"),
                _ => panic!("Expected Text content"),
            }
        }
    }
}

#[test]
fn test_chat_request_streaming() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }
        ],
        "stream": true
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize streaming ChatRequest: {:?}",
        result.err()
    );

    let request = result.unwrap();
    assert_eq!(request.stream, Some(true));
}

#[test]
fn test_chat_request_with_system_prompt() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "system": "You are Claude Code, Anthropic's official CLI for Claude.",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize ChatRequest with system: {:?}",
        result.err()
    );

    let request = result.unwrap();
    match request.system {
        Some(system) => assert_eq!(
            system.as_string(),
            "You are Claude Code, Anthropic's official CLI for Claude."
        ),
        None => panic!("Expected system prompt"),
    }
}

#[test]
fn test_real_claude_code_simple_request() {
    // This is the exact format Claude Code sends for a simple message
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello"
                    }
                ]
            }
        ],
        "stream": true
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize real Claude Code request: {:?}",
        result.err()
    );

    let request = result.unwrap();
    assert_eq!(request.model, "claude-3-5-sonnet-20241022");
    assert_eq!(request.stream, Some(true));
    assert_eq!(request.messages.len(), 1);

    let message = &request.messages[0];
    match &message.content {
        StringOrContents::Contents(contents) => {
            assert_eq!(contents.len(), 1);
            match &contents[0] {
                Content::Text(text) => assert_eq!(text.text, "hello"),
                _ => panic!("Expected Text content"),
            }
        }
        StringOrContents::String(_) => panic!("Real Claude Code sends array format"),
    }
}

#[test]
fn test_real_claude_code_complex_request() {
    // This simulates Claude Code sending multiple content blocks
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please help me with this code:"
                    },
                    {
                        "type": "text", 
                        "text": "fn main() {\n    println!(\"Hello, world!\");\n}"
                    }
                ]
            }
        ],
        "stream": true
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Failed to deserialize complex Claude Code request: {:?}",
        result.err()
    );

    let request = result.unwrap();
    let message = &request.messages[0];
    match &message.content {
        StringOrContents::Contents(contents) => {
            assert_eq!(contents.len(), 2);
            match &contents[0] {
                Content::Text(text) => assert_eq!(text.text, "Please help me with this code:"),
                _ => panic!("Expected Text content"),
            }
            match &contents[1] {
                Content::Text(text) => assert_eq!(
                    text.text,
                    "fn main() {\n    println!(\"Hello, world!\");\n}"
                ),
                _ => panic!("Expected Text content"),
            }
        }
        StringOrContents::String(_) => panic!("Expected Contents variant"),
    }
}

#[test]
fn test_serialization_roundtrip_string_content() {
    let original = ChatRequest {
        model: "claude-3-5-sonnet-20241022".to_string(),
        max_tokens: 4096,
        messages: Messages::from(vec![Message {
            role: Role::User,
            content: StringOrContents::String("Hello world".to_string()),
        }]),
        system: None,
        metadata: None,
        stop_sequences: None,
        stream: None,
        temperature: None,
        tool_choice: None,
        tools: None,
        top_k: None,
        top_p: None,
        thinking: None,
    };

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: ChatRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(original.model, deserialized.model);
    assert_eq!(original.messages.len(), deserialized.messages.len());

    // Check content specifically
    match (
        &original.messages[0].content,
        &deserialized.messages[0].content,
    ) {
        (StringOrContents::String(orig), StringOrContents::String(deser)) => {
            assert_eq!(orig, deser);
        }
        _ => panic!("Content type mismatch in roundtrip"),
    }
}

#[test]
fn test_serialization_roundtrip_array_content() {
    let original = ChatRequest {
        model: "claude-3-5-sonnet-20241022".to_string(),
        max_tokens: 4096,
        messages: Messages::from(vec![Message {
            role: Role::User,
            content: StringOrContents::Contents(vec![Content::Text(Text::new(
                "Hello world".to_string(),
            ))]),
        }]),
        system: None,
        metadata: None,
        stop_sequences: None,
        stream: None,
        temperature: None,
        tool_choice: None,
        tools: None,
        top_k: None,
        top_p: None,
        thinking: None,
    };

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: ChatRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(original.messages.len(), deserialized.messages.len());

    // Check content specifically
    match (
        &original.messages[0].content,
        &deserialized.messages[0].content,
    ) {
        (StringOrContents::Contents(orig), StringOrContents::Contents(deser)) => {
            assert_eq!(orig.len(), deser.len());
            match (&orig[0], &deser[0]) {
                (Content::Text(orig_text), Content::Text(deser_text)) => {
                    assert_eq!(orig_text.text, deser_text.text);
                }
                _ => panic!("Text content mismatch in roundtrip"),
            }
        }
        _ => panic!("Content type mismatch in roundtrip"),
    }
}

#[test]
fn test_edge_case_empty_content_array() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": []
            }
        ]
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Should handle empty content array: {:?}",
        result.err()
    );

    let request = result.unwrap();
    let message = &request.messages[0];
    match &message.content {
        StringOrContents::Contents(contents) => {
            assert_eq!(contents.len(), 0);
        }
        StringOrContents::String(_) => panic!("Expected empty Contents variant"),
    }
}

#[test]
fn test_failed_claude_code_request_with_system_array() {
    // This is the exact JSON that failed from the logs
    let json = r#"{
        "model":"claude-3-5-haiku-20241022",
        "max_tokens":512,
        "messages":[{
            "role":"user",
            "content":"Please write a 5-10 word title for the following conversation:\n\n[Last 3 of 251 messages]\n\nUser: \n\nClaude: Perfect! Now I've added request body logging for failed ChatRequest parsing. This will help us capture the exact JSON that Claude Code is sending, so we can write proper tests in ai-ox to handle those specific cases.\n\nThe updated logging will show:\n1. The serde error message \n2. **The complete failed request body** - this is what we need to create accurate tests\n\nNow when you trigger a Claude Code request and it fails, we'll see the exact JSON payload in the logs at `/home/ribelo/.local/share/setu/logs/setu.2025-08-23`, and we can use that to write comprehensive tests in ai-ox that match the real-world data format.\n\nRespond with the title for the conversation and nothing else."
        }],
        "system":[{
            "type":"text",
            "text":"Summarize this coding conversation in under 50 characters.\nCapture the main task, key files, problems addressed, and current status.",
            "cache_control":{"type":"ephemeral"}
        }],
        "temperature":0,
        "metadata":{"user_id":"user_32c7d5469e4065450d5863d6df846e2ffab1f0e97f403d5ae7101b3e5c12337a_account_8d202d14-33f8-4e3d-a4b9-68273f8e9afa_session_1fabb14f-4fd9-4371-b901-ef0c8030f120"},
        "stream":true
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    // This currently fails but should pass after we fix the system field handling
    assert!(
        result.is_ok(),
        "Failed to deserialize real Claude Code request with system array: {:?}",
        result.err()
    );
}

#[test]
fn test_edge_case_empty_string_content() {
    let json = r#"{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": ""
            }
        ]
    }"#;

    let result: Result<ChatRequest, _> = serde_json::from_str(json);
    assert!(
        result.is_ok(),
        "Should handle empty string content: {:?}",
        result.err()
    );

    let request = result.unwrap();
    let message = &request.messages[0];
    match &message.content {
        StringOrContents::String(s) => assert_eq!(s, ""),
        StringOrContents::Contents(_) => panic!("Expected String variant"),
    }
}
