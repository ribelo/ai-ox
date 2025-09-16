use openai_ox::{
    Message, OpenAI, ReasoningConfig, ResponseOutputContent, ResponseOutputItem, ResponsesInput,
    ResponsesRequest, ResponsesResponse,
};
use serde_json::json;

#[tokio::test]
async fn test_responses_request_serialization() {
    let request = ResponsesRequest::builder()
        .model("o3-mini")
        .input(ResponsesInput::text("What is 2+2?"))
        .reasoning(openai_ox::ReasoningConfig {
            effort: Some("medium".to_string()),
            summary: Some("auto".to_string()),
        })
        .include(vec!["reasoning.encrypted_content".to_string()])
        .max_output_tokens(100)
        .build();

    let json = serde_json::to_value(&request).unwrap();

    assert_eq!(json["model"], "o3-mini");
    assert_eq!(json["input"], "What is 2+2?");
    assert_eq!(json["reasoning"]["effort"], "medium");
    assert_eq!(json["reasoning"]["summary"], "auto");
    assert!(
        json["include"]
            .as_array()
            .unwrap()
            .contains(&json!("reasoning.encrypted_content"))
    );
    assert_eq!(json["max_output_tokens"], 100);
}

#[tokio::test]
async fn test_responses_request_with_messages() {
    let messages = vec![
        Message::user("Hello"),
        Message::assistant("Hi there! How can I help?"),
        Message::user("What is 2+2?"),
    ];

    let request = ResponsesRequest::builder()
        .model("gpt-5")
        .input(ResponsesInput::messages(messages.clone()))
        .reasoning(openai_ox::ReasoningConfig {
            effort: Some("high".to_string()),
            summary: Some("auto".to_string()),
        })
        .build();

    let json = serde_json::to_value(&request).unwrap();

    assert_eq!(json["model"], "gpt-5");
    assert_eq!(json["input"].as_array().unwrap().len(), 3);
    assert_eq!(json["reasoning"]["effort"], "high");
    assert_eq!(json["reasoning"]["summary"], "auto");
}

#[tokio::test]
async fn test_responses_response_deserialization() {
    let json = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 1234567890u64,
        "model": "o3-mini",
        "parallel_tool_calls": true,
        "tools": [],
        "output": [
             {
                 "type": "reasoning",
                 "id": "reasoning_1",
                 "summary": ["I need to add 2 and 2 together."],
                 "usage": {
                     "reasoning_tokens": 15
                 }
             },
             {
                 "type": "message",
                 "id": "msg_1",
                 "role": "assistant",
                 "content": [
                     {
                         "type": "text",
                         "text": "The answer is 4."
                     }
                 ],
                 "status": "completed"
             }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 15
        }
    });

    let response: ResponsesResponse = serde_json::from_value(json).unwrap();

    assert_eq!(response.id, "resp_123");
    assert_eq!(response.model, "o3-mini");
    assert_eq!(response.status.as_deref(), Some("completed"));
    assert_eq!(response.output.len(), 2);

    // Check reasoning item
    if let ResponseOutputItem::Reasoning { id, summary, .. } = &response.output[0] {
        assert_eq!(id, "reasoning_1");
        assert_eq!(
            summary[0].as_str().unwrap(),
            "I need to add 2 and 2 together."
        );
    } else {
        panic!("Expected reasoning item");
    }

    // Check message item
    if let ResponseOutputItem::Message { role, content, .. } = &response.output[1] {
        assert_eq!(role, "assistant");
        if let ResponseOutputContent::Text { text, .. } = &content[0] {
            assert_eq!(text, "The answer is 4.");
        } else {
            panic!("Expected text content");
        }
    } else {
        panic!("Expected message item");
    }

    // Check usage
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
    assert_eq!(usage.reasoning_tokens.unwrap(), 15);
}

#[tokio::test]
async fn test_responses_response_helper_methods() {
    let json = json!({
        "id": "resp_456",
        "object": "response",
        "created_at": 1234567890u64,
        "model": "gpt-5",
        "parallel_tool_calls": true,
        "tools": [],
        "output": [
             {
                 "type": "reasoning",
                 "id": "reasoning_1",
                 "summary": ["Thinking step 1"],
                 "encrypted_content": "encrypted_data_123"
             },
             {
                 "type": "reasoning",
                 "id": "reasoning_2",
                 "summary": ["Thinking step 2"]
             },
             {
                 "type": "message",
                 "id": "msg_2",
                 "role": "assistant",
                 "content": [
                     {
                         "type": "text",
                         "text": "Here's my analysis:"
                     }
                 ],
                 "status": "completed"
             },
             {
                 "type": "message",
                 "id": "msg_3",
                 "role": "assistant",
                 "content": [
                     {
                         "type": "text",
                         "text": "The final answer is 42."
                     }
                 ],
                 "status": "completed"
             }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 100,
            "total_tokens": 150,
            "reasoning_tokens": 75
        }
    });

    let response: ResponsesResponse = serde_json::from_value(json).unwrap();

    // Test helper methods
    assert!(response.is_completed());
    assert!(!response.is_in_progress());
    assert!(!response.is_failed());

    assert_eq!(response.reasoning_tokens(), 75);

    let reasoning_items = response.reasoning_items();
    assert_eq!(reasoning_items.len(), 2);
    assert_eq!(reasoning_items[0].0, "reasoning_1");
    assert_eq!(reasoning_items[1].0, "reasoning_2");

    let messages = response.messages();
    assert_eq!(messages.len(), 2);
    // Check the content of the second message (messages[1].2 is the content vector)
    if let ResponseOutputContent::Text { text, .. } = &messages[1].2[0] {
        assert_eq!(text, "The final answer is 42.");
    } else {
        panic!("Expected text content in second message");
    }

    let content = response.content().unwrap();
    assert!(content.contains("Thinking step 1"));
    assert!(content.contains("Thinking step 2"));
    assert!(content.contains("Here's my analysis:"));
    assert!(content.contains("The final answer is 42."));
}

#[tokio::test]
async fn test_reasoning_config_helpers() {
    let config1 = ReasoningConfig::with_effort("high");
    assert_eq!(config1.effort.unwrap(), "high");
    assert!(config1.summary.is_none());

    let config2 = ReasoningConfig::with_auto_summary();
    assert!(config2.effort.is_none());
    assert_eq!(config2.summary.unwrap(), "auto");

    let config3 = ReasoningConfig::with_effort_and_summary("medium");
    assert_eq!(config3.effort.unwrap(), "medium");
    assert_eq!(config3.summary.unwrap(), "auto");
}

#[tokio::test]
async fn test_responses_input_variants() {
    // Test text input
    let input1 = ResponsesInput::text("Hello world");
    let json1 = serde_json::to_value(&input1).unwrap();
    assert_eq!(json1, "Hello world");

    // Test messages input
    let messages = vec![Message::user("Hi"), Message::assistant("Hello!")];
    let input2 = ResponsesInput::messages(messages);
    let json2 = serde_json::to_value(&input2).unwrap();
    assert!(json2.is_array());
    assert_eq!(json2.as_array().unwrap().len(), 2);

    // Test round-trip deserialization
    let input1_deserialized: ResponsesInput = serde_json::from_value(json1).unwrap();
    if let ResponsesInput::Text(text) = input1_deserialized {
        assert_eq!(text, "Hello world");
    } else {
        panic!("Expected text input");
    }
}

#[tokio::test]
async fn test_client_responses_methods() {
    // This test would require mocking or a test server
    // For now, just test that the methods exist and can be called
    let client = OpenAI::new("test-key");

    let request = client
        .responses()
        .model("o3-mini")
        .input(ResponsesInput::text("Test"))
        .reasoning(openai_ox::ReasoningConfig {
            effort: Some("medium".to_string()),
            summary: None,
        })
        .build();

    assert_eq!(request.model, "o3-mini");

    // The actual send/stream methods would be tested with integration tests
    // using a mock server or the actual OpenAI API
}
