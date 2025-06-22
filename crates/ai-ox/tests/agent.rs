use std::sync::Arc;

use ai_ox::{
    ModelResponse,
    agent::{Agent, error::AgentError},
    content::{Message, MessageRole, Part},
    errors::GenerateContentError,
    model::{Model, request::ModelRequest, response::RawStructuredResponse},
    tool::{ToolBox, ToolCall, ToolError, ToolResult},
    usage::Usage,
};
use futures_util::{stream::BoxStream, future::BoxFuture};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Mock model for testing
#[derive(Debug, Clone)]
struct MockModel {
    responses: Vec<ModelResponse>,
    current_response: std::sync::Arc<std::sync::Mutex<usize>>,
    received_messages: std::sync::Arc<std::sync::Mutex<Vec<Message>>>,
}

impl MockModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses,
            current_response: Arc::new(std::sync::Mutex::new(0)),
            received_messages: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn with_text_response(text: &str) -> Self {
        let response = ModelResponse {
            message: Message::new(
                MessageRole::Assistant,
                vec![Part::Text {
                    text: text.to_string(),
                }],
            ),
            model_name: "mock-model".to_string(),
            usage: Usage::default(),
            vendor_name: "mock".to_string(),
        };
        Self::new(vec![response])
    }
}

impl Model for MockModel {
    fn model(&self) -> &str {
        "mock-model"
    }

    fn request(
        &self,
        request: ModelRequest,
    ) -> BoxFuture<'_, Result<ModelResponse, GenerateContentError>> {
        let messages = request.messages;
        Box::pin(async move {
            let mut received = self.received_messages.lock().unwrap();
            *received = messages;

            let mut index = self.current_response.lock().unwrap();
            if *index < self.responses.len() {
                let response = self.responses[*index].clone();
                *index += 1;
                Ok(response)
            } else {
                Err(GenerateContentError::NoResponse)
            }
        })
    }

    fn request_stream(
        &self,
        _request: ModelRequest,
    ) -> BoxStream<'_, Result<ai_ox::content::delta::MessageStreamEvent, GenerateContentError>>
    {
        use futures_util::stream;
        Box::pin(stream::empty())
    }

    fn request_structured_internal(
        &self,
        _request: ModelRequest,
        _schema: String,
    ) -> BoxFuture<'_, Result<RawStructuredResponse, GenerateContentError>> {
        Box::pin(async {
            Err(GenerateContentError::unsupported_feature(
                "structured response",
            ))
        })
    }
}

// Mock ToolBox for testing
#[derive(Debug)]
struct MockToolBox {
    should_fail: bool,
}

impl MockToolBox {
    fn new() -> Self {
        Self { should_fail: false }
    }

    fn with_failure() -> Self {
        Self { should_fail: true }
    }
}

impl ToolBox for MockToolBox {
    fn tools(&self) -> Vec<ai_ox::tool::Tool> {
        vec![]
    }

    fn invoke(
        &self,
        call: ToolCall,
    ) -> futures_util::future::BoxFuture<Result<ToolResult, ToolError>> {
        Box::pin(async move {
            if self.should_fail {
                Err(ToolError::execution(
                    "mock_tool",
                    std::io::Error::new(std::io::ErrorKind::Other, "Mock tool failure"),
                ))
            } else {
                Ok(ToolResult {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    response: vec![Message::new(
                        MessageRole::Assistant,
                        vec![Part::Text {
                            text: "Tool executed successfully".to_string(),
                        }],
                    )],
                })
            }
        })
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TestResponse {
    message: String,
    count: u32,
}

#[tokio::test]
async fn test_agent_builder() {
    let model = MockModel::with_text_response("Hello, world!");

    let agent = Agent::builder(Arc::new(model))
        .system_instruction("You are a helpful assistant")
        .max_iterations(5)
        .build();

    assert_eq!(
        agent.system_instruction(),
        Some("You are a helpful assistant")
    );
    assert_eq!(agent.max_iterations(), 5);
}

#[tokio::test]
async fn test_agent_generate_success() {
    let model = MockModel::with_text_response("Hello, world!");
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Say hello".to_string(),
        }],
    )];

    let response = agent.generate(messages).await.unwrap();

    // Extract text from response
    if let Part::Text { text } = &response.message.content[0] {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text response");
    }
}

#[tokio::test]
async fn test_agent_generate_no_response() {
    let model = MockModel::new(vec![]);
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Say hello".to_string(),
        }],
    )];

    let result = agent.generate(messages).await;

    assert!(matches!(
        result,
        Err(AgentError::Api(GenerateContentError::NoResponse))
    ));
}

#[tokio::test]
async fn test_agent_execute_without_tools() {
    let model = MockModel::with_text_response("Hello, world!");
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Say hello".to_string(),
        }],
    )];

    let response = agent.run(messages).await.unwrap();

    // Extract text from response
    if let Part::Text { text } = &response.message.content[0] {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text response");
    }
}

#[tokio::test]
async fn test_agent_execute_with_tools() {
    let model = MockModel::with_text_response("Hello, world!");
    let toolbox = MockToolBox::new();
    let agent = Agent::builder(Arc::new(model))
        .toolbox(toolbox)
        .build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Say hello".to_string(),
        }],
    )];

    let response = agent.run(messages).await.unwrap();

    // Since our mock doesn't return tool calls, it should behave like generate
    if let Part::Text { text } = &response.message.content[0] {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text response");
    }
}

#[tokio::test]
async fn test_agent_max_iterations() {
    // Create a model that returns responses requiring tool calls (in a real scenario)
    let responses = vec![
        ModelResponse {
            message: Message::new(
                MessageRole::Assistant,
                vec![Part::Text {
                    text: "Iteration 1".to_string(),
                }],
            ),
            model_name: "mock-model".to_string(),
            usage: Usage::default(),
            vendor_name: "mock".to_string(),
        },
        ModelResponse {
            message: Message::new(
                MessageRole::Assistant,
                vec![Part::Text {
                    text: "Iteration 2".to_string(),
                }],
            ),
            model_name: "mock-model".to_string(),
            usage: Usage::default(),
            vendor_name: "mock".to_string(),
        },
    ];

    let model = MockModel::new(responses);
    let agent = Agent::builder(Arc::new(model))
        .max_iterations(1) // Set very low limit
        .build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Test".to_string(),
        }],
    )];

    // Since our mock model doesn't actually return tool calls,
    // this test verifies the structure but won't actually hit max iterations
    let response = agent.run(messages).await.unwrap();
    assert!(response.message.content.len() > 0);
}

#[tokio::test]
async fn test_agent_generate_typed_unsupported() {
    let model = MockModel::with_text_response(r#"{"message": "test", "count": 42}"#);
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return JSON".to_string(),
        }],
    )];

    let result: Result<ai_ox::model::response::StructuredResponse<TestResponse>, _> = agent.generate_typed(messages).await;

    // Should succeed with fallback parsing
    let response: ai_ox::model::response::StructuredResponse<TestResponse> = result.unwrap();
    assert_eq!(response.data.message, "test");
    assert_eq!(response.data.count, 42);
}

#[tokio::test]
async fn test_agent_generate_typed_invalid_json() {
    let model = MockModel::with_text_response("Not valid JSON");
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Return JSON".to_string(),
        }],
    )];

    let result: Result<ai_ox::model::response::StructuredResponse<TestResponse>, _> = agent.generate_typed(messages).await;

    assert!(matches!(
        result,
        Err(AgentError::ResponseParsingFailed { .. })
    ));
}

#[tokio::test]
async fn test_agent_execute_typed() {
    let model = MockModel::with_text_response(r#"{"message": "executed", "count": 1}"#);
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Execute and return JSON".to_string(),
        }],
    )];

    let result = agent.execute_typed(messages).await;

    let response: ai_ox::model::response::StructuredResponse<TestResponse> = result.unwrap();
    assert_eq!(response.data.message, "executed");
    assert_eq!(response.data.count, 1);
}

#[tokio::test]
async fn test_agent_system_instruction() {
    let model = Arc::new(MockModel::with_text_response("Hello, I'm an assistant!"));
    let agent = Agent::builder(model.clone())
        .system_instruction("You are a helpful AI assistant")
        .build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "Who are you?".to_string(),
        }],
    )];

    let _response = agent.generate(messages).await.unwrap();

    // Check that only the user message was passed to the model
    // (system instruction is handled separately in ModelRequest)
    let received = model.received_messages.lock().unwrap();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].role, MessageRole::User);
}

#[tokio::test]
async fn test_agent_set_system_instruction() {
    let model = Arc::new(MockModel::with_text_response("Hello!"));
    let mut agent = Agent::builder(model.clone())
        .system_instruction("Initial instruction")
        .build();

    assert_eq!(agent.system_instruction(), Some("Initial instruction"));

    agent.set_system_instruction("New instruction");
    assert_eq!(agent.system_instruction(), Some("New instruction"));

    // Test that the system instruction is set correctly
    // (in the new design, system instructions don't appear in messages)
    assert_eq!(agent.system_instruction(), Some("New instruction"));

    agent.clear_system_instruction();
    assert_eq!(agent.system_instruction(), None);
}

#[tokio::test]
async fn test_agent_error_handling() {
    let model = MockModel::new(vec![]);
    let agent = Agent::builder(Arc::new(model)).build();

    let messages = vec![Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "This will fail".to_string(),
        }],
    )];

    let result = agent.generate(messages).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        AgentError::Api(GenerateContentError::NoResponse) => {
            // Expected error
        }
        other => panic!("Unexpected error: {:?}", other),
    }
}
