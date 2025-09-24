#[cfg(test)]
mod tests {
    use {{provider}}_ox::{{{Provider}}, Message, Model, Usage};

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.content, Some("Hello".to_string()));
        assert!(matches!(msg.role, {{provider}}_ox::message::Role::User));
    }

    #[test] 
    fn test_system_message() {
        let msg = Message::system("You are a helpful assistant");
        assert!(matches!(msg.role, {{provider}}_ox::message::Role::System));
        assert_eq!(msg.content, Some("You are a helpful assistant".to_string()));
    }

    #[test]
    fn test_assistant_message() {
        let msg = Message::assistant("I'm here to help!");
        assert!(matches!(msg.role, {{provider}}_ox::message::Role::Assistant));
        assert_eq!(msg.content, Some("I'm here to help!".to_string()));
    }

    #[test]
    fn test_tool_message() {
        let msg = Message::tool("call_123", "Result: 42");
        assert!(matches!(msg.role, {{provider}}_ox::message::Role::Tool));
        assert_eq!(msg.content, Some("Result: 42".to_string()));
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_chat_request_builder() {
        let request = {{provider}}_ox::ChatRequest::builder()
            .model("{{default_model}}")
            .user("Hello")
            .assistant("Hi there!")
            .temperature(0.7)
            .max_tokens(100)
            .build();

        assert_eq!(request.model, "{{default_model}}");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(100));
    }

    #[test] 
    fn test_model_string_conversion() {
        let model: Model = "{{model_1}}".into();
        assert_eq!(model, Model::{{Model1}});
        assert_eq!(model.as_str(), "{{model_1}}");
    }

    #[test]
    fn test_model_custom() {
        let model: Model = "custom-model".into();
        assert!(matches!(model, Model::Custom(ref s) if s == "custom-model"));
        assert_eq!(model.as_str(), "custom-model");
    }

    #[test]
    fn test_usage_calculation() {
        let usage = Usage::new(100, 50);
        assert_eq!(usage.prompt_tokens(), 100);
        assert_eq!(usage.completion_tokens(), 50);
        assert_eq!(usage.total_tokens(), 150);
        
        let cost = usage.calculate_cost(0.001, 0.002);
        assert!((cost - 0.2).abs() < f64::EPSILON); // 0.1 + 0.1 = 0.2
    }

    #[test]
    fn test_usage_addition() {
        let usage1 = Usage::new(100, 50);
        let usage2 = Usage::new(200, 75);
        let combined = usage1 + usage2;
        
        assert_eq!(combined.prompt_tokens(), 300);
        assert_eq!(combined.completion_tokens(), 125);
        assert_eq!(combined.total_tokens(), 425);
    }

    #[test]
    fn test_client_creation() {
        let client = {{Provider}}::new("test-key");
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.base_url, "{{base_url}}");
    }

    #[test]
    fn test_client_builder() {
        let client = {{Provider}}::builder()
            .api_key("test-key")
            .base_url("https://custom.api.com")
            .build();
        
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.base_url, "https://custom.api.com");
    }
}