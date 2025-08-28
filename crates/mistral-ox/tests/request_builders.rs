#[cfg(test)]
mod tests {
    use mistral_ox::*;
    use mistral_ox::message::{Message, Messages};

    #[test]
    fn test_embeddings_request_builder() {
        let request = EmbeddingsRequest::builder()
            .model("mistral-embed".to_string())
            .input(EmbeddingInput::Single("Hello world".to_string()))
            .encoding_format("float".to_string())
            .build();

        assert_eq!(request.model, "mistral-embed");
        assert!(matches!(request.input, EmbeddingInput::Single(ref s) if s == "Hello world"));
        assert_eq!(request.encoding_format, Some("float".to_string()));
    }

    #[test]
    fn test_moderation_request_builder() {
        let request = ModerationRequest::builder()
            .input(ModerationInput::Single("This is a test message".to_string()))
            .model("mistral-moderation-latest".to_string())
            .build();

        assert!(matches!(request.input, ModerationInput::Single(ref s) if s == "This is a test message"));
        assert_eq!(request.model, Some("mistral-moderation-latest".to_string()));
    }

    #[test]
    fn test_chat_moderation_request_builder() {
        let messages = Messages::new(vec![Message::user("Hello"), Message::assistant("Hi there!")]);
        let request = ChatModerationRequest::builder()
            .model("mistral-moderation-latest".to_string())
            .messages(messages.clone())
            .build();

        assert_eq!(request.model, "mistral-moderation-latest");
        assert_eq!(request.messages.len(), 2);
    }

    #[test]
    fn test_fine_tuning_request_builder() {
        let training_files = vec![TrainingFile {
            file_id: "file-abc123".to_string(),
            weight: Some(1.0),
        }];

        let hyperparameters = FineTuningHyperparameters::builder()
            .training_steps(100)
            .learning_rate(0.001)
            .weight_decay(0.01)
            .build();

        let request = FineTuningRequest::builder()
            .model("mistral-small".to_string())
            .training_files(training_files.clone())
            .hyperparameters(hyperparameters)
            .suffix("custom-model".to_string())
            .auto_start(true)
            .build();

        assert_eq!(request.model, "mistral-small");
        assert_eq!(request.training_files, training_files);
        assert_eq!(request.suffix, Some("custom-model".to_string()));
        assert_eq!(request.auto_start, Some(true));
    }

    #[test]
    fn test_batch_job_request_builder() {
        let request = BatchJobRequest::builder()
            .input_file_id("file-batch123".to_string())
            .endpoint("/v1/chat/completions".to_string())
            .completion_window("24h".to_string())
            .build();

        assert_eq!(request.input_file_id, "file-batch123");
        assert_eq!(request.endpoint, "/v1/chat/completions");
        assert_eq!(request.completion_window, "24h");
    }

    #[test]
    fn test_fim_request_builder() {
        let request = FimRequest::builder()
            .model("codestral-latest".to_string())
            .prompt("def hello():".to_string())
            .suffix("    return 'world'".to_string())
            .max_tokens(100)
            .temperature(0.7)
            .build();

        assert_eq!(request.model, "codestral-latest");
        assert_eq!(request.prompt, "def hello():");
        assert_eq!(request.suffix, Some("    return 'world'".to_string()));
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_agents_request_builder() {
        let messages = Messages::new(vec![Message::user("Solve this math problem: 2 + 2")]);
        let request = AgentsRequest::builder()
            .messages(messages.clone())
            .agent_id("agent-math-tutor".to_string())
            .max_tokens(200)
            .temperature(0.1)
            .build();

        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.agent_id, Some("agent-math-tutor".to_string()));
        assert_eq!(request.max_tokens, Some(200));
        assert_eq!(request.temperature, Some(0.1));
    }

    #[test]
    fn test_training_file_creation() {
        let training_file = TrainingFile {
            file_id: "file-123".to_string(),
            weight: Some(0.8),
        };

        assert_eq!(training_file.file_id, "file-123");
        assert_eq!(training_file.weight, Some(0.8));
    }

    #[test]
    fn test_fine_tuning_hyperparameters() {
        let hyperparams = FineTuningHyperparameters::builder()
            .training_steps(500)
            .learning_rate(0.0001)
            .weight_decay(0.1)
            .warmup_fraction(0.05)
            .epochs(3.0)
            .fim_ratio(0.9)
            .build();

        assert_eq!(hyperparams.training_steps, 500);
        assert_eq!(hyperparams.learning_rate, 0.0001);
        assert_eq!(hyperparams.weight_decay, Some(0.1));
        assert_eq!(hyperparams.warmup_fraction, Some(0.05));
        assert_eq!(hyperparams.epochs, Some(3.0));
        assert_eq!(hyperparams.fim_ratio, Some(0.9));
    }
}