#[cfg(test)]
mod tests {
    use openai_ox::*;

    #[test]
    fn test_embeddings_request_builder() {
        let request = EmbeddingsRequest::builder()
            .model("text-embedding-3-small".to_string())
            .input(EmbeddingInput::Single("Hello world".to_string()))
            .encoding_format("float".to_string())
            .build();

        assert_eq!(request.model, "text-embedding-3-small");
        assert!(matches!(request.input, EmbeddingInput::Single(ref s) if s == "Hello world"));
        assert_eq!(request.encoding_format, Some("float".to_string()));
    }

    #[test]
    fn test_embeddings_request_multiple_inputs() {
        let inputs = vec!["First text".to_string(), "Second text".to_string()];
        let request = EmbeddingsRequest::builder()
            .model("text-embedding-3-small".to_string())
            .input(EmbeddingInput::Multiple(inputs.clone()))
            .build();

        assert!(matches!(request.input, EmbeddingInput::Multiple(ref v) if v == &inputs));
    }

    #[test]
    fn test_moderation_request_builder() {
        let request = ModerationRequest::builder()
            .input(ModerationInput::Single(
                "This is a test message".to_string(),
            ))
            .model("text-moderation-latest".to_string())
            .build();

        assert!(
            matches!(request.input, ModerationInput::Single(ref s) if s == "This is a test message")
        );
        assert_eq!(request.model, Some("text-moderation-latest".to_string()));
    }

    #[test]
    fn test_image_request_builder() {
        let request = ImageRequest::builder()
            .prompt("A beautiful sunset over mountains".to_string())
            .model("dall-e-3".to_string())
            .n(1)
            .quality("hd".to_string())
            .response_format("url".to_string())
            .size("1024x1024".to_string())
            .style("vivid".to_string())
            .build();

        assert_eq!(request.prompt, "A beautiful sunset over mountains");
        assert_eq!(request.model, Some("dall-e-3".to_string()));
        assert_eq!(request.n, Some(1));
        assert_eq!(request.quality, Some("hd".to_string()));
        assert_eq!(request.response_format, Some("url".to_string()));
        assert_eq!(request.size, Some("1024x1024".to_string()));
        assert_eq!(request.style, Some("vivid".to_string()));
    }

    #[test]
    fn test_audio_request_builder() {
        let audio_data = vec![1, 2, 3, 4, 5];
        let request = AudioRequest::builder()
            .file(audio_data.clone())
            .filename("test.mp3".to_string())
            .model("whisper-1".to_string())
            .language("en".to_string())
            .prompt("This is a test".to_string())
            .response_format("json".to_string())
            .temperature(0.5)
            .build();

        assert_eq!(request.file, audio_data);
        assert_eq!(request.filename, "test.mp3");
        assert_eq!(request.model, "whisper-1");
        assert_eq!(request.language, Some("en".to_string()));
        assert_eq!(request.prompt, Some("This is a test".to_string()));
        assert_eq!(request.response_format, Some("json".to_string()));
        assert_eq!(request.temperature, Some(0.5));
    }

    #[test]
    fn test_fine_tuning_request_builder() {
        let request = FineTuningRequest::builder()
            .model("gpt-3.5-turbo".to_string())
            .training_file("file-abc123".to_string())
            .validation_file("file-def456".to_string())
            .suffix("custom-model".to_string())
            .build();

        assert_eq!(request.model, "gpt-3.5-turbo");
        assert_eq!(request.training_file, "file-abc123");
        assert_eq!(request.validation_file, Some("file-def456".to_string()));
        assert_eq!(request.suffix, Some("custom-model".to_string()));
    }

    #[test]
    fn test_assistant_request_builder() {
        let request = AssistantRequest::builder()
            .model("gpt-4".to_string())
            .name("Math Tutor".to_string())
            .description("A helpful math tutor".to_string())
            .instructions("You are a math tutor who helps students solve problems".to_string())
            .build();

        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.name, Some("Math Tutor".to_string()));
        assert_eq!(
            request.description,
            Some("A helpful math tutor".to_string())
        );
        assert_eq!(
            request.instructions,
            Some("You are a math tutor who helps students solve problems".to_string())
        );
    }
}
