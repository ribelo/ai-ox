use crate::content::{
    message::{Message, MessageRole},
    part::{ImageSource, Part},
};
use gemini_ox::content::{Content as GeminiContent, Part as GeminiPart, Role as GeminiRole};

/// Converts an `ai-ox` `Message` to a `gemini-ox` `Content`.
impl From<Message> for GeminiContent {
    fn from(message: Message) -> Self {
        let role = match message.role {
            MessageRole::User => GeminiRole::User,
            MessageRole::Assistant => GeminiRole::Model,
        };

        let parts: Vec<GeminiPart> = message.content.into_iter().map(Into::into).collect();

        GeminiContent { role, parts }
    }
}

/// Converts an `ai-ox` `Part` to a `gemini-ox` `Part`.
impl From<Part> for GeminiPart {
    fn from(part: Part) -> Self {
        use gemini_ox::content::{Blob, FileData as GeminiFileData, PartData, Text};

        let data = match part {
            Part::Text { text } => PartData::Text(Text::new(text)),
            Part::Image { source } => match source {
                ImageSource::Base64 { media_type, data } => {
                    PartData::InlineData(Blob::new(media_type, data))
                }
            },
            Part::File(file_data) => {
                let gemini_file_data = if let Some(display_name) = file_data.display_name {
                    GeminiFileData::new_with_display_name(
                        file_data.file_uri,
                        file_data.mime_type,
                        display_name,
                    )
                } else {
                    GeminiFileData::new(file_data.file_uri, file_data.mime_type)
                };
                PartData::FileData(gemini_file_data)
            }
            Part::ToolResult {
                call_id,
                name,
                content,
            } => {
                // Convert tool result to text representation for Gemini
                let text_content =
                    format!("Tool '{}' (call_id: {}) result: {}", name, call_id, content);
                PartData::Text(Text::new(text_content))
            }
        };

        GeminiPart {
            thought: None,
            video_metadata: None,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::part::FileData;

    #[test]
    fn test_message_to_content_user_role() {
        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Hello, world!".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let content: GeminiContent = message.into();
        assert_eq!(content.role, GeminiRole::User);
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn test_message_to_content_assistant_role() {
        let message = Message {
            role: MessageRole::Assistant,
            content: vec![Part::Text {
                text: "Hi there!".to_string(),
            }],
            timestamp: chrono::Utc::now(),
        };

        let content: GeminiContent = message.into();
        assert_eq!(content.role, GeminiRole::Model);
        assert_eq!(content.parts.len(), 1);
    }

    #[test]
    fn test_text_part_conversion() {
        let part = Part::Text {
            text: "Test text".to_string(),
        };

        let gemini_part: GeminiPart = part.into();
        let text_data = gemini_part.data.as_text().unwrap();
        assert_eq!(text_data.to_string(), "Test text");
    }

    #[test]
    fn test_image_part_conversion() {
        let part = Part::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        };

        let gemini_part: GeminiPart = part.into();
        let blob_data = gemini_part.data.as_inline_data().unwrap();
        assert_eq!(blob_data.mime_type, "image/png");
        assert_eq!(blob_data.data, "base64data");
    }

    #[test]
    fn test_file_part_conversion() {
        let file_data = FileData::new_with_display_name(
            "gs://bucket/file.pdf",
            "application/pdf",
            "document.pdf",
        );
        let part = Part::File(file_data);

        let gemini_part: GeminiPart = part.into();
        let file_data_result = gemini_part.data.as_file_data().unwrap();
        assert_eq!(file_data_result.file_uri, "gs://bucket/file.pdf");
        assert_eq!(file_data_result.mime_type, "application/pdf");
        assert_eq!(
            file_data_result.display_name,
            Some("document.pdf".to_string())
        );
    }
}
