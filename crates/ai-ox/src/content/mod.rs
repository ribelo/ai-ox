pub mod delta;
pub mod message;
pub mod part;

// Re-export commonly used types
pub use message::{Message, MessageRole};
pub use part::{DataRef, Part};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use message::{Message, MessageRole};
    use part::Part;
    use serde_json::json;

    #[test]
    fn test_message_timestamp_serialization() {
        let timestamp = Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap();
        let message = Message {
            role: MessageRole::User,
            content: vec![Part::Text {
                text: "Hello".to_string(),
                ext: std::collections::BTreeMap::new(),
            }],
            timestamp: Some(timestamp),
            ext: None,
        };

        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["timestamp"], "2024-01-15T10:30:00Z");
    }

    #[test]
    fn test_message_timestamp_deserialization() {
        let json = json!({
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "timestamp": "2024-01-15T10:30:00Z"
        });

        let message: Message = serde_json::from_value(json).unwrap();
        assert_eq!(
            message.timestamp,
            Some(Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap())
        );
    }
}
