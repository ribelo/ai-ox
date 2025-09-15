use serde::{Deserialize, Serialize};
use serde_json;

use crate::{
    content::Part,
    errors::GenerateContentError,
};

/// Structured format for encoding/decoding tool result parts
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolResultEncoding {
    ai_ox_tool_result: ToolResultContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolResultContent {
    name: String,
    content: Vec<Part>,
}

/// Encode tool result parts and name into a standardized JSON string format
///
/// This function serializes the parts and tool name into a structured JSON format that can be
/// safely transmitted through providers that don't support complex content types.
///
/// # Arguments
/// * `name` - The tool name
/// * `parts` - The parts to encode
///
/// # Returns
/// * `Ok(String)` - The encoded JSON string
/// * `Err(GenerateContentError)` - If encoding fails
///
/// # Format
/// ```json
/// {
///   "ai_ox_tool_result": {
///     "name": "tool_name",
///     "content": [
///       {"type": "text", "text": "..."},
///       {"type": "image", "source": {"type": "base64", "mediaType": "...", "data": "..."}},
///       ...
///     ]
///   }
/// }
/// ```
pub fn encode_tool_result_parts(name: &str, parts: &[Part]) -> Result<String, GenerateContentError> {
    let encoding = ToolResultEncoding {
        ai_ox_tool_result: ToolResultContent {
            name: name.to_string(),
            content: parts.to_vec(),
        },
    };

    serde_json::to_string(&encoding)
        .map_err(|e| GenerateContentError::message_conversion(
            &format!("Failed to encode tool result parts: {}", e)
        ))
}

/// Decode a standardized JSON string back into tool name and Vec<Part>
///
/// This function deserializes the structured JSON format back into the original tool name and Parts.
///
/// # Arguments
/// * `s` - The encoded JSON string
///
/// # Returns
/// * `Ok((String, Vec<Part>))` - The decoded tool name and parts
/// * `Err(GenerateContentError)` - If decoding fails or the format is invalid
pub fn decode_tool_result_parts(s: &str) -> Result<(String, Vec<Part>), GenerateContentError> {
    let encoding: ToolResultEncoding = serde_json::from_str(s)
        .map_err(|e| GenerateContentError::message_conversion(
            &format!("Failed to decode tool result parts: {}", e)
        ))?;

    Ok((encoding.ai_ox_tool_result.name, encoding.ai_ox_tool_result.content))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{DataRef, Part};
    use std::collections::BTreeMap;

    #[test]
    fn test_encode_decode_text_parts() {
        let name = "test_tool";
        let parts = vec![
            Part::Text { text: "Hello world".to_string(), ext: BTreeMap::new() },
            Part::Text { text: "Second message".to_string(), ext: BTreeMap::new() },
        ];

        let encoded = encode_tool_result_parts(name, &parts).unwrap();
        let (decoded_name, decoded_parts) = decode_tool_result_parts(&encoded).unwrap();

        assert_eq!(name, decoded_name);
        assert_eq!(parts, decoded_parts);
    }

    #[test]
    fn test_encode_decode_mixed_parts() {
        let name = "image_tool";
        let parts = vec![
            Part::Text { text: "Result:".to_string(), ext: BTreeMap::new() },
            Part::Blob {
                data_ref: DataRef::Base64 {
                    data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==".to_string(),
                },
                mime_type: "image/png".to_string(),
                name: None,
                description: None,
                ext: BTreeMap::new(),
            },
        ];

        let encoded = encode_tool_result_parts(name, &parts).unwrap();
        let (decoded_name, decoded_parts) = decode_tool_result_parts(&encoded).unwrap();

        assert_eq!(name, decoded_name);
        assert_eq!(parts, decoded_parts);
    }

    #[test]
    fn test_encode_empty_parts() {
        let name = "empty_tool";
        let parts = vec![];

        let encoded = encode_tool_result_parts(name, &parts).unwrap();
        let (decoded_name, decoded_parts) = decode_tool_result_parts(&encoded).unwrap();

        assert_eq!(name, decoded_name);
        assert_eq!(parts, decoded_parts);
    }

    #[test]
    fn test_decode_invalid_json() {
        let result = decode_tool_result_parts("invalid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_missing_structure() {
        let result = decode_tool_result_parts(r#"{"invalid": "structure"}"#);
        assert!(result.is_err());
    }
}