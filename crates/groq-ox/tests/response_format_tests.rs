use ai_ox_common::{openai_format::Message, response_format::ResponseFormat};
use groq_ox::ChatRequest;
use serde_json::json;

#[test]
fn json_object_response_format_serializes_as_object() {
    let request =
        ChatRequest::with_json_response("groq-test-model", vec![Message::user("hello world")]);

    let serialized = serde_json::to_value(&request).expect("request should serialize");

    assert_eq!(
        serialized
            .get("response_format")
            .expect("response_format field should exist"),
        &json!({"type": "json_object"})
    );
}

#[test]
fn json_schema_response_format_enforces_type_constant() {
    let schema_payload = json!({
        "schema": {
            "properties": {
                "answer": {"type": "string"}
            }
        }
    });

    let response_format = ResponseFormat::json_schema(schema_payload.clone());

    let serialized = serde_json::to_value(&response_format).expect("format should serialize");

    assert_eq!(
        serialized,
        json!({
            "type": "json_schema",
            "json_schema": schema_payload
        })
    );
}
