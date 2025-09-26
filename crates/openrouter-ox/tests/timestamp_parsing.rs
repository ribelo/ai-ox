use openrouter_ox::response::ChatCompletionChunk;

#[test]
fn chat_completion_chunk_accepts_integer_timestamp() {
    let json = r#"{
        "id": "chunk",
        "provider": "openrouter",
        "model": "grok",
        "object": "chat.completion.chunk",
        "created": 1758887156,
        "choices": [],
        "usage": null
    }"#;

    let chunk: ChatCompletionChunk =
        serde_json::from_str(json).expect("integer timestamp from OpenRouter should deserialize");
    assert_eq!(chunk.created.to_unix_timestamp_i64(), 1_758_887_156);
}
