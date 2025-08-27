use openrouter_ox::{
    provider_preference::{MaxPrice, Provider, ProviderPreferences, Quantization, Sort},
    request::ChatRequest,
};

#[test]
fn test_provider_preferences_serialization() {
    let preferences = ProviderPreferences {
        allow_fallbacks: Some(false),
        require_parameters: Some(true),
        data_collection: None,
        order: Some(vec![Provider::OpenAI, Provider::Google]),
        only: Some(vec![Provider::Anthropic]),
        ignore: Some(vec![Provider::Groq]),
        quantizations: Some(vec![Quantization::Fp16, Quantization::Int8]),
        sort: Some(Sort::Price),
        max_price: Some(MaxPrice {
            prompt: Some(0.5),
            completion: Some(1.5),
            image: None,
            audio: None,
            request: None,
        }),
    };

    let request = ChatRequest::builder()
        .model("anthropic/claude-3-opus")
        .messages(Vec::<openrouter_ox::message::Message>::new())
        .provider(preferences)
        .preset("my-preset".to_string())
        .models(vec!["model1".to_string(), "model2".to_string()])
        .build();

    let json = serde_json::to_string_pretty(&request).unwrap();

    let expected_json = serde_json::json!({
      "messages": [],
      "model": "anthropic/claude-3-opus",
      "preset": "my-preset",
      "models": ["model1", "model2"],
      "provider": {
        "allow_fallbacks": false,
        "require_parameters": true,
        "order": [
          "openAI",
          "google"
        ],
        "only": [
          "anthropic"
        ],
        "ignore": [
          "groq"
        ],
        "quantizations": [
          "fp16",
          "int8"
        ],
        "sort": "price",
        "max_price": {
          "prompt": 0.5,
          "completion": 1.5
        }
      }
    });

    let expected_str = serde_json::to_string_pretty(&expected_json).unwrap();

    let actual_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let expected_value: serde_json::Value = serde_json::from_str(&expected_str).unwrap();

    assert_eq!(actual_value, expected_value);
}

#[test]
fn test_structured_output_serialization() {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct MyStruct {
        foo: String,
    }

    let request = ChatRequest::builder()
        .model("anthropic/claude-3-opus")
        .messages(Vec::<openrouter_ox::message::Message>::new())
        .response_format::<MyStruct>()
        .build();

    let json = serde_json::to_value(&request).unwrap();
    let strict = json["response_format"]["json_schema"]["strict"].as_bool().unwrap();

    assert!(strict);
}

#[test]
fn test_prompt_caching_serialization() {
    let message = openrouter_ox::message::UserMessage::new(vec![
        openrouter_ox::message::ContentPart::from("Given the book below:"),
        openrouter_ox::message::ContentPart::Text(
            openrouter_ox::message::TextContent::builder()
                .text("HUGE TEXT BODY".to_string())
                .cache_control(openrouter_ox::message::CacheControl {
                    r#type: "ephemeral".to_string(),
                })
                .build(),
        ),
        openrouter_ox::message::ContentPart::from("Name all the characters in the above book"),
    ]);

    let request = ChatRequest::builder()
        .model("anthropic/claude-3-opus")
        .messages(vec![openrouter_ox::message::Message::from(message)])
        .build();

    let json = serde_json::to_string_pretty(&request).unwrap();

    let expected_json = serde_json::json!({
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Given the book below:"
            },
            {
              "type": "text",
              "text": "HUGE TEXT BODY",
              "cache_control": {
                "type": "ephemeral"
              }
            },
            {
              "type": "text",
              "text": "Name all the characters in the above book"
            }
          ]
        }
      ],
      "model": "anthropic/claude-3-opus"
    });

    let expected_str = serde_json::to_string_pretty(&expected_json).unwrap();

    let actual_value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let expected_value: serde_json::Value = serde_json::from_str(&expected_str).unwrap();

    assert_eq!(actual_value, expected_value);
}
