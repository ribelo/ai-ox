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
        .build();

    let json = serde_json::to_string_pretty(&request).unwrap();

    let expected_json = serde_json::json!({
      "messages": [],
      "model": "anthropic/claude-3-opus",
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
