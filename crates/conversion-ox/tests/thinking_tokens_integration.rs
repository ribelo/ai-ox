#![cfg(feature = "anthropic-gemini")]

use anthropic_ox::{
    message::{
        Content as AnthropicContent, Message, Messages, Role as AnthropicRole, Text,
        ThinkingContent,
    },
    request::ChatRequest as AnthropicRequest,
};
use conversion_ox::anthropic_gemini::{anthropic_to_gemini_request, gemini_to_anthropic_response};
use gemini_ox::{
    content::{
        Content as GeminiContent, Part as GeminiPart, PartData, Role as GeminiRole,
        Text as GeminiText,
    },
    generate_content::{
        FinishReason as GeminiFinishReason, ResponseCandidate, response::GenerateContentResponse,
    },
};

#[test]
fn test_real_gemini_thinking_response_conversion() {
    // Create a simplified Gemini response based on real API data
    let real_gemini_response = GeminiContent {
        role: GeminiRole::Model,
        parts: vec![
            // First part: thinking content (thought: true)
            GeminiPart {
                thought: Some(true),
                thought_signature: Some("sig_abc123".to_string()),
                video_metadata: None,
                data: PartData::Text(GeminiText::from(
                    "**My Approach to Solving the Math Problem**\n\nOkay, so I'm looking at this problem: \"What is 15 * 23 + 7?\" My first thought is, this is pretty straightforward, but I need to make sure I'm clear and methodical in my explanation.",
                )),
            },
            // Second part: regular response content (no thought field)
            GeminiPart {
                thought: None,
                thought_signature: None,
                video_metadata: None,
                data: PartData::Text(GeminiText::from(
                    "To solve the expression $15 * 23 + 7$, we need to follow the order of operations. First multiply: $15 * 23 = 345$. Then add: $345 + 7 = 352$.",
                )),
            },
        ],
    };

    let gemini_response = GenerateContentResponse {
        candidates: vec![ResponseCandidate {
            content: real_gemini_response,
            finish_reason: Some(GeminiFinishReason::Stop),
            index: Some(0),
            safety_ratings: Vec::new(),
            citation_metadata: None,
            token_count: None,
            grounding_attributions: Some(Vec::new()),
            avg_logprobs: None,
            logprobs_result: None,
            grounding_metadata: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: Some("gemini-2.5-flash".to_string()),
    };

    // Convert Gemini response to Anthropic response
    let anthropic_response = gemini_to_anthropic_response(gemini_response).unwrap();

    // Verify the conversion
    assert_eq!(anthropic_response.content.len(), 2);

    // First content should be thinking content
    match &anthropic_response.content[0] {
        AnthropicContent::Thinking(thinking) => {
            assert!(
                thinking
                    .text
                    .contains("My Approach to Solving the Math Problem")
            );
            assert!(thinking.text.contains("15 * 23 + 7"));
            assert_eq!(thinking.signature.as_ref().unwrap(), "sig_abc123");
        }
        _ => panic!("First content should be thinking content"),
    }

    // Second content should be regular text content
    match &anthropic_response.content[1] {
        AnthropicContent::Text(text) => {
            assert!(text.text.contains("To solve the expression $15 * 23 + 7$"));
            assert!(text.text.contains("345 + 7 = 352"));
        }
        _ => panic!("Second content should be text content"),
    }

    // Verify stop reason conversion
    assert_eq!(
        anthropic_response.stop_reason,
        Some(anthropic_ox::response::StopReason::EndTurn)
    );
}

#[test]
fn test_anthropic_to_gemini_thinking_conversion() {
    // Create Anthropic request with thinking content
    let anthropic_request = AnthropicRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: Messages(vec![
            Message {
                role: AnthropicRole::User,
                content: anthropic_ox::message::StringOrContents::String(
                    "What is 7 * 8?".to_string(),
                ),
            },
            Message {
                role: AnthropicRole::Assistant,
                content: anthropic_ox::message::StringOrContents::Contents(vec![
                    AnthropicContent::Thinking(ThinkingContent {
                        text: "Let me think step by step. 7 * 8 means adding 7 eight times."
                            .to_string(),
                        signature: Some("thinking_sig_456".to_string()),
                    }),
                    AnthropicContent::Text(Text::new("7 * 8 = 56".to_string())),
                ]),
            },
        ]),
        max_tokens: 1000,
        metadata: None,
        stop_sequences: None,
        stream: None,
        temperature: None,
        top_p: None,
        top_k: None,
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    // Convert to Gemini request
    let gemini_request = anthropic_to_gemini_request(anthropic_request);

    // Verify the conversion
    assert_eq!(gemini_request.model, "gemini-2.5-flash");
    assert_eq!(gemini_request.contents.len(), 2);

    // First content should be user message
    let user_content = &gemini_request.contents[0];
    assert_eq!(user_content.role, GeminiRole::User);
    assert_eq!(user_content.parts.len(), 1);

    // Second content should be assistant message with thinking part
    let assistant_content = &gemini_request.contents[1];
    assert_eq!(assistant_content.role, GeminiRole::Model);
    assert_eq!(assistant_content.parts.len(), 2);

    // First part should be thinking content
    let thinking_part = &assistant_content.parts[0];
    assert_eq!(thinking_part.thought, Some(true));
    assert_eq!(
        thinking_part.thought_signature.as_ref().unwrap(),
        "thinking_sig_456"
    );
    if let PartData::Text(text) = &thinking_part.data {
        assert!(text.to_string().contains("step by step"));
        assert!(text.to_string().contains("7 * 8"));
    } else {
        panic!("Thinking part should contain text data");
    }

    // Second part should be regular text content
    let text_part = &assistant_content.parts[1];
    assert_eq!(text_part.thought, None);
    assert_eq!(text_part.thought_signature, None);
    if let PartData::Text(text) = &text_part.data {
        assert_eq!(text.to_string(), "7 * 8 = 56");
    } else {
        panic!("Text part should contain text data");
    }

    // Verify thinking config is enabled
    let generation_config = gemini_request
        .generation_config
        .expect("Generation config should be set");
    let thinking_config = generation_config
        .thinking_config
        .expect("Thinking config should be set");
    assert_eq!(thinking_config.include_thoughts, true);
    assert_eq!(thinking_config.thinking_budget, -1); // Dynamic budget
}

#[test]
fn test_gemini_to_anthropic_to_gemini_round_trip() {
    // Start with original thinking and answer text
    let original_thinking_text =
        "I need to solve this carefully. Let me break it down step by step.";
    let original_answer_text = "The final answer is 42.";
    let original_signature = "round_trip_sig_789";

    // Original Gemini response with thinking
    let original_gemini_response = GenerateContentResponse {
        candidates: vec![ResponseCandidate {
            content: GeminiContent {
                role: GeminiRole::Model,
                parts: vec![
                    // Thinking part with thought=true and signature
                    GeminiPart {
                        thought: Some(true),
                        thought_signature: Some(original_signature.to_string()),
                        video_metadata: None,
                        data: PartData::Text(GeminiText::from(original_thinking_text.clone())),
                    },
                    // Regular answer part
                    GeminiPart {
                        thought: None,
                        thought_signature: None,
                        video_metadata: None,
                        data: PartData::Text(GeminiText::from(original_answer_text.clone())),
                    },
                ],
            },
            finish_reason: Some(GeminiFinishReason::Stop),
            index: Some(0),
            safety_ratings: Vec::new(),
            citation_metadata: None,
            token_count: None,
            grounding_attributions: Some(Vec::new()),
            avg_logprobs: None,
            logprobs_result: None,
            grounding_metadata: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: Some("gemini-2.5-flash".to_string()),
    };

    // Step 1: Convert Gemini -> Anthropic
    let anthropic_response = gemini_to_anthropic_response(original_gemini_response).unwrap();

    // Verify Anthropic conversion
    assert_eq!(anthropic_response.content.len(), 2);

    let thinking_content = match &anthropic_response.content[0] {
        AnthropicContent::Thinking(thinking) => {
            assert_eq!(thinking.text, original_thinking_text);
            assert_eq!(thinking.signature.as_ref().unwrap(), original_signature);
            thinking.clone()
        }
        _ => panic!("First content should be thinking content"),
    };

    let text_content = match &anthropic_response.content[1] {
        AnthropicContent::Text(text) => {
            assert_eq!(text.text, original_answer_text);
            text.clone()
        }
        _ => panic!("Second content should be text content"),
    };

    // Step 2: Convert Anthropic -> Gemini (round trip)
    let anthropic_request = AnthropicRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: Messages(vec![Message {
            role: AnthropicRole::Assistant,
            content: anthropic_ox::message::StringOrContents::Contents(vec![
                AnthropicContent::Thinking(thinking_content),
                AnthropicContent::Text(text_content),
            ]),
        }]),
        max_tokens: 1000,
        metadata: None,
        stop_sequences: None,
        stream: None,
        temperature: None,
        top_p: None,
        top_k: None,
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    let final_gemini_request = anthropic_to_gemini_request(anthropic_request);

    // Step 3: Verify round-trip preservation
    assert_eq!(final_gemini_request.contents.len(), 1); // One assistant message
    let assistant_content = &final_gemini_request.contents[0];
    assert_eq!(assistant_content.role, GeminiRole::Model);
    assert_eq!(assistant_content.parts.len(), 2);

    // Verify thinking part is exactly preserved
    let thinking_part = &assistant_content.parts[0];
    assert_eq!(thinking_part.thought, Some(true));
    assert_eq!(
        thinking_part.thought_signature.as_ref().unwrap(),
        original_signature
    );
    if let PartData::Text(text) = &thinking_part.data {
        assert_eq!(text.to_string(), original_thinking_text);
    } else {
        panic!("Thinking part should contain text data");
    }

    // Verify regular text part is exactly preserved
    let text_part = &assistant_content.parts[1];
    assert_eq!(text_part.thought, None);
    assert_eq!(text_part.thought_signature, None);
    if let PartData::Text(text) = &text_part.data {
        assert_eq!(text.to_string(), original_answer_text);
    } else {
        panic!("Text part should contain text data");
    }

    // Verify thinking config is enabled in round-trip
    let generation_config = final_gemini_request
        .generation_config
        .expect("Generation config should be set");
    let thinking_config = generation_config
        .thinking_config
        .expect("Thinking config should be set");
    assert_eq!(thinking_config.include_thoughts, true);
    assert_eq!(thinking_config.thinking_budget, -1); // Dynamic budget
}

#[test]
fn test_anthropic_to_gemini_to_anthropic_round_trip() {
    // Test the full round trip: Anthropic -> Gemini -> Anthropic
    let original_thinking_text =
        "I need to carefully analyze this mathematical problem step by step.";
    let original_answer_text = "The solution is 84.";
    let original_signature = "anthropic_round_trip_sig";

    // Step 1: Start with Anthropic request containing thinking content
    let original_anthropic_request = AnthropicRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: Messages(vec![
            Message {
                role: AnthropicRole::User,
                content: anthropic_ox::message::StringOrContents::String(
                    "What is 12 * 7?".to_string(),
                ),
            },
            Message {
                role: AnthropicRole::Assistant,
                content: anthropic_ox::message::StringOrContents::Contents(vec![
                    AnthropicContent::Thinking(ThinkingContent {
                        text: original_thinking_text.to_string(),
                        signature: Some(original_signature.to_string()),
                    }),
                    AnthropicContent::Text(Text::new(original_answer_text.to_string())),
                ]),
            },
        ]),
        max_tokens: 1000,
        metadata: None,
        stop_sequences: None,
        stream: None,
        temperature: None,
        top_p: None,
        top_k: None,
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    // Step 2: Convert Anthropic -> Gemini
    let gemini_request = anthropic_to_gemini_request(original_anthropic_request);

    // Verify the Gemini conversion has thinking parts
    assert_eq!(gemini_request.contents.len(), 2);
    let assistant_content = &gemini_request.contents[1];
    assert_eq!(assistant_content.parts.len(), 2);

    let thinking_part = &assistant_content.parts[0];
    assert_eq!(thinking_part.thought, Some(true));
    assert_eq!(
        thinking_part.thought_signature.as_ref().unwrap(),
        original_signature
    );

    // Step 3: Simulate Gemini response using the request content
    let simulated_gemini_response = GenerateContentResponse {
        candidates: vec![ResponseCandidate {
            content: assistant_content.clone(),
            finish_reason: Some(GeminiFinishReason::Stop),
            index: Some(0),
            safety_ratings: Vec::new(),
            citation_metadata: None,
            token_count: None,
            grounding_attributions: Some(Vec::new()),
            avg_logprobs: None,
            logprobs_result: None,
            grounding_metadata: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: Some("gemini-2.5-flash".to_string()),
    };

    // Step 4: Convert Gemini -> Anthropic (complete round trip)
    let final_anthropic_response = gemini_to_anthropic_response(simulated_gemini_response).unwrap();

    // Step 5: Verify round-trip preservation
    assert_eq!(final_anthropic_response.content.len(), 2);

    // Verify thinking content is exactly preserved
    match &final_anthropic_response.content[0] {
        AnthropicContent::Thinking(thinking) => {
            assert_eq!(thinking.text, original_thinking_text);
            assert_eq!(thinking.signature.as_ref().unwrap(), original_signature);
        }
        _ => panic!("First content should be thinking content"),
    }

    // Verify text content is exactly preserved
    match &final_anthropic_response.content[1] {
        AnthropicContent::Text(text) => {
            assert_eq!(text.text, original_answer_text);
        }
        _ => panic!("Second content should be text content"),
    }
}

#[test]
fn test_full_gemini_to_anthropic_to_gemini_round_trip() {
    // Test the full round trip: Gemini -> Anthropic -> Gemini
    let original_thinking_text = "Let me work through this calculation methodically.";
    let original_answer_text = "The result is 144.";
    let original_signature = "gemini_round_trip_sig";

    // Step 1: Start with Gemini response containing thinking
    let original_gemini_response = GenerateContentResponse {
        candidates: vec![ResponseCandidate {
            content: GeminiContent {
                role: GeminiRole::Model,
                parts: vec![
                    GeminiPart {
                        thought: Some(true),
                        thought_signature: Some(original_signature.to_string()),
                        video_metadata: None,
                        data: PartData::Text(GeminiText::from(original_thinking_text)),
                    },
                    GeminiPart {
                        thought: None,
                        thought_signature: None,
                        video_metadata: None,
                        data: PartData::Text(GeminiText::from(original_answer_text)),
                    },
                ],
            },
            finish_reason: Some(GeminiFinishReason::Stop),
            index: Some(0),
            safety_ratings: Vec::new(),
            citation_metadata: None,
            token_count: None,
            grounding_attributions: Some(Vec::new()),
            avg_logprobs: None,
            logprobs_result: None,
            grounding_metadata: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: Some("gemini-2.5-flash".to_string()),
    };

    // Step 2: Convert Gemini -> Anthropic
    let anthropic_response = gemini_to_anthropic_response(original_gemini_response).unwrap();

    // Verify the Anthropic conversion
    assert_eq!(anthropic_response.content.len(), 2);

    // Extract the converted content for round-trip
    let thinking_content = match &anthropic_response.content[0] {
        AnthropicContent::Thinking(thinking) => thinking.clone(),
        _ => panic!("First content should be thinking content"),
    };

    let text_content = match &anthropic_response.content[1] {
        AnthropicContent::Text(text) => text.clone(),
        _ => panic!("Second content should be text content"),
    };

    // Step 3: Convert back to Anthropic request format
    let anthropic_request = AnthropicRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: Messages(vec![Message {
            role: AnthropicRole::Assistant,
            content: anthropic_ox::message::StringOrContents::Contents(vec![
                AnthropicContent::Thinking(thinking_content),
                AnthropicContent::Text(text_content),
            ]),
        }]),
        max_tokens: 1000,
        metadata: None,
        stop_sequences: None,
        stream: None,
        temperature: None,
        top_p: None,
        top_k: None,
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    // Step 4: Convert Anthropic -> Gemini (complete round trip)
    let final_gemini_request = anthropic_to_gemini_request(anthropic_request);

    // Step 5: Verify round-trip preservation
    assert_eq!(final_gemini_request.contents.len(), 1);
    let assistant_content = &final_gemini_request.contents[0];
    assert_eq!(assistant_content.role, GeminiRole::Model);
    assert_eq!(assistant_content.parts.len(), 2);

    // Verify thinking part is exactly preserved
    let thinking_part = &assistant_content.parts[0];
    assert_eq!(thinking_part.thought, Some(true));
    assert_eq!(
        thinking_part.thought_signature.as_ref().unwrap(),
        original_signature
    );
    if let PartData::Text(text) = &thinking_part.data {
        assert_eq!(text.to_string(), original_thinking_text);
    } else {
        panic!("Thinking part should contain text data");
    }

    // Verify text part is exactly preserved
    let text_part = &assistant_content.parts[1];
    assert_eq!(text_part.thought, None);
    assert_eq!(text_part.thought_signature, None);
    if let PartData::Text(text) = &text_part.data {
        assert_eq!(text.to_string(), original_answer_text);
    } else {
        panic!("Text part should contain text data");
    }

    // Verify thinking config is enabled
    let generation_config = final_gemini_request
        .generation_config
        .expect("Generation config should be set");
    let thinking_config = generation_config
        .thinking_config
        .expect("Thinking config should be set");
    assert_eq!(thinking_config.include_thoughts, true);
    assert_eq!(thinking_config.thinking_budget, -1); // Dynamic budget
}
