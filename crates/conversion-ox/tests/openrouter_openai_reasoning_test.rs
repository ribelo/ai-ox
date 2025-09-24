#![cfg(feature = "anthropic-openrouter")]

use anthropic_ox::{
    message::{
        Content as AnthropicContent, Message as AnthropicMessage, Role as AnthropicRole,
        StringOrContents, ThinkingContent,
    },
    request::ChatRequest as AnthropicRequest,
};

use openrouter_ox::{
    message::{AssistantMessage, ContentPart, Message as OpenRouterMessage},
    response::{
        ChatCompletionResponse as OpenRouterResponse, Choice,
        FinishReason as OpenRouterFinishReason, ReasoningDetail, Usage as OpenRouterUsage,
    },
};

use conversion_ox::anthropic_openrouter::{
    anthropic_to_openrouter_request, openrouter_to_anthropic_response,
};

use ai_ox_common::timestamp::Timestamp;

#[test]
fn test_openrouter_openai_reasoning_conversion() {
    // Real OpenRouter → OpenAI GPT-5-mini response format with reasoning
    let reasoning_summary = ReasoningDetail {
        detail_type: "reasoning.summary".to_string(),
        text: Some("**Explaining the riddle**\n\nI need to tackle this classic riddle step by step. When it says \"all but 9 die,\" it means 9 survive, so the answer is that 9 sheep are left. I can start with 17 sheep, subtracting those that died, so alive = 9. If I have 17 sheep and all but 9 die, then alive = 9 and dead = 8. It's important to highlight a common trap: some might mistakenly think the answer is 8 by misinterpreting it. I'll clarify this in my response.**Clarifying the final steps**\n\nAlright, let's put this all together. I start with a total of 17 sheep, and since 9 survive, I find that the number dead is 17 minus 9, which equals 8. So, the clear answer is that there are 9 sheep left. I want to ensure my final output reflects this process clearly and concisely for the user, so it's easy to understand. Let's craft that final output!".to_string()),
        summary: None,
        data: None,
        id: None,
        format: Some("openai-responses-v1".to_string()),
        index: Some(0),
    };

    let reasoning_encrypted = ReasoningDetail {
        detail_type: "reasoning.encrypted".to_string(),
        text: Some("gAAAAABosvmR6htThzBEsGCedZT14SwuCMOx...".to_string()), // Truncated for test
        summary: None,
        data: None,
        id: None,
        format: Some("openai-responses-v1".to_string()),
        index: Some(0),
    };

    let choice = Choice {
        index: 0,
        message: AssistantMessage::text(
            "Step 1: Interpret the phrase \"all but 9 die\" — it means every sheep except 9 die.\n\nStep 2: The farmer started with 17 sheep. The number that remain alive is the 9 that did not die.\n\nYou can also compute the number that died: 17 − 9 = 8.\n\nAnswer: 9 sheep are left.",
        ),
        logprobs: None,
        finish_reason: OpenRouterFinishReason::Stop,
        native_finish_reason: Some(OpenRouterFinishReason::Stop),
        reasoning: Some(
            "**Explaining the riddle**\n\nI need to tackle this classic riddle step by step..."
                .to_string(),
        ),
        reasoning_details: Some(vec![reasoning_summary, reasoning_encrypted]),
    };

    let openrouter_response = OpenRouterResponse {
        id: "gen-1756559753-7bsZW2jzW4PtoIYjLa3W".to_string(),
        object: "chat.completion".to_string(),
        created: Timestamp::from_unix_timestamp(1_756_559_753),
        model: "openai/gpt-5-mini".to_string(),
        choices: vec![choice],
        system_fingerprint: None,
        usage: OpenRouterUsage::with_prompt_completion(32, 338),
    };

    // Convert OpenRouter→OpenAI response to Anthropic format
    let anthropic_response = openrouter_to_anthropic_response(openrouter_response).unwrap();

    // Verify thinking content was created from reasoning_details
    assert_eq!(anthropic_response.content.len(), 2);

    // First content should be thinking (from reasoning_details[0].text)
    if let AnthropicContent::Thinking(thinking) = &anthropic_response.content[0] {
        assert!(thinking.text.contains("**Explaining the riddle**"));
        assert!(
            thinking
                .text
                .contains("I need to tackle this classic riddle step by step")
        );
        assert!(thinking.text.contains("Let's craft that final output"));
    } else {
        panic!(
            "Expected thinking content from OpenRouter→OpenAI reasoning, got: {:?}",
            anthropic_response.content[0]
        );
    }

    // Second content should be regular text (the final answer)
    if let AnthropicContent::Text(text) = &anthropic_response.content[1] {
        assert!(text.text.contains("Step 1: Interpret the phrase"));
        assert!(text.text.contains("Answer: 9 sheep are left"));
    } else {
        panic!(
            "Expected text content, got: {:?}",
            anthropic_response.content[1]
        );
    }
}

#[test]
fn test_anthropic_thinking_to_openrouter_openai_request() {
    // Test that Anthropic thinking content enables reasoning for OpenAI models via OpenRouter
    let thinking_content = ThinkingContent::new("I need to solve this sheep riddle carefully. Let me think about what 'all but 9 die' means.".to_string());

    let anthropic_request = AnthropicRequest::builder()
        .model("openai/gpt-5-mini")
        .messages(vec![
            AnthropicMessage {
                role: AnthropicRole::User,
                content: StringOrContents::Contents(vec![AnthropicContent::Text(
                    anthropic_ox::message::Text::new(
                        "A farmer has 17 sheep. All but 9 die. How many sheep are left?"
                            .to_string(),
                    ),
                )]),
            },
            AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: StringOrContents::Contents(vec![
                    AnthropicContent::Thinking(thinking_content),
                    AnthropicContent::Text(anthropic_ox::message::Text::new(
                        "Let me work through this step by step...".to_string(),
                    )),
                ]),
            },
        ])
        .build();

    // Convert to OpenRouter request
    let openrouter_request = anthropic_to_openrouter_request(anthropic_request).unwrap();

    // Verify reasoning is enabled for OpenAI models via OpenRouter
    assert!(openrouter_request.reasoning.is_some());
    assert_eq!(
        openrouter_request.reasoning.as_ref().unwrap().enabled,
        Some(true)
    );
    assert_eq!(openrouter_request.model, "openai/gpt-5-mini");

    // Verify thinking content was converted to text in OpenRouter format
    assert_eq!(openrouter_request.messages.len(), 2);
    if let OpenRouterMessage::Assistant(assistant_msg) = &openrouter_request.messages[1] {
        assert_eq!(assistant_msg.content.len(), 2);

        // First part should be the converted thinking text
        if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
            assert!(
                text_content
                    .text
                    .contains("I need to solve this sheep riddle carefully")
            );
            assert!(text_content.text.contains("all but 9 die"));
        } else {
            panic!("Expected thinking text content part");
        }

        // Second part should be the regular response
        if let ContentPart::Text(text_content) = &assistant_msg.content[1] {
            assert_eq!(
                text_content.text,
                "Let me work through this step by step..."
            );
        } else {
            panic!("Expected regular text content part");
        }
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_openrouter_openai_encrypted_reasoning_handling() {
    // Test that we can handle OpenRouter's encrypted reasoning format
    let reasoning_details = vec![
        ReasoningDetail {
            detail_type: "reasoning.summary".to_string(),
            text: Some("This is the readable reasoning summary".to_string()),
            summary: None,
            data: None,
            id: None,
            format: Some("openai-responses-v1".to_string()),
            index: Some(0),
        },
        ReasoningDetail {
            detail_type: "reasoning.encrypted".to_string(),
            text: Some("encrypted_reasoning_data_here".to_string()), // This would be encrypted in real response
            summary: None,
            data: None,
            id: None,
            format: Some("openai-responses-v1".to_string()),
            index: Some(0),
        },
    ];

    let choice = Choice {
        index: 0,
        message: AssistantMessage::text("The answer is 42."),
        logprobs: None,
        finish_reason: OpenRouterFinishReason::Stop,
        native_finish_reason: None,
        reasoning: Some("Summary of reasoning".to_string()),
        reasoning_details: Some(reasoning_details),
    };

    let openrouter_response = OpenRouterResponse {
        id: "test-openai-encrypted".to_string(),
        object: "chat.completion".to_string(),
        created: Timestamp::from_unix_timestamp(1_756_559_753),
        model: "openai/gpt-5-mini".to_string(),
        choices: vec![choice],
        system_fingerprint: None,
        usage: OpenRouterUsage::with_prompt_completion(10, 50),
    };

    let anthropic_response = openrouter_to_anthropic_response(openrouter_response).unwrap();

    // Should prioritize the summary over encrypted data
    if let AnthropicContent::Thinking(thinking) = &anthropic_response.content[0] {
        // Should use the first reasoning_detail (summary) for thinking content
        assert_eq!(thinking.text, "This is the readable reasoning summary");
    } else {
        panic!("Expected thinking content with summary text");
    }

    if let AnthropicContent::Text(text) = &anthropic_response.content[1] {
        assert_eq!(text.text, "The answer is 42.");
    } else {
        panic!("Expected regular text content");
    }
}
