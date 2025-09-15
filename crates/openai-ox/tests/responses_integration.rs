use openai_ox::{OpenAI, ResponsesInput, ReasoningConfig};
use std::env;

/// Real integration test with OpenAI Responses API using gpt-5-nano (cheapest reasoning model)
/// Run with: `cargo test responses_integration -- --ignored`
#[tokio::test]
#[ignore]
async fn test_real_responses_api_basic() {
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set for integration tests");
    
    let client = OpenAI::new(api_key);
    
    let request = client.responses()
        .model("o4-mini")
        .input(ResponsesInput::text("What is 2 + 2? Please show your reasoning."))
        .reasoning(ReasoningConfig::with_effort("low")) // Use minimal reasoning to keep costs down
        .max_output_tokens(50) // Keep response short and cheap
        .build();
    
    let response = client.send_responses(&request).await
        .expect("Failed to get response from OpenAI Responses API");
    
    // Basic assertions
    assert_eq!(response.model, "o4-mini");
    assert!(response.is_completed());
    assert!(!response.output.is_empty());
    
    // Should have some content
    let content = response.content().expect("Response should have content");
    assert!(!content.is_empty());
    
    // For a reasoning model, we should get some reasoning output
    let reasoning_items = response.reasoning_items();
    if !reasoning_items.is_empty() {
        println!("Reasoning items found: {}", reasoning_items.len());
        for (i, (id, content)) in reasoning_items.iter().enumerate() {
            println!("Reasoning {}: id={}, content={:?}", i, id, content);
        }
    }
    
    // Print the full response for debugging
    println!("Response ID: {}", response.id);
    println!("Content: {}", content);
    if let Some(usage) = &response.usage {
        println!("Token usage - Input: {}, Output: {}, Total: {}", 
                 usage.input_tokens, usage.output_tokens, usage.total_tokens);
        if let Some(reasoning_tokens) = usage.reasoning_tokens {
            println!("Reasoning tokens: {}", reasoning_tokens);
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_real_responses_api_with_encrypted_reasoning() {
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set for integration tests");
    
    let client = OpenAI::new(api_key);
    
    let request = client.responses()
        .model("o4-mini")
        .input(ResponsesInput::text("Calculate 15 * 7. Show your work."))
        .reasoning(ReasoningConfig::with_effort("minimal")) // Cheapest option
        .include(vec!["reasoning.encrypted_content".to_string()]) // Request encrypted reasoning content
        .max_output_tokens(30) // Very short to minimize cost
        .build();
    
    let response = client.send_responses(&request).await
        .expect("Failed to get response from OpenAI Responses API");
    
    assert_eq!(response.model, "o4-mini");
    assert!(response.is_completed());
    
    // Check if we got encrypted reasoning
    let reasoning_items = response.reasoning_items();
    let has_encrypted = reasoning_items.iter().any(|(_, content)| !content.is_empty());
    if has_encrypted {
        println!("✓ Encrypted reasoning content received");
        for reasoning in reasoning_items {
            if !reasoning.1.is_empty() {
                println!("✓ Found encrypted content for reasoning item: {}", reasoning.0);
            }
        }
    } else {
        println!("ℹ No encrypted reasoning content in response");
    }
    
    let content = response.content().expect("Response should have content");
    println!("Response: {}", content);
}

#[tokio::test]
#[ignore]
async fn test_real_responses_api_streaming() {
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set for integration tests");
    
    let client = OpenAI::new(api_key);
    
    let request = client.responses()
        .model("o4-mini")
        .input(ResponsesInput::text("Count from 1 to 5"))
        .reasoning(ReasoningConfig::with_effort("minimal"))
        .max_output_tokens(25)
        .build();
    
    let mut stream = client.stream_responses(&request);
    
    use futures_util::StreamExt;
    let mut chunks_received = 0;
    let mut final_usage = None;
    
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                chunks_received += 1;
                println!("Chunk {}: status={}", chunks_received, chunk.status);
                
                // Print any output in this chunk
                for output in &chunk.output {
                    match output {
                        openai_ox::OutputDelta::MessageDelta(msg_delta) => {
                            if let Some(content) = &msg_delta.content {
                                print!("{}", content);
                            }
                        }
                        openai_ox::OutputDelta::TextDelta(text_delta) => {
                            print!("{}", text_delta.text);
                        }
                        openai_ox::OutputDelta::ReasoningDelta(reasoning_delta) => {
                            if let Some(summary) = &reasoning_delta.summary {
                                println!("\nReasoning: {}", summary);
                            }
                        }
                        _ => {}
                    }
                }
                
                // Capture final usage stats
                if let Some(usage) = chunk.usage {
                    final_usage = Some(usage);
                }
                
                // Break if completed to avoid infinite loop
                if chunk.status == "completed" {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Streaming error: {}", e);
                break;
            }
        }
    }
    
    println!("\n\nStreaming completed!");
    println!("Total chunks received: {}", chunks_received);
    
    if let Some(usage) = final_usage {
        println!("Final usage - Input: {}, Output: {}, Total: {}", 
                 usage.input_tokens, usage.output_tokens, usage.total_tokens);
        if let Some(reasoning_tokens) = usage.reasoning_tokens {
            println!("Reasoning tokens: {}", reasoning_tokens);
        }
    }
    
    assert!(chunks_received > 0, "Should have received at least one chunk");
}

#[tokio::test]
#[ignore]
async fn test_real_responses_api_with_messages() {
    use openai_ox::Message;
    
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set for integration tests");
    
    let client = OpenAI::new(api_key);
    
    let messages = vec![
        Message::user("I need help with a math problem"),
        Message::assistant("I'd be happy to help! What's the math problem?"),
        Message::user("What is 8 + 7?"),
    ];
    
    let request = client.responses()
        .model("o4-mini")
        .input(ResponsesInput::messages(messages))
        .reasoning(ReasoningConfig::with_effort("minimal"))
        .max_output_tokens(20)
        .build();
    
    let response = client.send_responses(&request).await
        .expect("Failed to get response from OpenAI Responses API");
    
    assert_eq!(response.model, "o4-mini");
    assert!(response.is_completed());
    
    let content = response.content().expect("Response should have content");
    println!("Conversational response: {}", content);
    
    // Should contain the answer
    assert!(content.to_lowercase().contains("15") || content.contains("eight") || content.contains("seven"));
}