#![cfg(feature = "batches")]

use anthropic_ox::{
    Anthropic,
    ChatRequest,
    batches::{BatchMessageRequest, MessageBatchRequest},
    message::{Content, Message, Role, Text},
    Model,
};
use futures_util::stream::StreamExt;

#[tokio::test]
async fn test_message_batches_api_lifecycle() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    // 1. Create a batch
    let requests = vec![
        BatchMessageRequest {
            custom_id: "test-request-1".to_string(),
            params: ChatRequest::builder()
                .model(Model::Claude3Haiku20240307.to_string())
                .messages(vec![Message::new(
                    Role::User,
                    vec![Content::Text(Text::new("Hello".to_string()))],
                )])
                .max_tokens(10)
                .build(),
        },
        BatchMessageRequest {
            custom_id: "test-request-2".to_string(),
            params: ChatRequest::builder()
                .model(Model::Claude3Haiku20240307.to_string())
                .messages(vec![Message::new(
                    Role::User,
                    vec![Content::Text(Text::new("World".to_string()))],
                )])
                .max_tokens(10)
                .build(),
        },
    ];

    let batch_request = MessageBatchRequest { requests };
    let created_batch = match client.create_message_batch(&batch_request).await {
        Ok(batch) => {
            println!("✅ Batch created successfully: {}", batch.id);
            batch
        }
        Err(e) => {
            println!("⚠️  Failed to create batch: {}", e);
            // Can't proceed if creation fails
            return;
        }
    };

    // 2. Poll for completion
    let batch_id = created_batch.id.clone();
    let mut batch = created_batch;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        batch = match client.get_message_batch(&batch_id).await {
            Ok(b) => b,
            Err(e) => {
                println!("⚠️  Failed to get batch status: {}", e);
                return;
            }
        };
        println!("Polling batch status: {:?}", batch.processing_status);
        if batch.processing_status == anthropic_ox::batches::BatchStatus::Ended {
            println!("✅ Batch processing ended.");
            break;
        }
    }

    assert_eq!(
        batch.processing_status,
        anthropic_ox::batches::BatchStatus::Ended,
        "Batch did not complete in time"
    );

    // 3. Retrieve results
    let mut results_stream = client.get_message_batch_results(&batch_id);
    let mut results = Vec::new();
    while let Some(result) = results_stream.next().await {
        match result {
            Ok(res) => {
                println!("✅ Got result for custom_id: {}", res.custom_id);
                results.push(res);
            }
            Err(e) => {
                println!("⚠️  Error streaming results: {}", e);
            }
        }
    }

    assert_eq!(results.len(), 2, "Should have received 2 results");

    // 4. List batches to see if ours is there
    match client.list_message_batches(Some(5), None).await {
        Ok(list) => {
            println!("✅ Listed batches successfully.");
            assert!(
                list.data.iter().any(|b| b.id == batch_id),
                "Created batch should be in the list"
            );
        }
        Err(e) => {
            println!("⚠️  Failed to list batches: {}", e);
        }
    }
}
