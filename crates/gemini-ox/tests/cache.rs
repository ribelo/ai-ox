use gemini_ox::{Gemini, content::{Content, Role}};
use std::time::Duration;

fn get_api_key() -> String {
    std::env::var("GOOGLE_AI_API_KEY").expect("GOOGLE_AI_API_KEY must be set")
}

// #[tokio::test]
// #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
// async fn test_cache_lifecycle() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//     let gemini = Gemini::new(get_api_key());

//     // 1. Create a cache
//     let create_request = gemini.caches().create()
//         .model("gemini-1.5-flash-latest".to_string())
//         .display_name("Test Cache".to_string())
//         .ttl(Duration::from_secs(600))
//         .contents(vec![Content::new(Role::User, vec!["Why is the sky blue?"])])
//         .build();

//     let created_cache = create_request.send().await?;
//     assert_eq!(created_cache.display_name.as_deref(), Some("Test Cache"));
//     let cache_name = created_cache.name.clone();

//     println!("Created cache: {}", cache_name);

//     // 2. Get the cache
//     let retrieved_cache = gemini.caches().get(&cache_name).await?;
//     assert_eq!(retrieved_cache.name, cache_name);
//     assert_eq!(retrieved_cache.display_name.as_deref(), Some("Test Cache"));

//     println!("Retrieved cache: {}", retrieved_cache.name);

//     // 3. Use the cache in a generate_content request
//     let generate_request = gemini.generate_content()
//         .model("gemini-1.5-flash-latest")
//         .cached_content(cache_name.clone())
//         .content("Please provide a short answer.")
//         .build();

//     let response = generate_request.send().await?;
//     assert!(!response.candidates.is_empty());

//     println!("Used cache in generation request. Response: {:?}", response);

//     // 4. List the caches
//     let list_response = gemini.caches().list(Some(10), None).await?;
//     assert!(list_response.cached_contents.iter().any(|c| c.name == cache_name));

//     println!("Listed caches. Found our cache.");

//     // 5. Update the cache
//     let update_request = gemini.caches().update(&cache_name)
//         .ttl(Duration::from_secs(1200))
//         .build();
//     let updated_cache = update_request.send().await?;
//     assert_eq!(updated_cache.name, cache_name);
//     // Note: We can't easily assert the expire_time change without knowing the exact server time.
//     // We just assert that the request was successful.

//     println!("Updated cache: {}", updated_cache.name);

//     // 6. Delete the cache
//     gemini.caches().delete(&cache_name).await?;
//     println!("Deleted cache: {}", cache_name);

//     // 7. Verify deletion by trying to get it again
//     let get_after_delete_result = gemini.caches().get(&cache_name).await;
//     assert!(matches!(get_after_delete_result, Err(gemini_ox::GeminiRequestError::InvalidRequestError { .. })));
//     println!("Verified cache deletion.");

//     Ok(())
// }
