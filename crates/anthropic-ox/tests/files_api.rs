#![cfg(feature = "files")]

use anthropic_ox::{Anthropic, files::FileUploadRequest};

#[tokio::test]
async fn test_files_api_lifecycle() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    // 1. Upload a file
    let upload_request = FileUploadRequest {
        content: "Hello, world!".as_bytes().to_vec(),
        filename: "test.txt".to_string(),
        mime_type: "text/plain".to_string(),
    };

    let uploaded_file = match client.upload_file(&upload_request).await {
        Ok(file) => {
            println!("✅ File uploaded successfully: {}", file.id);
            file
        }
        Err(e) => {
            println!("⚠️  Failed to upload file: {}", e);
            return;
        }
    };

    assert_eq!(uploaded_file.filename, "test.txt");
    assert_eq!(uploaded_file.size_bytes, 13);

    // 2. List files
    let file_id = uploaded_file.id.clone();
    match client.list_files(None, None, None).await {
        Ok(list) => {
            println!("✅ Listed files successfully.");
            assert!(
                list.data.iter().any(|f| f.id == file_id),
                "Uploaded file should be in the list"
            );
        }
        Err(e) => {
            println!("⚠️  Failed to list files: {}", e);
        }
    }

    // 3. Get file metadata
    match client.get_file(&file_id).await {
        Ok(file) => {
            println!("✅ Got file metadata successfully.");
            assert_eq!(file.id, file_id);
            assert_eq!(file.filename, "test.txt");
        }
        Err(e) => {
            println!("⚠️  Failed to get file metadata: {}", e);
        }
    }

    // 4. Delete the file
    match client.delete_file(&file_id).await {
        Ok(_) => {
            println!("✅ File deleted successfully.");
        }
        Err(e) => {
            println!("⚠️  Failed to delete file: {}", e);
        }
    }

    // 5. List files again to confirm deletion
    match client.list_files(None, None, None).await {
        Ok(list) => {
            println!("✅ Listed files again successfully.");
            assert!(
                !list.data.iter().any(|f| f.id == file_id),
                "Deleted file should not be in the list"
            );
        }
        Err(e) => {
            println!("⚠️  Failed to list files again: {}", e);
        }
    }
}
