use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use tokio::fs::File;

use crate::{Gemini, GeminiRequestError};

const STANDARD_BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, Deserialize, Serialize)]
pub struct FileInfo {
    pub file: FileData,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FileData {
    pub uri: String,
    #[serde(rename = "displayName")]
    pub display_name: String,
}

#[derive(Debug, Clone, Builder)]
pub struct FileUploadRequest {
    #[builder(into)]
    file_name: String,
    #[builder(into)]
    mime_type: String,
    #[builder(into)]
    data: Vec<u8>,
    gemini: Gemini,
}

#[derive(Debug, Clone, Builder)]
pub struct FileStreamUploadRequest {
    #[builder(into)]
    file_path: PathBuf,
    #[builder(into)]
    mime_type: String,
    gemini: Gemini,
}

impl FileUploadRequest {
    pub async fn send(self) -> Result<String, GeminiRequestError> {
        let num_bytes = self.data.len();

        let api_key = self
            .gemini
            .api_key
            .as_ref()
            .ok_or(GeminiRequestError::AuthenticationMissing)?;

        let init_url = format!(
            "{}/upload/{}/files",
            STANDARD_BASE_URL,
            self.gemini.api_version
        );

        let init_response = self
            .gemini
            .client
            .post(&init_url)
            .query(&[("key", api_key.as_str())])
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Header-Content-Type", &self.mime_type)
            .json(&json!({
                "file": {
                    "display_name": self.file_name
                }
            }))
            .send()
            .await?;

        let upload_url = init_response
            .headers()
            .get("X-Goog-Upload-URL")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| GeminiRequestError::InvalidRequestError {
                code: None,
                details: json!({}),
                message: "Missing upload URL in response".to_string(),
                status: None,
            })?
            .to_string();

        let upload_response = self
            .gemini
            .client
            .post(&upload_url)
            .header("Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(self.data)
            .send()
            .await?;

        let file_info: FileInfo = upload_response.json().await?;
        Ok(file_info.file.uri)
    }
}

impl FileStreamUploadRequest {
    pub async fn send(self) -> Result<String, GeminiRequestError> {
        let file = File::open(&self.file_path).await.map_err(|e| {
            GeminiRequestError::InvalidRequestError {
                code: None,
                details: json!({}),
                message: format!("Failed to open file: {}", e),
                status: None,
            }
        })?;

        let metadata =
            file.metadata()
                .await
                .map_err(|e| GeminiRequestError::InvalidRequestError {
                    code: None,
                    details: json!({}),
                    message: format!("Failed to read file metadata: {}", e),
                    status: None,
                })?;

        let num_bytes = metadata.len();
        let file_name = self
            .file_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unnamed_file");

        let api_key = self
            .gemini
            .api_key
            .as_ref()
            .ok_or(GeminiRequestError::AuthenticationMissing)?;

        let init_url = format!(
            "{}/upload/{}/files",
            STANDARD_BASE_URL,
            self.gemini.api_version
        );

        let init_response = self
            .gemini
            .client
            .post(&init_url)
            .query(&[("key", api_key.as_str())])
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Header-Content-Type", &self.mime_type)
            .json(&json!({
                "file": {
                    "display_name": file_name
                }
            }))
            .send()
            .await?;

        let upload_url = init_response
            .headers()
            .get("X-Goog-Upload-URL")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| GeminiRequestError::InvalidRequestError {
                code: None,
                details: json!({}),
                message: "Missing upload URL in response".to_string(),
                status: None,
            })?
            .to_string();

        let upload_response = self
            .gemini
            .client
            .post(&upload_url)
            .header("Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(reqwest::Body::from(file))
            .send()
            .await?;

        let file_info: FileInfo = upload_response.json().await?;
        Ok(file_info.file.uri)
    }
}

impl Gemini {
    pub fn upload_file(&self) -> FileUploadRequestBuilder<file_upload_request_builder::SetGemini> {
        FileUploadRequest::builder().gemini(self.clone())
    }

    pub fn upload_file_stream(
        &self,
    ) -> FileStreamUploadRequestBuilder<file_stream_upload_request_builder::SetGemini> {
        FileStreamUploadRequest::builder().gemini(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_api_key() -> String {
        std::env::var("GOOGLE_AI_API_KEY").expect("GOOGLE_AI_API_KEY must be set")
    }

    #[tokio::test]
    async fn test_file_upload_request_send_data() {
        let api_key = get_api_key();
        let gemini = Gemini::builder()
            .api_key(api_key)
            .api_version("v1beta")
            .build();

        let file_content = include_bytes!("/home/ribelo/documents/kio/2009_1488.pdf");
        let request = gemini
            .upload_file()
            .file_name("test_file.pdf")
            .mime_type("application/pdf")
            .data(file_content)
            .build();

        let result = request.send().await;
        assert!(result.is_ok(), "File upload failed: {:?}", result.err());

        let file_uri = result.expect("Failed to get file URI");
        println!("Uploaded file URI: {file_uri}");
        assert!(
            file_uri.starts_with("https://"),
            "File URI does not start with 'https://'"
        );
    }

    #[ignore]
    #[tokio::test]
    async fn test_file_upload_request_send_file() {
        let api_key = get_api_key();
        let gemini = Gemini::builder()
            .api_key(api_key)
            .api_version("v1beta")
            .build();

        let file_path = std::env::var("TEST_FILE").expect("TEST_FILE env var not set");
        let file_content = std::fs::read(&file_path).expect("Failed to read file");
        let file_name = std::path::Path::new(&file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .expect("Invalid file name");
        let mime_type = match file_name.rsplit('.').next() {
            Some("pdf") => "application/pdf",
            Some("txt") => "text/plain",
            _ => "application/octet-stream",
        };
        let request = gemini
            .upload_file()
            .file_name(file_name)
            .mime_type(mime_type)
            .data(file_content)
            .build();

        let result = request.send().await;
        assert!(result.is_ok(), "File upload failed: {:?}", result.err());

        let file_uri = result.expect("Failed to get file URI");
        println!("Uploaded file URI: {}", file_uri);
        assert!(
            file_uri.starts_with("https://"),
            "File URI does not start with 'https://'"
        );
    }

    #[tokio::test]
    async fn test_file_upload_request_builder_with_data() {
        let api_key = get_api_key();
        let gemini = Gemini::builder().api_key(api_key).build();

        let data = b"Test data".to_vec();
        let request = gemini
            .upload_file()
            .file_name("test.txt")
            .mime_type("text/plain")
            .data(data.clone())
            .build();

        assert_eq!(request.file_name, "test.txt");
        assert_eq!(request.mime_type, "text/plain");
        assert_eq!(request.data, data);

        let result = request.send().await;
        assert!(result.is_ok(), "File upload failed: {:?}", result.err());

        let file_uri = result.unwrap();
        println!("Uploaded file URI: {file_uri}");
        assert!(file_uri.starts_with("https://"));
    }
}
