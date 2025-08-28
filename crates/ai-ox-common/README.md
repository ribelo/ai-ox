# `ai-ox-common` Crate

This crate provides a shared infrastructure for building AI provider clients in Rust. It's the backbone of the `ai-ox` ecosystem, offering a generic `RequestBuilder` that handles common HTTP patterns, authentication, error handling, and streaming.

The primary goal of this crate is to eliminate boilerplate code and ensure consistency across all provider-specific crates like `openai-ox`, `anthropic-ox`, and `mistral-ox`.

## The `RequestBuilder` Pattern

The `RequestBuilder` pattern is a declarative way to define and execute API requests. Instead of manually constructing `reqwest` clients and handling JSON serialization/deserialization for every API call, you define an `Endpoint` and let the `RequestBuilder` handle the rest.

### Benefits
- **Reduced Boilerplate:** Eliminates hundreds of lines of repetitive HTTP client code.
- **Consistency:** Ensures all provider clients behave similarly.
- **Centralized Logic:** Authentication, error handling, and streaming are handled in one place.
- **Type Safety:** Leverages Rust's type system to ensure request and response types are correct.
- **Maintainability:** Simplifies adding new endpoints and providers.

---

## Before and After: A Migration Example

The `RequestBuilder` significantly simplifies provider implementations. Here’s a conceptual look at how code changes when migrating an API call.

### Before: Manual `reqwest` Calls

```rust
// conceptual example of old pattern
pub async fn create_chat_completion(
    &self,
    req: &ChatCompletionRequest,
) -> Result<ChatCompletion, ProviderError> {
    let mut headers = self.get_headers()?;
    headers.insert("Content-Type", "application/json".parse().unwrap());

    let res = self.client.post("https://api.example.com/v1/chat/completions")
        .headers(headers)
        .json(req)
        .send()
        .await
        .map_err(|e| ProviderError::HttpRequest(e.to_string()))?;

    if !res.status().is_success() {
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        return Err(ProviderError::ApiError(format!("{} - {}", status, text)));
    }

    res.json::<ChatCompletion>()
        .await
        .map_err(|e| ProviderError::Deserialization(e.to_string()))
}
```

### After: Using the `RequestBuilder`

With the `RequestBuilder`, the same operation becomes much cleaner.

1.  **Define the Endpoint:**
    First, you define the endpoint declaratively. This can be stored as a constant or created on the fly.

    ```rust
    use ai_ox_common::{Endpoint, HttpMethod};

    const CHAT_COMPLETIONS_ENDPOINT: Endpoint = Endpoint {
        path: "/v1/chat/completions",
        method: HttpMethod::Post,
        // Other options can be set here
    };
    ```

2.  **Execute the Request:**
    The client method is now just a thin wrapper around the `RequestBuilder`.

    ```rust
    // new implementation
    use ai_ox_common::CommonRequestError;

    pub async fn create_chat_completion(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<ChatCompletion, CommonRequestError> {
        self.builder.request_json(
            &CHAT_COMPLETIONS_ENDPOINT,
            Some(req)
        ).await
    }
    ```

---

## Core Components

### `Endpoint` Struct

The `Endpoint` struct is used to define an API endpoint.

```rust
pub struct Endpoint {
    pub path: &'static str,
    pub method: HttpMethod,
    pub extra_headers: Option<HashMap<String, String>>,
    pub query_params: Option<Vec<(String, String)>>,
}
```

- `path`: The API path (e.g., `/v1/chat/completions`).
- `method`: The HTTP method (`HttpMethod::Get`, `HttpMethod::Post`, etc.).
- `extra_headers`: Optional headers specific to this endpoint.
- `query_params`: Optional query parameters.

### `AuthMethod` Enum

The `AuthMethod` enum handles different authentication strategies. This is configured once when the client is created.

```rust
pub enum AuthMethod {
    Bearer(String),
    ApiKey { header_name: String, key: String },
    OAuth { header_name: String, token: String },
    QueryParam(String, String),
}
```

The `RequestBuilder` automatically adds the correct headers or query parameters based on the selected `AuthMethod`.

### `SseParser` for Streaming

The `RequestBuilder::stream` method returns a stream of Server-Sent Events (SSE). The underlying `SseParser` handles the complexity of parsing the event stream.

To consume a stream:

```rust
use futures_util::stream::StreamExt;

let mut stream = client.create_chat_completion_stream(&req).await?;

while let Some(result) = stream.next().await {
    match result {
        Ok(chunk) => {
            // process chunk
        },
        Err(e) => {
            // handle error
        }
    }
}
```

The `stream` method automatically adds `stream: true` to the request body and handles parsing the response.

### `MultipartForm` Helper

For file uploads, the `MultipartForm` helper simplifies building `multipart/form-data` requests.

```rust
use ai_ox_common::MultipartForm;

// Example for uploading a file
pub async fn upload_file(
    &self,
    file_path: &str,
    purpose: &str,
) -> Result<FileUploadResponse, CommonRequestError> {
    let data = tokio::fs::read(file_path).await?;
    let form = MultipartForm::new()
        .text("purpose", purpose.to_string())
        .file_from_bytes("file", "my-file.txt", data)
        .build();

    self.builder.request_multipart(
        &FILES_ENDPOINT,
        form
    ).await
}
```

The `request_multipart` method handles sending the form and deserializing the JSON response.
