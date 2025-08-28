# Migration Guide: From Manual `reqwest` to `RequestBuilder`

This guide provides step-by-step instructions for migrating a custom AI provider crate to use the new `RequestBuilder` pattern from `ai-ox-common`. This refactoring centralizes HTTP logic, reduces boilerplate, and improves consistency.

## Key Breaking Changes

- **Error Handling:** Provider-specific error types should now implement `From<CommonRequestError>`. The `RequestBuilder` returns `Result<T, CommonRequestError>`, which should be mapped to your provider's error enum.
- **Client Structure:** The provider's client struct (e.g., `Client`) should now hold an instance of `RequestBuilder`.
- **API Call Implementation:** Instead of manual `reqwest::Client::post()` or `get()` calls, you will now use methods like `builder.request_json()`, `builder.stream()`, or `builder.request_multipart()`.

---

## Step-by-Step Migration Process

Follow these steps to update your provider crate.

### 1. Add `ai-ox-common` Dependency

In your provider's `Cargo.toml`, add `ai-ox-common` as a dependency:

```toml
[dependencies]
ai-ox-common = { path = "../ai-ox-common" }
# other dependencies
```

### 2. Update Your Client Struct

Modify your main client struct to include the `RequestBuilder`.

**Before:**
```rust
pub struct Client {
    client: reqwest::Client,
    api_key: String,
    // other fields
}
```

**After:**
```rust
use ai_ox_common::RequestBuilder;

pub struct Client {
    builder: RequestBuilder,
}
```

### 3. Instantiate the `RequestBuilder`

In your client's constructor (e.g., `new()`), create a `RequestConfig` and instantiate the `RequestBuilder`.

```rust
use ai-ox-common::{RequestConfig, AuthMethod};

impl Client {
    pub fn new() -> Self {
        let config = RequestConfig::new("https://api.provider.com/v1")
            .with_auth(AuthMethod::Bearer("YOUR_API_KEY".to_string()));
            // .with_header("custom-header", "value");

        let client = reqwest::Client::new(); // or your custom client
        let builder = RequestBuilder::new(client, config);

        Self { builder }
    }

    // Add methods to set the API key
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        let config = self.builder.config.clone().with_auth(AuthMethod::Bearer(api_key.into()));
        self.builder = RequestBuilder::new(self.builder.client.clone(), config);
        self
    }
}
```

### 4. Define `Endpoint` Constants

For each API call, define a constant `Endpoint`.

```rust
use ai_ox_common::{Endpoint, HttpMethod};

const CHAT_COMPLETIONS_ENDPOINT: Endpoint = Endpoint {
    path: "/chat/completions",
    method: HttpMethod::Post,
    extra_headers: None,
    query_params: None,
};

const MODELS_ENDPOINT: Endpoint = Endpoint {
    path: "/models",
    method: HttpMethod::Get,
    extra_headers: None,
    query_params: None,
};
```

### 5. Replace HTTP Calls with `RequestBuilder` Methods

Refactor each API method to use the `RequestBuilder`.

**Before:**
```rust
pub async fn create_chat_completion(
    &self,
    req: &ChatRequest,
) -> Result<ChatResponse, MyError> {
    let res = self.client.post("https://api.provider.com/v1/chat/completions")
        .bearer_auth(&self.api_key)
        .json(req)
        .send()
        .await?;

    if !res.status().is_success() {
        // ... error handling ...
    }

    res.json().await.map_err(Into::into)
}
```

**After:**
```rust
pub async fn create_chat_completion(
    &self,
    req: &ChatRequest,
) -> Result<ChatResponse, MyError> {
    self.builder
        .request_json(&CHAT_COMPLETIONS_ENDPOINT, Some(req))
        .await
        .map_err(Into::into)
}
```

### 6. Update Error Handling

Ensure your provider's error enum can be created from `CommonRequestError`.

```rust
use ai_ox_common::CommonRequestError;

#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("API Error: {0}")]
    ApiError(String),

    #[error("HTTP Request Error: {0}")]
    RequestError(#[from] CommonRequestError),
    // other error variants
}
```
This allows you to use `?` or `.map_err(Into::into)` to convert errors seamlessly.

---

## Example: Migrating a Streaming Endpoint

Migrating streaming endpoints follows the same pattern.

**Before:**
```rust
// manual SSE parsing
```

**After:**
```rust
use futures_util::stream::BoxStream;

pub fn create_chat_completion_stream(
    &self,
    req: &ChatRequest,
) -> Result<BoxStream<'static, Result<ChatStreamChunk, MyError>>, MyError> {
    Ok(Box::pin(
        self.builder
            .stream(&CHAT_COMPLETIONS_ENDPOINT, Some(req))
            .map(|res| res.map_err(Into::into))
    ))
}
```
The `builder.stream` method handles adding `stream: true` to the JSON payload and parsing the SSE stream, yielding deserialized chunks.
