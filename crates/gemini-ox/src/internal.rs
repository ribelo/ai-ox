use ai_ox_common::{
    request_builder::{RequestBuilder, RequestConfig, Endpoint, HttpMethod, AuthMethod},
    CommonRequestError, BoxStream
};
use futures_util::stream::BoxStream as FuturesBoxStream;
use crate::{GeminiRequestError, Gemini, generate_content::{request::GenerateContentRequest, response::GenerateContentResponse}};

/// Convert CommonRequestError to GeminiRequestError
impl From<CommonRequestError> for GeminiRequestError {
    fn from(err: CommonRequestError) -> Self {
        match err {
            CommonRequestError::Http(e) => GeminiRequestError::ReqwestError(e),
            CommonRequestError::Json(e) => GeminiRequestError::SerdeError(e),
            CommonRequestError::InvalidEventData(msg) => GeminiRequestError::InvalidEventData(msg),
            CommonRequestError::AuthenticationMissing => GeminiRequestError::AuthenticationMissing,
            CommonRequestError::InvalidMimeType(msg) => GeminiRequestError::InvalidEventData(msg),
            CommonRequestError::Utf8Error(e) => GeminiRequestError::InvalidEventData(e.to_string()),
        }
    }
}

/// Gemini client helper methods using the common RequestBuilder
pub struct GeminiRequestHelper {
    request_builder: RequestBuilder,
    is_oauth: bool,
}

impl GeminiRequestHelper {
    pub fn new(gemini: &Gemini) -> Result<Self, GeminiRequestError> {
        // Determine authentication method
        let (auth_method, is_oauth) = if let Some(oauth_token) = &gemini.oauth_token {
            (AuthMethod::Bearer(oauth_token.clone()), true)
        } else if let Some(api_key) = &gemini.api_key {
            (AuthMethod::QueryParam("key".to_string(), api_key.clone()), false)
        } else {
            return Err(GeminiRequestError::AuthenticationMissing);
        };

        let config = RequestConfig::new(gemini.base_url())
            .with_auth(auth_method)
            .with_header("content-type", "application/json");

        let request_builder = RequestBuilder::new(gemini.client.clone(), config);

        Ok(Self { request_builder, is_oauth })
    }

    /// Send a generate content request
    pub async fn send_generate_content_request(
        &self, 
        request: &GenerateContentRequest,
        gemini: &Gemini
    ) -> Result<GenerateContentResponse, GeminiRequestError> {
        // Build endpoint based on authentication method
        let endpoint_path = if self.is_oauth {
            "v1internal:generateContent".to_string()
        } else {
            format!("{}/models/{}:generateContent", gemini.api_version, request.model)
        };

        let endpoint = Endpoint::new(endpoint_path, HttpMethod::Post);
        
        // Prepare request body based on API type
        let request_body = if self.is_oauth {
            // Cloud Code Assist API: wrapped format
            let minimal_request = serde_json::json!({
                "contents": request.contents,
                "generationConfig": request.generation_config,
                "systemInstruction": request.system_instruction
            });
            serde_json::json!({
                "model": request.model.to_string(),
                "project": gemini.project_id,
                "request": minimal_request
            })
        } else {
            // Standard API: direct format
            serde_json::to_value(request)
                .map_err(GeminiRequestError::SerdeError)?
        };

        let response: serde_json::Value = self.request_builder
            .request_json(&endpoint, Some(&request_body))
            .await?;

        // Handle wrapped response for OAuth
        if self.is_oauth {
            if let Some(inner_response) = response.get("response") {
                Ok(serde_json::from_value(inner_response.clone())?)
            } else {
                Err(GeminiRequestError::UnexpectedResponse(
                    "Missing 'response' field in Cloud Code Assist API response".to_string()
                ))
            }
        } else {
            Ok(serde_json::from_value(response)?)
        }
    }

    /// Stream a generate content request
    pub fn stream_generate_content_request(
        self, 
        request: GenerateContentRequest,
        gemini: Gemini
    ) -> FuturesBoxStream<'static, Result<GenerateContentResponse, GeminiRequestError>> {
        // Build endpoint based on authentication method
        let endpoint_path = if self.is_oauth {
            format!("v1internal:streamGenerateContent?alt=sse")
        } else {
            format!("{}/models/{}:streamGenerateContent?alt=sse", gemini.api_version, request.model)
        };

        let endpoint = Endpoint::new(endpoint_path, HttpMethod::Post);
        
        // Prepare request body based on API type
        let request_body = if self.is_oauth {
            // Cloud Code Assist API: wrapped format
            let minimal_request = serde_json::json!({
                "contents": request.contents,
                "generationConfig": request.generation_config,
                "systemInstruction": request.system_instruction
            });
            serde_json::json!({
                "model": request.model.to_string(),
                "project": gemini.project_id,
                "request": minimal_request
            })
        } else {
            // Standard API: direct format
            match serde_json::to_value(&request) {
                Ok(value) => value,
                Err(e) => {
                    return Box::pin(futures_util::stream::once(async move {
                        Err(GeminiRequestError::from(CommonRequestError::Json(e)))
                    }));
                }
            }
        };
        
        // Use the common streaming implementation and convert errors
        let common_stream: BoxStream<'static, Result<serde_json::Value, CommonRequestError>> = 
            self.request_builder.stream(&endpoint, Some(&request_body));
        
        Box::pin(async_stream::try_stream! {
            use futures_util::StreamExt;
            
            let mut stream = common_stream;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(value) => {
                        // Handle wrapped response for OAuth
                        let response = if self.is_oauth {
                            if let Some(inner_response) = value.get("response") {
                                serde_json::from_value::<GenerateContentResponse>(inner_response.clone())?
                            } else {
                                continue; // Skip wrapped responses without "response" field
                            }
                        } else {
                            serde_json::from_value::<GenerateContentResponse>(value)?
                        };
                        yield response;
                    },
                    Err(e) => yield Err(GeminiRequestError::from(e))?,
                }
            }
        })
    }
}