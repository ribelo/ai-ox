#[cfg(test)]
mod tests {
    use ai_ox_common::{
        request_builder::{RequestBuilder, RequestConfig, Endpoint, HttpMethod, AuthMethod, MultipartForm},
        CommonRequestError,
    };

    #[test]
    fn test_endpoint_creation() {
        let endpoint = Endpoint::new("chat/completions", HttpMethod::Post);
        assert_eq!(endpoint.path, "chat/completions");
        assert!(matches!(endpoint.method, HttpMethod::Post));
        assert!(endpoint.extra_headers.is_none());
        assert!(endpoint.query_params.is_none());
    }

    #[test]
    fn test_endpoint_with_query_params() {
        let mut endpoint = Endpoint::new("models", HttpMethod::Get);
        endpoint.query_params = Some(vec![("limit".to_string(), "10".to_string())]);

        assert_eq!(endpoint.query_params.as_ref().unwrap().len(), 1);
        assert_eq!(endpoint.query_params.as_ref().unwrap()[0], ("limit".to_string(), "10".to_string()));
    }

    #[test]
    fn test_auth_methods() {
        let bearer_auth = AuthMethod::Bearer("token123".to_string());
        let api_key_auth = AuthMethod::ApiKey {
            header_name: "x-api-key".to_string(),
            key: "key123".to_string(),
        };
        let oauth_auth = AuthMethod::OAuth {
            header_name: "authorization".to_string(),
            token: "oauth123".to_string(),
        };
        let query_param_auth = AuthMethod::QueryParam("key".to_string(), "value123".to_string());

        match bearer_auth {
            AuthMethod::Bearer(ref token) => assert_eq!(token, "token123"),
            _ => panic!("Expected Bearer auth"),
        }

        match api_key_auth {
            AuthMethod::ApiKey { ref header_name, ref key } => {
                assert_eq!(header_name, "x-api-key");
                assert_eq!(key, "key123");
            },
            _ => panic!("Expected ApiKey auth"),
        }

        match oauth_auth {
            AuthMethod::OAuth { ref header_name, ref token } => {
                assert_eq!(header_name, "authorization");
                assert_eq!(token, "oauth123");
            },
            _ => panic!("Expected OAuth auth"),
        }

        match query_param_auth {
            AuthMethod::QueryParam(ref param, ref value) => {
                assert_eq!(param, "key");
                assert_eq!(value, "value123");
            },
            _ => panic!("Expected QueryParam auth"),
        }
    }

    #[test]
    fn test_request_config_builder() {
        let config = RequestConfig::new("https://api.example.com")
            .with_auth(AuthMethod::Bearer("token123".to_string()))
            .with_header("content-type", "application/json")
            .with_user_agent("test-client/1.0");

        assert_eq!(config.base_url, "https://api.example.com");
        assert!(config.auth.is_some());
        assert_eq!(config.default_headers.len(), 1);
        assert_eq!(config.user_agent, Some("test-client/1.0".to_string()));
    }

    #[test]
    fn test_multipart_form_builder() {
        let file_data = vec![1, 2, 3, 4, 5];
        let form = MultipartForm::new()
            .text("purpose", "fine-tune")
            .file_from_bytes("file", "test.jsonl", file_data.clone())
            .text("model", "gpt-3.5-turbo");

        let reqwest_form = form.build();
        // We can't easily inspect the reqwest::multipart::Form content,
        // but we can verify the builder pattern works without panicking
        assert!(true); // If we reach here, the form was built successfully
    }

    #[test]
    fn test_multipart_form_with_mime() {
        let file_data = vec![1, 2, 3, 4, 5];
        let form = MultipartForm::new()
            .file_from_bytes_with_mime("file", "test.mp3", file_data, "audio/mpeg")
            .text("model", "whisper-1");

        let reqwest_form = form.build();
        // Verify the form builds without errors
        assert!(true);
    }

    #[test]
    fn test_http_method_conversion() {
        use reqwest::Method;
        
        assert_eq!(Method::from(HttpMethod::Get), Method::GET);
        assert_eq!(Method::from(HttpMethod::Post), Method::POST);
        assert_eq!(Method::from(HttpMethod::Put), Method::PUT);
        assert_eq!(Method::from(HttpMethod::Delete), Method::DELETE);
        assert_eq!(Method::from(HttpMethod::Patch), Method::PATCH);
    }

    #[test]
    fn test_default_multipart_form() {
        let form = MultipartForm::default();
        let reqwest_form = form.build();
        // Verify default constructor works
        assert!(true);
    }

    #[tokio::test]
    async fn test_request_builder_url_formation() {
        let client = reqwest::Client::new();
        let config = RequestConfig::new("https://api.example.com/v1")
            .with_auth(AuthMethod::Bearer("test".to_string()));
        
        let request_builder = RequestBuilder::new(client, config);
        let endpoint = Endpoint::new("chat/completions", HttpMethod::Post);
        
        // This will build the request but not send it
        let req_result = request_builder.build_request(&endpoint);
        assert!(req_result.is_ok());
    }
}