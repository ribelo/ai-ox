use bon::Builder;
use futures_util::SinkExt;
use futures_util::StreamExt;
use http::Request;
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{Message, protocol::WebSocketConfig},
};
use url::Url;

use super::message_types::LiveApiResponseChunk;
use super::request_configs::{BidiSetupArgs, LiveConnectConfig, ResponseModality};
use super::session::ActiveLiveSession;
use crate::{Gemini, GeminiRequestError, Model};

#[derive(Debug, Builder)]
#[builder(builder_type(vis = "pub"), state_mod(vis = "pub"))]
pub struct LiveOperation {
    #[builder(into)]
    pub gemini: Gemini,

    #[builder(into)]
    pub model: Model,

    #[builder(into)]
    pub system_instruction: Option<crate::content::Content>,

    pub generation_config: Option<crate::generate_content::GenerationConfig>,

    pub safety_settings: Option<crate::generate_content::SafetySettings>,

    pub tools: Option<Vec<crate::tool::Tool>>,

    pub speech_config: Option<super::request_configs::SpeechConfig>,

    pub realtime_input_config: Option<super::request_configs::RealtimeInputConfig>,

    pub response_modalities: Option<Vec<ResponseModality>>,

    pub proactivity: Option<super::request_configs::Proactivity>,

    pub context_window_compression: Option<super::request_configs::ContextWindowCompression>,
}

impl LiveOperation {
    /// Establish a WebSocket connection to the Gemini Live API
    ///
    /// This method consumes the `LiveOperation` and returns an `ActiveLiveSession`
    /// that can be used to send and receive messages over the WebSocket connection.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gemini_ox::{Gemini, Model};
    /// use gemini_ox::live::message_types::{ClientContentPayload, ClientMessage};
    /// use gemini_ox::content::{Content, Role};
    ///
    /// #[tokio::main(flavor = "current_thread")]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let gemini = Gemini::new("your-api-key");
    ///
    ///     // Create a live session
    ///     let mut session = gemini.live_session()
    ///         .model(Model::Gemini20FlashLive001)
    ///         .build()
    ///         .connect()
    ///         .await?;
    ///
    ///     // Send a message
    ///     let content = Content::new(Role::User, vec!["Hello, how are you?"]);
    ///     let payload = ClientContentPayload {
    ///         turns: vec![content],
    ///         turn_complete: Some(true),
    ///     };
    ///     session.send_client_content(payload).await?;
    ///
    ///     // Receive responses
    ///     while let Some(result) = session.receive().await {
    ///         match result? {
    ///             gemini_ox::live::LiveApiResponseChunk::ModelTurn { server_content } => {
    ///                 if let Some(parts) = server_content.model_turn.parts {
    ///                     for part in parts {
    ///                         if let Some(text) = part.text {
    ///                             println!("Model: {}", text);
    ///                         }
    ///                     }
    ///                 }
    ///             }
    ///             gemini_ox::live::LiveApiResponseChunk::TurnComplete { server_content } => {
    ///                 break;
    ///             }
    ///             _ => {} // Handle other message types as needed
    ///         }
    ///     }
    ///
    ///     session.close().await?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `GeminiRequestError` if:
    /// - The WebSocket URL cannot be constructed
    /// - The WebSocket handshake fails
    /// - The initial configuration cannot be sent
    /// - The server's initial response is invalid
    #[allow(clippy::too_many_lines)]
    pub async fn connect(self) -> Result<ActiveLiveSession, GeminiRequestError> {
        // Construct the WebSocket URL
        let url_str = format!(
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={}",
            self.gemini.api_key
        );

        let url =
            Url::parse(&url_str).map_err(|e| GeminiRequestError::UrlBuildError(e.to_string()))?;

        // Create the HTTP request for WebSocket handshake
        let request = Request::builder()
            .uri(url.as_str())
            .header(
                "Host",
                url.host_str()
                    .unwrap_or("generativelanguage.googleapis.com"),
            )
            .header("Upgrade", "websocket")
            .header("Connection", "Upgrade")
            .header(
                "Sec-WebSocket-Key",
                tokio_tungstenite::tungstenite::handshake::client::generate_key(),
            )
            .header("Sec-WebSocket-Version", "13")
            .body(())
            .map_err(|e| {
                GeminiRequestError::UnexpectedResponse(format!(
                    "Failed to build WebSocket request: {e}"
                ))
            })?;

        // Configure WebSocket settings
        let mut ws_config = WebSocketConfig::default();
        ws_config.max_message_size = Some(64 << 20); // 64 MB
        ws_config.max_frame_size = Some(16 << 20); // 16 MB
        ws_config.accept_unmasked_frames = false;

        // Establish the WebSocket connection with TLS
        let (ws_stream, _) = connect_async_with_config(request, Some(ws_config), false)
            .await
            .map_err(|e| {
                GeminiRequestError::UnexpectedResponse(format!("WebSocket connection failed: {e}"))
            })?;

        // Split the WebSocket stream
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        // Prepare the initial configuration
        let connect_config = LiveConnectConfig {
            setup_args: BidiSetupArgs {
                // Field name changed to setup_args, type is BidiSetupArgs
                model: format!("models/{}", self.model),
                generation_config: self.generation_config,
                safety_settings: self.safety_settings,
                tools: self.tools,
                system_instruction: self.system_instruction,
                realtime_input_config: self.realtime_input_config,
            },
            speech_config: self.speech_config, // These are root-level in LiveConnectConfig
            response_modalities: self.response_modalities,
            proactivity: self.proactivity,
            context_window_compression: self.context_window_compression,
        };

        // Send initial configuration
        let config_json =
            serde_json::to_string(&connect_config).map_err(GeminiRequestError::from)?;

        // Initial configuration sent

        ws_sender
            .send(Message::Text(config_json.into()))
            .await
            .map_err(|e| {
                GeminiRequestError::UnexpectedResponse(format!(
                    "Failed to send initial config: {e}"
                ))
            })?;

        // Wait for and validate the server's initial response
        match ws_receiver.next().await {
            Some(Ok(Message::Text(response_text))) => {
                let response: LiveApiResponseChunk = serde_json::from_str(&response_text)
                    .map_err(GeminiRequestError::JsonDeserializationError)?;

                match response {
                    LiveApiResponseChunk::SetupComplete { .. } => Ok(ActiveLiveSession {
                        ws_sender,
                        ws_receiver,
                    }),
                    other => Err(GeminiRequestError::UnexpectedResponse(format!(
                        "Expected SetupComplete message after config, got: {other:?}"
                    ))),
                }
            }
            Some(Ok(Message::Binary(response_bytes))) => {
                // Convert binary to text and parse
                let response_text = String::from_utf8(response_bytes.to_vec()).map_err(|e| {
                    GeminiRequestError::UnexpectedResponse(format!(
                        "Invalid UTF-8 in binary message: {e}"
                    ))
                })?;

                let response: LiveApiResponseChunk = serde_json::from_str(&response_text)
                    .map_err(GeminiRequestError::JsonDeserializationError)?;

                match response {
                    LiveApiResponseChunk::SetupComplete { .. } => Ok(ActiveLiveSession {
                        ws_sender,
                        ws_receiver,
                    }),
                    other => Err(GeminiRequestError::UnexpectedResponse(format!(
                        "Expected SetupComplete message after config, got: {other:?}"
                    ))),
                }
            }
            Some(Ok(Message::Close(close_frame))) => {
                let reason = close_frame.map_or_else(
                    || "Server closed connection without reason".to_string(),
                    |f| format!("Server closed connection: {}", f.reason),
                );
                Err(GeminiRequestError::UnexpectedResponse(reason))
            }
            Some(Ok(other_message)) => Err(GeminiRequestError::UnexpectedResponse(format!(
                "Expected text or binary message from server, got: {other_message:?}"
            ))),
            Some(Err(e)) => Err(GeminiRequestError::UnexpectedResponse(format!(
                "WebSocket error during handshake: {e}"
            ))),
            None => Err(GeminiRequestError::UnexpectedResponse(
                "WebSocket connection closed during handshake".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_operation_builder() {
        let gemini = Gemini::new("test_api_key");

        let operation = LiveOperation::builder()
            .gemini(gemini.clone()) // Clone gemini if it's used later, or ensure it's not moved
            .model(Model::Gemini20FlashLive001)
            .response_modalities(vec![ResponseModality::Audio])
            .build();

        assert_eq!(operation.model.to_string(), "gemini-2.0-flash-live-001");
        assert_eq!(operation.gemini.api_key, "test_api_key");
    }

    #[test]
    fn test_live_connect_config_serialization() {
        let config = LiveConnectConfig {
            setup_args: BidiSetupArgs {
                // Field name changed to setup_args, type is BidiSetupArgs
                model: "models/gemini-2.0-flash-exp".to_string(),
                generation_config: None,
                safety_settings: None,
                tools: None,
                system_instruction: None,
                realtime_input_config: None,
            },
            speech_config: None, // root level
            response_modalities: Some(vec![ResponseModality::Audio]), // root level
            proactivity: None,
            context_window_compression: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        println!("Generated JSON: {json}");
        // Check for a field within 'setup' (which is how setup_args is serialized)
        assert!(
            json.contains("\"setup\":{\"model\":\"models/gemini-2.0-flash-exp\"")
                || json.contains(
                    "\"setup\":{\"model\":\"models/gemini-2.0-flash-exp\",\"generationConfig\":null"
                )
        ); // Serde behavior varies
        // Check for a root-level field
        assert!(json.contains("\"responseModalities\":[\"AUDIO\"]"));
    }
}
