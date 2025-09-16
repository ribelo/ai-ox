use super::message_types::{
    ClientContentPayload, ClientMessage, LiveApiResponseChunk, RealtimeInputPayload,
    ToolResponsePayload,
};
use crate::GeminiRequestError;
use futures_util::{
    SinkExt, StreamExt,
    stream::{SplitSink, SplitStream},
};

use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::{
    Error as WsError, Message, protocol::frame::CloseFrame, protocol::frame::coding::CloseCode,
};

pub struct ActiveLiveSession {
    pub(crate) ws_sender:
        SplitSink<WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>, Message>,
    pub(crate) ws_receiver:
        SplitStream<WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>>,
}

impl ActiveLiveSession {
    /// Send client content to the server
    ///
    /// This method sends a client message containing conversation turns to the server.
    /// It's used to provide user input or continue a conversation.
    ///
    /// # Arguments
    ///
    /// * `payload` - The client content payload containing conversation turns
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gemini_ox::live::message_types::{ClientContentPayload};
    /// use gemini_ox::content::{Content, Role};
    ///
    /// # async fn example(session: &mut gemini_ox::live::ActiveLiveSession) -> Result<(), gemini_ox::GeminiRequestError> {
    /// let content = Content::new(Role::User, vec!["Hello!"]);
    /// let payload = ClientContentPayload {
    ///     turns: vec![content],
    ///     turn_complete: Some(true),
    /// };
    /// session.send_client_content(payload).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `GeminiRequestError` if:
    /// - The message cannot be serialized to JSON
    /// - The WebSocket connection fails or is closed
    pub async fn send_client_content(
        &mut self,
        payload: ClientContentPayload,
    ) -> Result<(), GeminiRequestError> {
        let client_message = ClientMessage::ClientContent(payload);
        let msg_json = serde_json::to_string(&client_message).map_err(GeminiRequestError::from)?;
        self.ws_sender
            .send(Message::Text(msg_json.into()))
            .await
            .map_err(Self::map_tungstenite_error)?;
        Ok(())
    }

    /// Send realtime input (e.g., audio) to the server
    ///
    /// This method sends realtime media data such as audio chunks to the server
    /// for live processing and response generation.
    ///
    /// # Arguments
    ///
    /// * `payload` - The realtime input payload containing media chunks
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gemini_ox::live::message_types::RealtimeInputPayload;
    /// use gemini_ox::content::Blob;
    ///
    /// # async fn example(session: &mut gemini_ox::live::ActiveLiveSession) -> Result<(), gemini_ox::GeminiRequestError> {
    /// let audio_chunk = Blob::new("audio/pcm", "base64_encoded_audio_data");
    /// let payload = RealtimeInputPayload {
    ///     media_chunks: Some(vec![audio_chunk]),
    /// };
    /// session.send_realtime_input(payload).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `GeminiRequestError` if:
    /// - The message cannot be serialized to JSON
    /// - The WebSocket connection fails or is closed
    pub async fn send_realtime_input(
        &mut self,
        payload: RealtimeInputPayload,
    ) -> Result<(), GeminiRequestError> {
        let client_message = ClientMessage::RealtimeInput(payload);
        let msg_json = serde_json::to_string(&client_message).map_err(GeminiRequestError::from)?;
        self.ws_sender
            .send(Message::Text(msg_json.into()))
            .await
            .map_err(Self::map_tungstenite_error)?;
        Ok(())
    }

    /// Send tool response to the server
    ///
    /// This method sends function call responses back to the server after
    /// the client has executed the requested tool functions.
    ///
    /// # Arguments
    ///
    /// * `payload` - The tool response payload containing function responses
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gemini_ox::live::message_types::ToolResponsePayload;
    /// use gemini_ox::content::FunctionResponse;
    ///
    /// # async fn example(session: &mut gemini_ox::live::ActiveLiveSession) -> Result<(), gemini_ox::GeminiRequestError> {
    /// let response = FunctionResponse::new("calculator", 42);
    /// let payload = ToolResponsePayload {
    ///     function_responses: vec![response],
    /// };
    /// session.send_tool_response(payload).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `GeminiRequestError` if:
    /// - The message cannot be serialized to JSON
    /// - The WebSocket connection fails or is closed
    pub async fn send_tool_response(
        &mut self,
        payload: ToolResponsePayload,
    ) -> Result<(), GeminiRequestError> {
        let client_message = ClientMessage::ToolResponse(payload);
        let msg_json = serde_json::to_string(&client_message).map_err(GeminiRequestError::from)?;
        self.ws_sender
            .send(Message::Text(msg_json.into()))
            .await
            .map_err(Self::map_tungstenite_error)?;
        Ok(())
    }

    /// Receive a message from the server
    ///
    /// This method receives and deserializes messages from the Live API server.
    /// It automatically handles WebSocket protocol messages like Ping/Pong.
    ///
    /// The method returns `None` when the connection is closed cleanly by the server,
    /// and `Some(Err(...))` when there's an error processing a message.
    ///
    /// # Returns
    ///
    /// * `Some(Ok(chunk))` - Successfully received and parsed a message
    /// * `Some(Err(error))` - Error receiving or parsing a message
    /// * `None` - Connection closed cleanly or stream ended
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gemini_ox::live::LiveApiResponseChunk;
    ///
    /// # async fn example(session: &mut gemini_ox::live::ActiveLiveSession) -> Result<(), Box<dyn std::error::Error>> {
    /// while let Some(result) = session.receive().await {
    ///     match result? {
    ///         LiveApiResponseChunk::ModelTurn { server_content } => {
    ///             println!("Received model response");
    ///         }
    ///         LiveApiResponseChunk::TurnComplete { server_content } => {
    ///             println!("Turn completed");
    ///             break;
    ///         }
    ///         LiveApiResponseChunk::SetupComplete { _setup } => {
    ///             println!("Session setup complete");
    ///         }
    ///         _ => {} // Handle other message types
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn receive(&mut self) -> Option<Result<LiveApiResponseChunk, GeminiRequestError>> {
        loop {
            match self.ws_receiver.next().await {
                Some(Ok(message)) => {
                    match message {
                        Message::Text(text) => {
                            // Debug: print raw message from API
                            println!("📨 DEBUG: Raw API response: {text}");
                            return Some(
                                serde_json::from_str::<LiveApiResponseChunk>(&text)
                                    .map_err(|e| {
                                        println!("❌ DEBUG: Failed to parse API response as LiveApiResponseChunk: {e}");
                                        println!("❌ DEBUG: Raw response was: {text}");
                                        GeminiRequestError::JsonDeserializationError(e)
                                    }),
                            );
                        }
                        Message::Binary(data) => {
                            // Convert binary to text and parse
                            match String::from_utf8(data.to_vec()) {
                                Ok(text) => {
                                    return Some(
                                        serde_json::from_str::<LiveApiResponseChunk>(&text)
                                            .map_err(|e| {
                                                println!("❌ DEBUG: Failed to parse binary API response: {e}");
                                                GeminiRequestError::JsonDeserializationError(e)
                                            }),
                                    );
                                }
                                Err(e) => {
                                    println!("❌ DEBUG: Binary message is not valid UTF-8: {e}");
                                    return Some(Err(GeminiRequestError::UnexpectedResponse(
                                        format!("Invalid UTF-8 in binary message: {e}"),
                                    )));
                                }
                            }
                        }
                        Message::Ping(_) => {
                            // tokio-tungstenite handles Pong responses automatically.
                            // Continue to get the next actual message.
                            continue;
                        }
                        Message::Pong(_) => {
                            // We are not sending Pings manually, so Pongs are unexpected unless library sends them.
                            // Continue to get the next actual message.
                            continue;
                        }
                        Message::Close(close_frame_opt) => {
                            if let Some(frame) = close_frame_opt {
                                println!(
                                    "🔌 WebSocket closed by server with code: {:?}, reason: '{}'",
                                    frame.code, frame.reason
                                );
                            } else {
                                println!("🔌 WebSocket closed by server (no specific frame info).");
                            }
                            return None; // Clean close by server
                        }
                        Message::Frame(_) => {
                            // Raw frame, likely not expected at this level of abstraction
                            return Some(Err(GeminiRequestError::UnexpectedResponse(
                                "Received unexpected raw WebSocket frame".to_string(),
                            )));
                        }
                    }
                }
                Some(Err(e)) => {
                    // More detailed logging for WebSocket errors
                    println!("❌ ActiveLiveSession::receive encountered WebSocket error: {e:#?}");

                    // Handle specific tungstenite errors
                    if matches!(e, WsError::ConnectionClosed | WsError::AlreadyClosed) {
                        return None;
                    }
                    return Some(Err(Self::map_tungstenite_error(e)));
                }
                None => {
                    println!("ActiveLiveSession::receive received None (stream ended)"); // Added logging
                    return None; // Stream ended
                }
            }
        }
    }

    /// Close the WebSocket connection
    ///
    /// This method gracefully closes the WebSocket connection by sending a close frame
    /// to the server and then closing the connection. It should be called when you're
    /// done with the live session.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(mut session: gemini_ox::live::ActiveLiveSession) -> Result<(), gemini_ox::GeminiRequestError> {
    /// // ... use the session ...
    /// session.close().await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `GeminiRequestError` if the close frame cannot be sent or
    /// if there's an error closing the underlying connection.
    pub async fn close(&mut self) -> Result<(), GeminiRequestError> {
        // Send a WebSocket Close frame
        let close_frame = CloseFrame {
            code: CloseCode::Normal,
            reason: "Client initiated close".into(),
        };
        self.ws_sender
            .send(Message::Close(Some(close_frame)))
            .await
            .map_err(Self::map_tungstenite_error)?;
        // Ensure all messages are flushed
        self.ws_sender
            .close()
            .await
            .map_err(Self::map_tungstenite_error)
    }

    /// Map tungstenite errors to GeminiRequestError
    fn map_tungstenite_error(error: WsError) -> GeminiRequestError {
        match error {
            WsError::ConnectionClosed | WsError::AlreadyClosed => {
                GeminiRequestError::UnexpectedResponse("WebSocket connection closed".to_string())
            }
            WsError::Io(io_err) => GeminiRequestError::IoError(io_err),
            WsError::Tls(tls_err) => {
                GeminiRequestError::UnexpectedResponse(format!("TLS error: {tls_err}"))
            }
            WsError::Capacity(cap_err) => {
                GeminiRequestError::UnexpectedResponse(format!("Capacity error: {cap_err}"))
            }
            WsError::Protocol(proto_err) => {
                GeminiRequestError::UnexpectedResponse(format!("Protocol error: {proto_err}"))
            }

            WsError::Utf8 => GeminiRequestError::UnexpectedResponse(
                "UTF-8 encoding error in WebSocket message".to_string(),
            ),
            WsError::Url(url_err) => {
                GeminiRequestError::UrlBuildError(format!("URL error: {url_err}"))
            }
            WsError::Http(response) => GeminiRequestError::UnexpectedResponse(format!(
                "HTTP error during WebSocket handshake: {}",
                response.status()
            )),
            WsError::HttpFormat(http_err) => {
                GeminiRequestError::UnexpectedResponse(format!("HTTP format error: {http_err}"))
            }
            _ => GeminiRequestError::UnexpectedResponse(format!("WebSocket error: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{Content, Role};
    use serde_json;

    #[test]
    fn test_serialize_client_content_message() {
        let content = Content::new(Role::User, vec!["Hello, world!"]);
        let payload = ClientContentPayload {
            turns: vec![content],
            turn_complete: Some(true),
        };
        let client_message = ClientMessage::ClientContent(payload);

        let json = serde_json::to_string(&client_message).unwrap();
        assert!(json.contains("clientContent"));
        assert!(json.contains("Hello, world!"));
        assert!(json.contains("turnComplete"));
    }

    #[test]
    fn test_deserialize_model_turn_chunk() {
        let json = r#"{
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "text": "Hello! How can I help you today?"
                        }
                    ]
                }
            }
        }"#;

        let chunk: Result<LiveApiResponseChunk, _> = serde_json::from_str(json);
        match chunk {
            Ok(LiveApiResponseChunk::ModelTurn { server_content }) => {
                assert!(server_content.model_turn.parts.is_some());
                let parts = server_content.model_turn.parts.unwrap();
                assert_eq!(parts.len(), 1);
                assert_eq!(
                    parts[0].text,
                    Some("Hello! How can I help you today?".to_string())
                );
            }
            Ok(_) => {
                panic!("Expected ModelTurn variant");
            }
            Err(e) => {
                panic!("Failed to deserialize: {e}");
            }
        }
    }

    #[test]
    fn test_deserialize_setup_complete() {
        let json = r#"{"setupComplete": {}}"#;
        let chunk: Result<LiveApiResponseChunk, _> = serde_json::from_str(json);
        match chunk {
            Ok(chunk) => {
                assert!(matches!(chunk, LiveApiResponseChunk::SetupComplete { .. }));
            }
            Err(e) => {
                panic!("Failed to deserialize: {e}");
            }
        }
    }
}
