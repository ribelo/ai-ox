pub mod live_operation;
pub mod message_types;
pub mod request_configs;
pub mod session;

#[cfg(feature = "audio")]
pub mod audio_input;
#[cfg(feature = "video")]
pub mod video_input;

pub use live_operation::LiveOperation;
pub use message_types::{ClientMessage, LiveApiResponseChunk};
pub use request_configs::LiveConnectConfig;
pub use session::ActiveLiveSession;

#[cfg(feature = "audio")]
pub use audio_input::AudioRecorder;
#[cfg(feature = "video")]
pub use video_input::VideoCapturer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::{Content, Role};
    use crate::generate_content::GenerationConfig;
    use crate::{Gemini, Model};
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::{SampleFormat, StreamConfig};
    use message_types::{ClientContentPayload, RealtimeInputPayload};
    use ringbuf::HeapRb; // Using HeapRb for simplicity in an async context, producer will be moved.
    use std::time::Duration;

    /// Test audio input functionality
    #[cfg(feature = "audio")]
    #[tokio::test]
    #[ignore = "Requires audio hardware and API key"]
    async fn test_live_api_with_audio_input() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
            println!("GOOGLE_AI_API_KEY not set, skipping test_live_api_with_audio_input");
            return Ok(());
        };

        let gemini = Gemini::builder().api_key(api_key).build();

        let generation_config = GenerationConfig {
            response_modalities: Some(vec!["TEXT".to_string()]),
            ..Default::default()
        };

        let mut session = gemini
            .live_session()
            .model(Model::Gemini20FlashLive001)
            .generation_config(generation_config)
            .build()
            .connect()
            .await?;

        // Start audio capture
        let (audio_recorder, mut audio_rx) = AudioRecorder::start_capturing()?;

        // Send a text message first
        let content = Content::new(Role::User, vec!["I'm going to send you some audio data."]);
        let payload = ClientContentPayload {
            turns: vec![content],
            turn_complete: Some(false), // Not complete yet, audio will follow
        };
        session.send_client_content(payload).await?;

        // Capture audio for a short duration and send it
        let timeout = tokio::time::timeout(Duration::from_secs(3), async {
            if let Some(audio_chunk) = audio_rx.recv().await {
                let realtime_payload = RealtimeInputPayload {
                    media_chunks: Some(vec![audio_chunk]),
                };
                session.send_realtime_input(realtime_payload).await?;
            }
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })
        .await;

        drop(audio_recorder); // Stop audio capture

        if timeout.is_ok() {
            // Send turn complete
            let content = Content::new(Role::User, vec![""]);
            let payload = ClientContentPayload {
                turns: vec![content],
                turn_complete: Some(true),
            };
            session.send_client_content(payload).await?;

            // Wait for response
            let mut received_response = false;
            while let Some(result) = session.receive().await {
                match result {
                    Ok(LiveApiResponseChunk::ModelTurn { .. }) => {
                        received_response = true;
                    }
                    Ok(LiveApiResponseChunk::TurnComplete { .. }) => {
                        break;
                    }
                    Ok(LiveApiResponseChunk::Interrupted { .. }) => {
                        break;
                    }
                    Ok(_) => {} // Other message types
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        break;
                    }
                }
            }

            assert!(
                received_response,
                "Should have received a response to audio input"
            );
        }

        session.close().await?;
        Ok(())
    }

    /// Test video input functionality
    #[cfg(feature = "video")]
    #[tokio::test]
    #[ignore = "Requires camera hardware and API key"]
    async fn test_live_api_with_video_input() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
            println!("GOOGLE_AI_API_KEY not set, skipping test_live_api_with_video_input");
            return Ok(());
        };

        let gemini = Gemini::builder().api_key(api_key).build();

        let generation_config = GenerationConfig {
            response_modalities: Some(vec!["TEXT".to_string()]),
            ..Default::default()
        };

        let mut session = gemini
            .live_session()
            .model(Model::Gemini20FlashLive001)
            .generation_config(generation_config)
            .build()
            .connect()
            .await?;

        // Start video capture
        let (video_capturer, mut video_rx) = VideoCapturer::start_capturing_default()?;

        // Send a text message first
        let content = Content::new(
            Role::User,
            vec!["I'm going to send you a video frame. Describe what you see."],
        );
        let payload = ClientContentPayload {
            turns: vec![content],
            turn_complete: Some(false), // Not complete yet, video will follow
        };
        session.send_client_content(payload).await?;

        // Capture video for a short duration and send a frame
        let timeout = tokio::time::timeout(Duration::from_secs(5), async {
            if let Some(video_chunk) = video_rx.recv().await {
                let realtime_payload = RealtimeInputPayload {
                    media_chunks: Some(vec![video_chunk]),
                };
                session.send_realtime_input(realtime_payload).await?;
            }
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })
        .await;

        drop(video_capturer); // Stop video capture

        if timeout.is_ok() {
            // Send turn complete
            let content = Content::new(Role::User, vec![""]);
            let payload = ClientContentPayload {
                turns: vec![content],
                turn_complete: Some(true),
            };
            session.send_client_content(payload).await?;

            // Wait for response
            let mut received_response = false;
            while let Some(result) = session.receive().await {
                match result {
                    Ok(LiveApiResponseChunk::ModelTurn { .. }) => {
                        received_response = true;
                    }
                    Ok(LiveApiResponseChunk::TurnComplete { .. }) => {
                        break;
                    }
                    Ok(LiveApiResponseChunk::Interrupted { .. }) => {
                        break;
                    }
                    Ok(_) => {} // Other message types
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        break;
                    }
                }
            }

            assert!(
                received_response,
                "Should have received a response to video input"
            );
        }

        session.close().await?;
        Ok(())
    }

    /// Test that audio and video features can be used together
    #[cfg(all(feature = "audio", feature = "video"))]
    #[tokio::test]
    #[ignore = "Requires audio and video hardware and API key"]
    async fn test_live_api_multimodal() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
            println!("GOOGLE_AI_API_KEY not set, skipping test_live_api_multimodal");
            return Ok(());
        };

        let gemini = Gemini::builder().api_key(api_key).build();

        let generation_config = GenerationConfig {
            response_modalities: Some(vec!["TEXT".to_string(), "AUDIO".to_string()]),
            ..Default::default()
        };

        let mut session = gemini
            .live_session()
            .model(Model::Gemini20FlashLive001)
            .generation_config(generation_config)
            .build()
            .connect()
            .await?;

        // Start both audio and video capture
        let (audio_recorder, mut audio_rx) = AudioRecorder::start_capturing()?;
        let (video_capturer, mut video_rx) = VideoCapturer::start_capturing_default()?;

        // Send initial message
        let content = Content::new(
            Role::User,
            vec!["I'm sending you both audio and video. Please respond with both text and audio."],
        );
        let payload = ClientContentPayload {
            turns: vec![content],
            turn_complete: Some(false),
        };
        session.send_client_content(payload).await?;

        // Send some audio and video data
        let mut audio_sent = false;
        let mut video_sent = false;

        let timeout = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    audio_chunk = audio_rx.recv(), if !audio_sent => {
                        if let Some(chunk) = audio_chunk {
                            let payload = RealtimeInputPayload {
                                media_chunks: Some(vec![chunk]),
                            };
                            session.send_realtime_input(payload).await?;
                            audio_sent = true;
                        }
                    }
                    video_chunk = video_rx.recv(), if !video_sent => {
                        if let Some(chunk) = video_chunk {
                            let payload = RealtimeInputPayload {
                                media_chunks: Some(vec![chunk]),
                            };
                            session.send_realtime_input(payload).await?;
                            video_sent = true;
                        }
                    }
                    else => break,
                }

                if audio_sent && video_sent {
                    break;
                }
            }
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })
        .await;

        drop(audio_recorder);
        drop(video_capturer);

        if timeout.is_ok() {
            // Send turn complete
            let content = Content::new(Role::User, vec![""]);
            let payload = ClientContentPayload {
                turns: vec![content],
                turn_complete: Some(true),
            };
            session.send_client_content(payload).await?;

            // Wait for multimodal response
            let mut received_text = false;
            let mut received_audio = false;

            while let Some(result) = session.receive().await {
                match result {
                    Ok(LiveApiResponseChunk::ModelTurn { server_content }) => {
                        if let Some(parts) = server_content.model_turn.parts {
                            for part in parts {
                                if part.text.is_some() {
                                    received_text = true;
                                }
                                if let Some(inline_data) = part.inline_data {
                                    if inline_data.mime_type.as_deref()
                                        == Some("audio/pcm;rate=24000")
                                    {
                                        received_audio = true;
                                    }
                                }
                            }
                        }
                    }
                    Ok(LiveApiResponseChunk::TurnComplete { .. }) => {
                        break;
                    }
                    Ok(LiveApiResponseChunk::Interrupted { .. }) => {
                        break;
                    }
                    Ok(_) => {} // Other message types
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        break;
                    }
                }
            }

            // We should receive at least text response
            assert!(received_text, "Should have received text response");
            println!(
                "Multimodal test completed - Text: {}, Audio: {}",
                received_text, received_audio
            );
        }

        session.close().await?;
        Ok(())
    }

    /// Test the Live API by establishing a WebSocket connection, sending a message,
    /// and receiving a response. This test validates the complete live session workflow:
    /// 1. Connect to the Live API WebSocket endpoint
    /// 2. Wait for setup completion
    /// 3. Send a user message
    /// 4. Receive and validate the model's response
    /// 5. Properly close the connection
    #[tokio::test]
    #[ignore = "Requires GOOGLE_AI_API_KEY environment variable and makes actual API calls"]
    #[allow(clippy::too_many_lines)]
    async fn test_live_api_session_text() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Initialize rustls crypto provider for TLS support
        let _ = rustls::crypto::ring::default_provider().install_default();

        dbg!(1);
        let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
            // Skip test if API key isn't set instead of failing
            println!("GOOGLE_AI_API_KEY not set, skipping test_live_api_session");
            return Ok(());
        };

        let gemini = Gemini::builder().api_key(api_key).build();

        dbg!(2);
        // Set a reasonable timeout (30 seconds) for the entire test
        let timeout_result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            // Create a GenerationConfig with responseModalities and responseMimeType
            dbg!(3);
            let generation_config = GenerationConfig {
                response_modalities: Some(vec!["TEXT".to_string()]),
                response_mime_type: Some("text/plain".to_string()),
                ..Default::default()
            };

            dbg!(4);
            // Create a live session
            let mut session = gemini
                .live_session()
                .model(Model::Gemini20FlashLive001)
                .generation_config(generation_config) // Set generation_config
                .build()
                .connect()
                .await?;

            dbg!(5);
            // Session is now connected and setup is complete as connect() handles it.
            let mut responses = Vec::new();
            let mut stream_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

            // Send a simple message
            let content = Content::new(
                Role::User,
                vec!["Hello! Please respond with a short greeting."],
            );
            let payload = ClientContentPayload {
                turns: vec![content],
                turn_complete: Some(true),
            };
            session.send_client_content(payload).await?;

            // Receive the response
            let mut turn_complete = false;
            while let Some(result) = session.receive().await {
                match result {
                    Ok(LiveApiResponseChunk::ModelTurn { server_content }) => {
                        if let Some(parts) = server_content.model_turn.parts {
                            for part in parts {
                                if let Some(text) = part.text {
                                    responses.push(text);
                                }
                            }
                        }
                    }
                    Ok(LiveApiResponseChunk::TurnComplete { .. }) => {
                        turn_complete = true;
                        break;
                    }
                    Ok(LiveApiResponseChunk::Interrupted { .. }) => {
                        break; // Connection was interrupted
                    }
                    Ok(other) => {
                        // Handle other message types
                        println!("Received unhandled Ok message in second loop: {:?}", other);
                    }
                    Err(err) => {
                        stream_error = Some(Box::new(err));
                        break;
                    }
                }
            }

            // Check for stream errors
            if let Some(err) = stream_error {
                return Err(err);
            }

            // Close the session
            session.close().await?;

            // Verify we got a response
            assert!(
                turn_complete || !responses.is_empty(),
                "Should receive either turn complete or at least one response part"
            );

            if !responses.is_empty() {
                let full_response = responses.join("");
                assert!(
                    !full_response.is_empty(),
                    "Combined response should not be empty"
                );
                assert!(
                    full_response.len() > 2,
                    "Response should have meaningful content"
                );
            }

            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })
        .await;

        // Check for timeout
        match timeout_result {
            Ok(result) => result,
            Err(_) => Err(Box::<dyn std::error::Error + Send + Sync>::from(
                "Test timed out after 30 seconds",
            )),
        }
    }
    #[tokio::test]
    #[ignore = "Requires GOOGLE_AI_API_KEY, makes API calls, and attempts audio playback"]
    #[allow(clippy::too_many_lines)]
    async fn test_live_api_session_audio() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Initialize rustls crypto provider for TLS support
        let _ = rustls::crypto::ring::default_provider().install_default();

        let Ok(api_key) = std::env::var("GOOGLE_AI_API_KEY") else {
            println!("GOOGLE_AI_API_KEY not set, skipping test_live_api_session_audio");
            return Ok(());
        };

        let gemini = Gemini::builder().api_key(api_key).build();

        // Set a reasonable timeout (e.g., 30 seconds) for the entire test
        let timeout_result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            // Configure for audio output
            let generation_config = GenerationConfig {
                response_modalities: Some(vec!["AUDIO".to_string()]), // Request AUDIO modality for parts
                ..Default::default()
            };

            // Create a live session
            let mut session = gemini
                .live_session()
                .model(Model::Gemini20FlashLive001) // Or other suitable model
                .generation_config(generation_config)
                .build()
                .connect()
                .await?;

            // CPAL setup for audio playback
            let host = cpal::default_host();
            let device = host
                .default_output_device()
                .ok_or_else(|| "No default output device available".to_string())?;

            // Find a suitable output configuration: 24000 Hz, 1 channe
            let supported_config = device
                .supported_output_configs()
                .map_err(|e| format!("Error querying configs: {}", e))?
                .find(|config| {
                    config.sample_format() == SampleFormat::I16
                        && config.channels() == 1
                        && config.min_sample_rate() <= cpal::SampleRate(24000)
                        && config.max_sample_rate() >= cpal::SampleRate(24000)
                })
                .ok_or_else(|| {
                    "No suitable i16 24kHz mono config found for output device".to_string()
                })?
                .with_sample_rate(cpal::SampleRate(24000));

            let stream_config: StreamConfig = supported_config.config();

            // Ring buffer for audio samples (i16). Capacity for ~5 seconds of audio.
            let ring_buffer_capacity = 24000 * 1 * 5; // sample_rate * channels * seconds
            let (mut producer, mut consumer) = HeapRb::<i16>::new(ring_buffer_capacity).split();

            let stream = device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let written = consumer.pop_slice(data);
                    // Zero out the rest of the buffer if not enough samples
                    for sample_ref in data.iter_mut().skip(written) {
                        *sample_ref = 0; // Or cpal::Sample::EQUILIBRIUM for the format
                    }
                },
                |err| eprintln!("CPAL stream error: {}", err),
                None, // Optional timeout
            )?;
            stream.play()?; // Start playback

            let mut samples_written_to_buffer_count = 0_usize;
            let mut stream_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;
            let mut final_event_received = false;

            // Send a simple text message
            let content = Content::new(Role::User, vec!["Say the word 'hello'"]);
            let payload = ClientContentPayload {
                turns: vec![content],
                turn_complete: Some(true),
            };
            session.send_client_content(payload).await?;

            // Receive the response
            while let Some(result) = session.receive().await {
                match result {
                    Ok(LiveApiResponseChunk::ModelTurn { server_content }) => {
                        if let Some(parts) = server_content.model_turn.parts {
                            for part in parts {
                                if let Some(inline_data) = part.inline_data {
                                    if inline_data.mime_type == "audio/pcm;rate=24000" {
                                        let b64_data = &inline_data.data;
                                        match BASE64_STANDARD.decode(b64_data) {
                                            Ok(pcm_bytes) => {
                                                // Assuming PCM data is 16-bit little-endian
                                                for chunk_bytes in pcm_bytes.chunks_exact(2) {
                                                    let sample = i16::from_le_bytes([
                                                        chunk_bytes[0],
                                                        chunk_bytes[1],
                                                    ]);
                                                    if producer.push(sample).is_ok() {
                                                        samples_written_to_buffer_count += 1;
                                                    } else {
                                                        // Optionally log if buffer is full and samples are dropped
                                                        // eprintln!("Audio ring buffer full, dropping sample.");
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                    "Base64 decode error for audio data: {}",
                                                    e
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(LiveApiResponseChunk::TurnComplete { .. }) => {
                        final_event_received = true;
                        break;
                    }
                    Ok(LiveApiResponseChunk::GenerationComplete { .. }) => {
                        // This can also signify the end of content generation for the turn
                        final_event_received = true;
                        // We might still get a TurnComplete after this, so don't break immediately
                        // unless specific logic dictates. For this test, we'll wait for TurnComplete or Interrupted.
                    }
                    Ok(LiveApiResponseChunk::Interrupted { .. }) => {
                        final_event_received = true;
                        break;
                    }
                    Ok(other) => {
                        println!("Received unhandled Ok message in audio test: {:?}", other);
                    }
                    Err(err) => {
                        stream_error = Some(Box::new(err));
                        break;
                    }
                }
            }

            if let Some(err) = stream_error {
                return Err(err);
            }

            session.close().await?;

            assert!(
                final_event_received,
                "Should have received a TurnComplete, GenerationComplete, or Interrupted message"
            );
            // assert!(
            //     !audio_data_chunks.is_empty(),
            //     "Should have received audio data chunks"
            // );
            // assert!(!audio_data_chunks.is_empty(), "Should have received audio data chunks");
            assert!(
                samples_written_to_buffer_count > 0,
                "Should have written audio samples to the playback buffer"
            );
            println!(
                "{} audio samples written to buffer for playback.",
                samples_written_to_buffer_count
            );

            // Allow some time for audio to play out
            // Note: In a CI environment, this sleep is just a delay.
            // For local testing, this gives you time to hear the audio.
            tokio::time::sleep(Duration::from_secs(5)).await;

            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })
        .await;

        match timeout_result {
            Ok(result) => result,
            Err(_) => Err(Box::<dyn std::error::Error + Send + Sync>::from(
                "Audio test timed out after 30 seconds",
            )),
        }
    }
}
