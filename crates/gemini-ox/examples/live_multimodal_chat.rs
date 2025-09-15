//! Live Multimodal Chat Example
//!
//! This example demonstrates how to use the Gemini Live API with optional audio and video input.
//!
//! Features:
//! - Text input/output (always available)
//! - Audio input from microphone (requires `audio` feature)
//! - Audio output playback (requires `audio-output` feature)
//! - Video input from camera (requires `video` feature)
//!
//! Usage:
//! ```bash
//! # Text only
//! cargo run --example live_multimodal_chat
//!
//! # With audio input support
//! cargo run --example live_multimodal_chat --features audio
//!
//! # With audio output support
//! cargo run --example live_multimodal_chat --features audio-output
//!
//! # With video support
//! cargo run --example live_multimodal_chat --features video
//!
//! # With full audio and video support
//! cargo run --example live_multimodal_chat --features audio,audio-output,video
//! ```

use clap::Parser;
use gemini_ox::content::{Content, Role};
use gemini_ox::generate_content::GenerationConfig;
use gemini_ox::live::{
    ActiveLiveSession, LiveApiResponseChunk, message_types::ClientContentPayload,
};
use gemini_ox::{Gemini, Model};
use std::io::{self, Write};
use tokio::io::{AsyncBufReadExt, BufReader};

#[cfg(feature = "audio")]
use gemini_ox::live::AudioRecorder;
#[cfg(feature = "video")]
use gemini_ox::live::VideoCapturer;

// Audio output support using cpal (requires audio-output feature)
#[cfg(feature = "audio-output")]
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};

// Audio playback dependencies for server audio responses (conditional)
#[cfg(feature = "audio-output")]
use {
    cpal::traits::{DeviceTrait, HostTrait, StreamTrait},
    cpal::{SampleFormat, StreamConfig},
    ringbuf::HeapRb,
    std::sync::{Arc, Mutex},
    tokio::sync::mpsc,
};

#[derive(Parser)]
#[command(name = "live_multimodal_chat")]
#[command(about = "A live multimodal chat example using Gemini Live API")]
struct Args {
    /// Enable verbose debug output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize rustls crypto provider
    let _ = rustls::crypto::ring::default_provider().install_default();

    let args = Args::parse();

    let api_key = std::env::var("GEMINI_API_KEY")
        .or_else(|_| std::env::var("GOOGLE_AI_API_KEY"))
        .expect("GEMINI_API_KEY or GOOGLE_AI_API_KEY environment variable must be set");

    println!("🚀 Starting Gemini Live Multimodal Chat");
    println!("Features enabled:");
    #[cfg(feature = "audio")]
    println!("  ✅ Audio input (microphone)");
    #[cfg(not(feature = "audio"))]
    println!("  ❌ Audio input (disabled - use --features audio)");

    #[cfg(feature = "audio-output")]
    println!("  ✅ Audio output (speakers)");
    #[cfg(not(feature = "audio-output"))]
    println!("  ❌ Audio output (disabled - use --features audio-output)");

    #[cfg(feature = "video")]
    println!("  ✅ Video input (camera)");
    #[cfg(not(feature = "video"))]
    println!("  ❌ Video input (disabled - use --features video)");

    // List available devices
    #[cfg(feature = "audio")]
    {
        println!("\n🎤 Available audio input devices:");
        match AudioRecorder::list_input_devices() {
            Ok(devices) => {
                for (i, device) in devices.iter().enumerate() {
                    println!("  {}: {}", i, device);
                }
                if devices.is_empty() {
                    println!("  No audio input devices found");
                }
            }
            Err(e) => println!("  Error listing audio devices: {}", e),
        }
    }

    #[cfg(feature = "video")]
    {
        println!("\n📹 Available cameras:");
        match VideoCapturer::list_cameras() {
            Ok(cameras) => {
                for camera in cameras {
                    println!("  {}", camera);
                }
            }
            Err(e) => println!("  Error listing cameras: {}", e),
        }
    }

    let gemini = Gemini::builder().api_key(api_key).build();

    // Define SpeechConfig (similar to Python example)
    let speech_config = gemini_ox::generate_content::SpeechConfig {
        voice_config: Some(gemini_ox::generate_content::VoiceConfig {
            prebuilt_voice_config: Some(gemini_ox::generate_content::PrebuiltVoiceConfig {
                voice_name: Some("Zephyr".to_string()),
            }),
        }),
        ..Default::default()
    };

    // Configure for audio response - speech_config goes inside generation_config
    let generation_config = GenerationConfig {
        response_modalities: Some(vec!["AUDIO".to_string()]),
        speech_config: Some(speech_config),
        ..Default::default()
    };

    // Configure realtime input with disabled automatic voice detection to prevent echo
    let realtime_input_config = gemini_ox::live::request_configs::RealtimeInputConfig {
        media_chunks: None,
        automatic_activity_detection: Some(
            gemini_ox::live::request_configs::AutomaticActivityDetection {
                disabled: Some(true),
                start_of_speech_sensitivity: None,
                prefix_padding_ms: None,
                end_of_speech_sensitivity: None,
                silence_duration_ms: None,
            },
        ),
        activity_handling: None,
        turn_coverage: None,
    };

    println!("\n🔌 Connecting to Gemini Live API...");
    let mut session = gemini
        .live_session()
        .model(Model::Gemini25FlashPreviewNativeAudioDialog)
        .generation_config(generation_config)
        .realtime_input_config(realtime_input_config)
        .build()
        .connect()
        .await?;

    println!("✅ Connected! Starting multimodal session...");

    // Setup audio output for server responses
    #[cfg(feature = "audio-output")]
    let audio_output = setup_audio_output(args.verbose).ok();
    #[cfg(not(feature = "audio-output"))]
    let audio_output: Option<()> = None;

    // Start audio input streaming if available
    #[cfg(feature = "audio")]
    let audio_input = {
        match AudioRecorder::start_capturing() {
            Ok((recorder, receiver)) => {
                println!("🎤 Audio input started");
                Some((recorder, receiver))
            }
            Err(e) => {
                println!("⚠️  Failed to start audio input: {}", e);
                None
            }
        }
    };
    #[cfg(not(feature = "audio"))]
    let _audio_input: Option<()> = None;

    // Start video input if available
    #[cfg(feature = "video")]
    let _video_input = {
        match VideoCapturer::start_capturing_default() {
            Ok((capturer, receiver)) => {
                println!("📹 Video input started (640x480, 1 FPS)");
                Some((capturer, receiver))
            }
            Err(e) => {
                println!("⚠️  Failed to start video input: {}", e);
                None
            }
        }
    };

    println!("📝 Type messages and press Enter. Type 'quit' to exit.");
    println!("💡 Tips:");
    #[cfg(feature = "audio")]
    println!("  - Audio input is streaming continuously");
    #[cfg(feature = "audio-output")]
    println!("  - Audio output is enabled for speech responses");
    #[cfg(feature = "video")]
    println!("  - Camera frames are being sent every second");
    println!("  - The AI can respond with both text and audio");
    println!("  - Type 'hello' to test text responses");
    println!("  - Type 'status' to check audio chunk count");

    // Setup audio communication channel (used by audio and video features)
    #[cfg(any(feature = "audio", feature = "video"))]
    let (audio_tx, mut audio_rx) =
        tokio::sync::mpsc::unbounded_channel::<gemini_ox::content::Blob>();

    // Setup speech activity detection channel (used by audio feature)
    #[cfg(feature = "audio")]
    let (speech_activity_tx, mut speech_activity_rx) =
        tokio::sync::mpsc::unbounded_channel::<bool>(); // true = speech started, false = speech ended

    // Setup stdin reader
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    // Show initial prompt
    print!("\n> ");
    io::stdout().flush()?;

    // Start streaming tasks
    #[cfg(feature = "audio")]
    let _audio_streaming_task = if let Some((_, mut audio_receiver)) = audio_input {
        println!("🎤 Starting audio streaming task...");
        let audio_tx_clone = audio_tx.clone();
        let speech_activity_tx_clone = speech_activity_tx.clone();
        let verbose = args.verbose;
        println!("Verbose: {verbose}");
        Some(tokio::spawn(async move {
            let mut chunk_count = 0;
            let mut is_speaking = false;
            let mut silence_chunks = 0;
            const SPEECH_THRESHOLD: f64 = 1000.0; // RMS threshold for speech
            const SILENCE_CHUNKS_REQUIRED: usize = 20; // ~500ms of silence at 25fps

            while let Some(audio_chunk) = audio_receiver.recv().await {
                chunk_count += 1;

                if verbose {
                    if chunk_count == 1 {
                        let data_len = audio_chunk.data.len();
                        println!(
                            "🎤 DEBUG: First audio chunk - mime_type={:?}, data_len={}",
                            audio_chunk.mime_type, data_len
                        );
                    }
                    if chunk_count % 10 == 0 {
                        // More frequent logging to see if we're getting audio
                        println!("🎤 DEBUG: Received {} audio chunks so far", chunk_count);
                    }
                }

                // Check audio level and detect speech activity
                let data = &audio_chunk.data;
                if let Ok(decoded) =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                {
                    // Calculate RMS level to see if there's actually audio
                    let mut sum_squares = 0.0_f64;
                    let samples = decoded.chunks_exact(2).count();

                    for chunk_bytes in decoded.chunks_exact(2) {
                        let sample = i16::from_le_bytes([chunk_bytes[0], chunk_bytes[1]]) as f64;
                        sum_squares += sample * sample;
                    }

                    if samples > 0 {
                        let rms = (sum_squares / samples as f64).sqrt();
                        if verbose && chunk_count % 50 == 0 {
                            println!(
                                "🔊 DEBUG: Audio level RMS: {:.1} (samples: {})",
                                rms, samples
                            );
                        }

                        // Speech activity detection
                        if rms > SPEECH_THRESHOLD {
                            silence_chunks = 0;
                            if !is_speaking {
                                is_speaking = true;
                                let _ = speech_activity_tx_clone.send(true);
                                if verbose {
                                    println!("🗣️  DEBUG: Speech started! RMS: {:.1}", rms);
                                }
                            }
                        } else {
                            if is_speaking {
                                silence_chunks += 1;
                                if silence_chunks >= SILENCE_CHUNKS_REQUIRED {
                                    is_speaking = false;
                                    silence_chunks = 0;
                                    let _ = speech_activity_tx_clone.send(false);
                                    if verbose {
                                        println!("🤫 DEBUG: Speech ended - silence detected");
                                    }
                                }
                            }
                        }
                    }
                }

                // Send audio chunk to main loop via channel
                if let Err(_) = audio_tx_clone.send(audio_chunk) {
                    if verbose {
                        println!(
                            "❌ DEBUG: Failed to send audio chunk #{} to main loop",
                            chunk_count
                        );
                    }
                    break;
                }
            }
            if verbose {
                println!("🎤 DEBUG: Audio streaming task ended");
            }
        }))
    } else {
        println!("⚠️  Audio streaming task not started - no audio input available");
        None
    };

    #[cfg(feature = "video")]
    let _video_streaming_task = if let Some((_, mut video_receiver)) = _video_input {
        let video_tx = audio_tx.clone();
        let verbose = args.verbose;
        Some(tokio::spawn(async move {
            let mut frame_count = 0;
            while let Some(video_chunk) = video_receiver.recv().await {
                frame_count += 1;
                if verbose && frame_count % 10 == 0 {
                    println!("📹 DEBUG: Received {} video frames so far", frame_count);
                }

                // Send video chunk to main loop via channel
                if let Err(_) = video_tx.send(video_chunk) {
                    if verbose {
                        println!(
                            "❌ DEBUG: Failed to send video chunk #{} to main loop",
                            frame_count
                        );
                    }
                    break;
                }
            }
            if verbose {
                println!("📹 DEBUG: Video streaming task ended");
            }
        }))
    } else {
        None
    };

    #[cfg(any(feature = "audio", feature = "video"))]
    let mut audio_chunk_count = 0;
    #[cfg(any(feature = "audio", feature = "video"))]
    let mut last_audio_time = std::time::Instant::now();

    // Main event loop - simplified approach
    #[cfg(any(feature = "audio", feature = "video", feature = "audio-output"))]
    {
        loop {
            tokio::select! {
                // Handle speech activity detection
                speech_activity = speech_activity_rx.recv(), if cfg!(feature = "audio") => {
                    if let Some(is_speech_active) = speech_activity {
                        if is_speech_active {
                            if args.verbose {
                                println!("🎯 DEBUG: Starting turn - speech detected");
                            }
                        } else {
                            // Send turn complete when speech ends
                            if args.verbose {
                                println!("🎯 DEBUG: Ending turn - speech finished");
                            }
                            let content = Content::new(
                                Role::User,
                                vec![""]
                            );
                            let payload = gemini_ox::live::message_types::ClientContentPayload {
                                turns: vec![content],
                                turn_complete: Some(true),
                            };
                            if let Err(e) = session.send_client_content(payload).await {
                                eprintln!("❌ Error sending turn complete: {}", e);
                            } else if args.verbose {
                                println!("✅ DEBUG: Turn complete sent");
                            }
                        }
                    }
                }

                // Handle audio chunks from microphone
                audio_chunk = audio_rx.recv(), if cfg!(any(feature = "audio", feature = "video")) => {
                    if let Some(chunk) = audio_chunk {
                        audio_chunk_count += 1;
                        last_audio_time = std::time::Instant::now();

                        // Check if it's audio or video chunk based on mime type
                        if chunk.mime_type == "audio/pcm;rate=16000" {
                            #[cfg(feature = "audio")]
                            {
                                if let Err(e) = send_audio_chunk(&mut session, chunk, args.verbose).await {
                                    eprintln!("❌ Error sending audio chunk: {}", e);
                                } else if args.verbose && audio_chunk_count % 100 == 0 {
                                    println!("🔍 DEBUG: Sent {} audio chunks so far", audio_chunk_count);
                                }
                            }
                        } else if chunk.mime_type == "image/jpeg" {
                            #[cfg(feature = "video")]
                            {
                                if let Err(e) = send_video_chunk(&mut session, chunk).await {
                                    eprintln!("❌ Error sending video chunk: {}", e);
                                }
                            }
                        }
                    }
                }

                // Handle responses from the API
                response = session.receive() => {
                    match response {
                        Some(Ok(chunk)) => {
                            if args.verbose {
                                println!("📥 DEBUG: Received response chunk: {:?}", std::mem::discriminant(&chunk));
                            }
                            if let Err(e) = handle_response_chunk(chunk, &audio_output, args.verbose).await {
                                eprintln!("❌ Error handling response: {}", e);
                            }
                        }
                        Some(Err(e)) => {
                            eprintln!("❌ Error receiving response: {}", e);
                            break;
                        }
                        None => {
                            if args.verbose {
                                println!("🔌 DEBUG: Connection closed by server");
                            }
                            break;
                        }
                    }
                }

                // Handle text input
                result = reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            let input = line.trim();
                            if input.eq_ignore_ascii_case("quit") {
                                break;
                            }
                            if input.eq_ignore_ascii_case("status") {
                                println!("📊 DEBUG: Status - sent {} audio chunks, last audio: {:?} ago",
                                       audio_chunk_count, last_audio_time.elapsed());
                                print!("> ");
                                io::stdout().flush()?;
                                line.clear();
                                continue;
                            }
                            if !input.is_empty() {
                                send_text_message(&mut session, input, args.verbose).await?;
                                if args.verbose {
                                    println!("⏳ DEBUG: Text message sent, waiting for response...");
                                }
                            }
                            line.clear();
                            print!("\n> ");
                            io::stdout().flush()?;
                        }
                        Err(e) => {
                            eprintln!("Error reading input: {}", e);
                            break;
                        }
                    }
                }

                // Timeout check - if no audio for a while, suggest testing
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(15)) => {
                    if last_audio_time.elapsed() > tokio::time::Duration::from_secs(15) {
                        println!("⏰ DEBUG: No recent audio activity. Try:");
                        println!("  - Type 'hello' to test text responses");
                        println!("  - Speak into your microphone");
                        println!("  - Type 'status' to check audio stats");
                        print!("> ");
                        let _ = io::stdout().flush();
                    }
                }
            }
        }
    }

    // Text-only loop when no media features are enabled
    #[cfg(not(any(feature = "audio", feature = "video", feature = "audio-output")))]
    {
        loop {
            tokio::select! {
                // Handle responses from the API
                response = session.receive() => {
                    match response {
                        Some(Ok(chunk)) => {
                            if args.verbose {
                                println!("📥 DEBUG: Received response chunk: {:?}", std::mem::discriminant(&chunk));
                            }
                            if let Err(e) = handle_response_chunk(chunk, &audio_output, args.verbose).await {
                                eprintln!("❌ Error handling response: {}", e);
                            }
                        }
                        Some(Err(e)) => {
                            eprintln!("❌ Error receiving response: {}", e);
                            break;
                        }
                        None => {
                            if args.verbose {
                                println!("🔌 DEBUG: Connection closed by server");
                            }
                            break;
                        }
                    }
                }

                // Handle text input
                result = reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => break, // EOF
                        Ok(_) => {
                            let input = line.trim();
                            if input.eq_ignore_ascii_case("quit") {
                                break;
                            }
                            if input.eq_ignore_ascii_case("status") {
                                println!("📊 DEBUG: Status - audio/video features disabled");
                                print!("> ");
                                io::stdout().flush()?;
                                line.clear();
                                continue;
                            }
                            if !input.is_empty() {
                                send_text_message(&mut session, input, args.verbose).await?;
                                if args.verbose {
                                    println!("⏳ DEBUG: Text message sent, waiting for response...");
                                }
                            }
                            line.clear();
                            print!("\n> ");
                            io::stdout().flush()?;
                        }
                        Err(e) => {
                            eprintln!("Error reading input: {}", e);
                            break;
                        }
                    }
                }
            }
        }
    }

    println!("\n👋 Closing session...");
    session.close().await?;
    println!("✅ Session closed. Goodbye!");

    Ok(())
}

async fn send_text_message(
    session: &mut ActiveLiveSession,
    text: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if verbose {
        println!("📤 DEBUG: Sending text message: '{}'", text);
    }

    let content = Content::new(Role::User, vec![text]);
    let payload = ClientContentPayload {
        turns: vec![content],
        turn_complete: Some(true),
    };

    session.send_client_content(payload).await?;
    Ok(())
}

#[cfg(feature = "audio")]
async fn send_audio_chunk(
    session: &mut ActiveLiveSession,
    chunk: gemini_ox::content::Blob,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use gemini_ox::live::message_types::RealtimeInputPayload;

    // if verbose {
    //     println!("📤 DEBUG: Sending audio chunk (mime_type: {:?}, data length: {})",
    //              chunk.mime_type, chunk.data.as_ref().map_or(0, |d| d.len()));
    // }

    let payload = RealtimeInputPayload {
        media_chunks: Some(vec![chunk]),
    };

    session.send_realtime_input(payload).await?;
    Ok(())
}

#[cfg(feature = "video")]
async fn send_video_chunk(
    session: &mut ActiveLiveSession,
    chunk: gemini_ox::content::Blob,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use gemini_ox::live::message_types::RealtimeInputPayload;

    println!(
        "📤 DEBUG: Preparing to send video chunk as RealtimeInput (mime_type: {:?}, data length: {})",
        chunk.mime_type,
        chunk.data.len()
    );

    let payload = RealtimeInputPayload {
        media_chunks: Some(vec![chunk]),
    };

    match session.send_realtime_input(payload).await {
        Ok(()) => {
            println!("✅ DEBUG: Video frame sent successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ DEBUG: Failed to send video chunk: {}", e);
            Err(e.into())
        }
    }
}

#[cfg(feature = "audio-output")]
async fn handle_response_chunk(
    chunk: LiveApiResponseChunk,
    audio_output: &Option<AudioOutputHandler>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match chunk {
        LiveApiResponseChunk::ModelTurn { server_content } => {
            if verbose {
                println!("🤖 DEBUG: ModelTurn received");
            }
            if let Some(parts) = server_content.model_turn.parts {
                if verbose {
                    println!("🤖 DEBUG: Processing {} parts", parts.len());
                }
                for (i, part) in parts.iter().enumerate() {
                    if let Some(text) = &part.text {
                        println!("🤖 {}", text);
                    }

                    if let Some(inline_data) = &part.inline_data {
                        if verbose {
                            println!(
                                "🤖 DEBUG: Inline data part {} - mime_type: {:?}, data length: {}",
                                i,
                                inline_data.mime_type,
                                inline_data.data.len()
                            );
                        }

                        if inline_data.mime_type == "audio/pcm;rate=24000" {
                            let audio_data = &inline_data.data;
                            #[cfg(feature = "audio-output")]
                            if let Some(output) = audio_output {
                                play_audio_data(output, audio_data)?;

                                if verbose {
                                    println!(
                                        "🔊 DEBUG: Audio chunk queued (length: {})",
                                        audio_data.len()
                                    );
                                }
                            }
                            #[cfg(feature = "audio-output")]
                            if audio_output.is_none() && verbose {
                                println!("⚠️  DEBUG: Audio output not available");
                            }
                            #[cfg(not(feature = "audio-output"))]
                            if verbose {
                                println!(
                                    "⚠️  DEBUG: Audio output disabled (missing audio-output feature)"
                                );
                            }
                        }
                    }
                }
            } else if verbose {
                println!("🤖 DEBUG: ModelTurn has no parts");
            }
        }
        LiveApiResponseChunk::TurnComplete { .. } => {
            if verbose {
                println!("✅ DEBUG: Turn complete - clearing audio queue");
            }
            // Clear audio queue when turn completes to prevent overlapping responses
            #[cfg(feature = "audio-output")]
            if let Some(output) = audio_output {
                let _ = output.clear_signal.send(());
            }
        }
        LiveApiResponseChunk::GenerationComplete { .. } => {
            if verbose {
                println!("🏁 DEBUG: Generation complete - clearing audio queue");
            }
            // Clear audio queue when generation completes
            #[cfg(feature = "audio-output")]
            if let Some(output) = audio_output {
                let _ = output.clear_signal.send(());
            }
        }
        LiveApiResponseChunk::Interrupted { .. } => {
            if verbose {
                println!("⚠️  DEBUG: Interrupted - clearing audio");
            }
            // Clear audio queue on interruption - this is key to preventing overlaps
            #[cfg(feature = "audio-output")]
            if let Some(output) = audio_output {
                let _ = output.clear_signal.send(());
            }
        }
        LiveApiResponseChunk::SetupComplete { .. } => {
            if verbose {
                println!("🔧 DEBUG: Setup complete");
            }
        }
        LiveApiResponseChunk::ToolCall { .. } => {
            if verbose {
                println!("🔧 DEBUG: Tool call received");
            }
        }
        LiveApiResponseChunk::ToolCallCancellation { .. } => {
            if verbose {
                println!("🚫 DEBUG: Tool call cancelled");
            }
        }
    }
    Ok(())
}

// Audio output handling for receiving server audio responses
#[cfg(feature = "audio-output")]
struct AudioOutputHandler {
    _stream: cpal::Stream,
    audio_sender: mpsc::UnboundedSender<Vec<i16>>,
    clear_signal: mpsc::UnboundedSender<()>,
}

#[cfg(feature = "audio-output")]
fn setup_audio_output(
    verbose: bool,
) -> Result<AudioOutputHandler, Box<dyn std::error::Error + Send + Sync>> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("No default output device available")?;

    // Find a suitable output configuration: 24000 Hz, 1 channel, i16
    let supported_config = device
        .supported_output_configs()?
        .find(|config| {
            config.sample_format() == SampleFormat::I16
                && config.channels() == 1
                && config.min_sample_rate() <= cpal::SampleRate(24000)
                && config.max_sample_rate() >= cpal::SampleRate(24000)
        })
        .ok_or("No suitable i16 24kHz mono config found for output device")?
        .with_sample_rate(cpal::SampleRate(24000));

    let stream_config: StreamConfig = supported_config.config();

    // Ring buffer for audio samples (i16). Capacity for ~2 seconds of audio.
    let ring_buffer_capacity = 24000 * 1 * 2; // sample_rate * channels * seconds
    let (producer, consumer) = HeapRb::<i16>::new(ring_buffer_capacity).split();

    let producer = Arc::new(Mutex::new(producer));
    let consumer = Arc::new(Mutex::new(consumer));

    // Channel for queuing audio chunks
    let (audio_sender, mut audio_receiver) = mpsc::unbounded_channel::<Vec<i16>>();

    // Channel for clearing audio queue on interruption
    let (clear_signal, mut clear_receiver) = mpsc::unbounded_channel::<()>();

    // Spawn task to handle audio queue sequentially
    let producer_for_task = Arc::clone(&producer);
    let consumer_for_task = Arc::clone(&consumer);
    tokio::spawn(async move {
        let mut is_skipping_until: Option<tokio::time::Instant> = None;

        loop {
            tokio::select! {
                // Handle clear signal (interruption/turn complete)
                _ = clear_receiver.recv() => {
                    // Drain any pending audio chunks from the queue
                    while audio_receiver.try_recv().is_ok() {
                        // Discard pending chunks
                    }

                    // Clear the ring buffer by draining all samples from consumer
                    {
                        let mut consumer = consumer_for_task.lock().unwrap();
                        // Drain all pending samples from the ring buffer
                        while consumer.pop().is_some() {}
                    }

                    if verbose {
                        println!("🔇 Audio playback cleared - queue and buffer emptied");
                    }
                    is_skipping_until = Some(tokio::time::Instant::now() + tokio::time::Duration::from_millis(50));
                }

                // Handle audio chunks
                audio_chunk = audio_receiver.recv() => {
                    if let Some(audio_chunk) = audio_chunk {
                        // Skip audio if we're in interruption/settling period
                        if let Some(skip_until_time) = is_skipping_until {
                            if tokio::time::Instant::now() < skip_until_time {
                                if verbose {
                                    println!("DEBUG: 🔇 Skipping audio chunk during settling period.");
                                }
                                continue;
                            } else {
                                if verbose {
                                    println!("DEBUG: 🔇 Settling period over. Resuming audio playback.");
                                }
                                is_skipping_until = None; // Reset skip state
                            }
                        }

                        // Wait for buffer to have enough space
                        loop {
                            let buffer_len = {
                                let producer = producer_for_task.lock().unwrap();
                                producer.len()
                            };

                            // If buffer has space for this chunk, proceed
                            if buffer_len + audio_chunk.len() <= ring_buffer_capacity - 1000 {
                                break;
                            }

                            // Wait a bit for buffer to drain
                            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                        }

                        // Add samples to ring buffer with reduced volume to prevent feedback
                        {
                            let mut producer = producer_for_task.lock().unwrap();
                            for &sample in &audio_chunk {
                                // Reduce volume by 50% to prevent audio feedback
                                let reduced_sample = (sample as f32 * 0.5) as i16;
                                if producer.push(reduced_sample).is_err() {
                                    // Buffer full, should not happen due to check above
                                    break;
                                }
                            }
                        }
                    } else {
                        // Audio receiver closed
                        break;
                    }
                }
            }
        }
    });

    let consumer_for_stream = Arc::clone(&consumer);
    let stream = device.build_output_stream(
        &stream_config,
        move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
            let mut consumer = consumer_for_stream.lock().unwrap();
            let written = consumer.pop_slice(data);
            // Zero out the rest of the buffer if not enough samples
            for sample_ref in data.iter_mut().skip(written) {
                *sample_ref = 0;
            }
        },
        |err| eprintln!("CPAL stream error: {}", err),
        None,
    )?;

    stream.play()?;

    Ok(AudioOutputHandler {
        _stream: stream,
        audio_sender,
        clear_signal,
    })
}

#[cfg(feature = "audio-output")]
fn play_audio_data(
    output: &AudioOutputHandler,
    base64_data: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let pcm_bytes = BASE64_STANDARD.decode(base64_data)?;

    // Convert bytes to i16 samples
    let mut samples = Vec::new();
    for chunk_bytes in pcm_bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk_bytes[0], chunk_bytes[1]]);
        samples.push(sample);
    }

    // Send to audio queue for streaming playback - no buffering, immediate streaming
    if let Err(_) = output.audio_sender.send(samples) {
        eprintln!("Audio output channel closed");
        return Ok(());
    }

    Ok(())
}

#[cfg(not(feature = "audio-output"))]
async fn handle_response_chunk(
    chunk: LiveApiResponseChunk,
    _audio_output: &Option<()>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match chunk {
        LiveApiResponseChunk::ModelTurn { server_content } => {
            if verbose {
                println!("🤖 DEBUG: ModelTurn received");
            }
            if let Some(parts) = server_content.model_turn.parts {
                if verbose {
                    println!("🤖 DEBUG: Processing {} parts", parts.len());
                }
                for (i, part) in parts.iter().enumerate() {
                    if let Some(text) = &part.text {
                        println!("🤖 {}", text);
                    }

                    if let Some(inline_data) = &part.inline_data {
                        if verbose {
                            println!(
                                "🤖 DEBUG: Inline data part {} - mime_type: {:?}, data length: {}",
                                i,
                                inline_data.mime_type,
                                inline_data.data.len()
                            );
                        }

                        if inline_data.mime_type == "audio/pcm;rate=24000" {
                            if verbose {
                                println!(
                                    "⚠️  DEBUG: Audio output disabled (missing audio-output feature)"
                                );
                            }
                        }
                    }
                }
            } else if verbose {
                println!("🤖 DEBUG: ModelTurn has no parts");
            }
        }
        LiveApiResponseChunk::TurnComplete { .. } => {
            if verbose {
                println!("✅ DEBUG: Turn complete");
            }
        }
        LiveApiResponseChunk::GenerationComplete { .. } => {
            if verbose {
                println!("🏁 DEBUG: Generation complete");
            }
        }
        LiveApiResponseChunk::Interrupted { .. } => {
            if verbose {
                println!("⚠️  DEBUG: Interrupted");
            }
        }
        LiveApiResponseChunk::SetupComplete { .. } => {
            if verbose {
                println!("🔧 DEBUG: Setup complete");
            }
        }
        LiveApiResponseChunk::ToolCall { .. } => {
            if verbose {
                println!("🔧 DEBUG: Tool call received");
            }
        }
        LiveApiResponseChunk::ToolCallCancellation { .. } => {
            if verbose {
                println!("🚫 DEBUG: Tool call cancelled");
            }
        }
    }
    Ok(())
}
