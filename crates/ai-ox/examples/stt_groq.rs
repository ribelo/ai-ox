#[cfg(feature = "groq")]
use ai_ox::stt::{AudioSource, OutputFormat, SpeechToText, TranscriptionRequest, groq_stt};
use std::path::PathBuf;

#[cfg(feature = "groq")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create Groq STT provider using builder
    let stt = groq_stt("whisper-large-v3")?;

    // Load the test audio file
    let audio_path = PathBuf::from("resources/harvard.wav");
    println!("Transcribing audio file: {:?}", audio_path);

    // Create transcription request
    let request = TranscriptionRequest::builder()
        .audio(AudioSource::from_file(audio_path))
        .output_format(OutputFormat::Verbose)
        .build();

    // Transcribe the audio
    match stt.transcribe(request).await {
        Ok(response) => {
            println!("\n=== Transcription Results ===");
            println!("Text: {}", response.text);
            println!("Language: {:?}", response.language);
            println!("Duration: {:?}", response.duration);
            println!("Provider: {}", response.provider);
            println!("Model: {}", response.model);

            if response.has_segments() {
                println!("\n=== Segments ===");
                for (i, segment) in response.segments.iter().enumerate() {
                    println!(
                        "Segment {}: {} - {} | {}",
                        i + 1,
                        segment.start.as_secs_f32(),
                        segment.end.as_secs_f32(),
                        segment.text
                    );
                    if let Some(confidence) = segment.confidence {
                        println!("  Confidence: {:.2}", confidence);
                    }
                }
            }

            if response.has_words() {
                println!("\n=== Words ===");
                for word in response.words.iter().take(10) {
                    // Show first 10 words
                    println!(
                        "{}: {} - {}",
                        word.text,
                        word.start.as_secs_f32(),
                        word.end.as_secs_f32()
                    );
                }
                if response.words.len() > 10 {
                    println!("... and {} more words", response.words.len() - 10);
                }
            }

            println!("\n=== Usage Stats ===");
            println!("Audio duration: {:?}", response.usage.audio_duration);
            println!(
                "Characters transcribed: {}",
                response.usage.characters_transcribed
            );
            if let Some(processing_time) = response.usage.processing_time {
                println!("Processing time: {:?}", processing_time);
                if let Some(speed_ratio) = response.usage.speed_ratio() {
                    println!("Speed ratio: {:.2}x", speed_ratio);
                }
            }
        }
        Err(e) => {
            eprintln!("Transcription failed: {}", e);
        }
    }

    Ok(())
}

#[cfg(not(feature = "groq"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "This example requires the 'groq' feature. Run with: cargo run --example stt_groq --features groq"
    );
    Ok(())
}
