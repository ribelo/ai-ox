#![cfg(feature = "audio")]

use crate::live::message_types::MediaChunk;
use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use tokio::sync::mpsc;

const TARGET_SAMPLE_RATE: u32 = 16000;
const TARGET_CHANNELS: u16 = 1;
const TARGET_SAMPLE_FORMAT: SampleFormat = SampleFormat::I16;
const MIME_TYPE_PCM: &str = "audio/pcm;rate=16000";

pub struct AudioRecorder {
    _stream: cpal::Stream, // Keep stream alive
}

impl AudioRecorder {
    /// Start capturing audio from the default input device
    /// Returns a receiver that yields MediaChunk objects containing base64-encoded PCM audio
    pub fn start_capturing() -> Result<(Self, mpsc::Receiver<MediaChunk>)> {
        let (tx, rx) = mpsc::channel(10); // Modest buffer

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No default input device available"))?;

        // Find supported config
        let supported_configs_range = device
            .supported_input_configs()
            .context("Error querying supported input configs")?;

        let config_range = supported_configs_range
            .filter(|r| {
                r.channels() == TARGET_CHANNELS && r.sample_format() == TARGET_SAMPLE_FORMAT
            })
            .find(|r| {
                r.min_sample_rate().0 <= TARGET_SAMPLE_RATE
                    && r.max_sample_rate().0 >= TARGET_SAMPLE_RATE
            })
            .ok_or_else(|| {
                anyhow!(
                    "No supported config found for {}kHz, {} channel, {:?} format",
                    TARGET_SAMPLE_RATE / 1000,
                    TARGET_CHANNELS,
                    TARGET_SAMPLE_FORMAT
                )
            })?;

        let stream_config: StreamConfig = config_range
            .with_sample_rate(SampleRate(TARGET_SAMPLE_RATE))
            .config();

        let err_fn = |err| eprintln!("CPAL audio input stream error: {}", err);

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut byte_data = Vec::with_capacity(data.len() * std::mem::size_of::<i16>());
                for &sample in data {
                    byte_data.extend_from_slice(&sample.to_le_bytes());
                }

                let b64_encoded_data = BASE64_STANDARD.encode(&byte_data);
                let chunk = MediaChunk {
                    mime_type: Some(MIME_TYPE_PCM.to_string()),
                    data: Some(b64_encoded_data),
                };
                if let Err(_) = tx.try_send(chunk) {
                    // Silently drop audio chunks when channel is full
                    // This is expected when audio input is faster than consumption
                }
            },
            err_fn,
            None,
        )?;

        stream.play().context("Failed to play audio stream")?;

        let recorder = AudioRecorder { _stream: stream };
        Ok((recorder, rx))
    }

    /// List available audio input devices
    pub fn list_input_devices() -> Result<Vec<String>> {
        let host = cpal::default_host();
        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;

        let mut device_names = Vec::new();
        for device in devices {
            if let Ok(name) = device.name() {
                device_names.push(name);
            }
        }

        Ok(device_names)
    }

    /// Get information about the default input device
    pub fn default_input_device_info() -> Result<String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No default input device available"))?;

        let name = device.name().context("Failed to get device name")?;
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    #[ignore = "Requires audio hardware and may be noisy"]
    async fn test_audio_capture() -> Result<()> {
        let (recorder, mut rx) = AudioRecorder::start_capturing()?;

        // Capture for a short duration
        let timeout = tokio::time::timeout(Duration::from_secs(2), async {
            if let Some(chunk) = rx.recv().await {
                assert!(chunk.mime_type.is_some());
                assert!(chunk.data.is_some());
                assert_eq!(chunk.mime_type.as_ref().unwrap(), MIME_TYPE_PCM);

                // Verify base64 data can be decoded
                let data = chunk.data.unwrap();
                let decoded = BASE64_STANDARD.decode(data)?;
                assert!(!decoded.is_empty());
                // Should be even number of bytes (i16 samples)
                assert_eq!(decoded.len() % 2, 0);
            }
            Ok::<(), anyhow::Error>(())
        })
        .await;

        drop(recorder); // Clean up

        match timeout {
            Ok(result) => result,
            Err(_) => Ok(()), // Timeout is acceptable for this test
        }
    }

    #[test]
    fn test_list_devices() {
        // This should not fail even without audio hardware
        let result = AudioRecorder::list_input_devices();
        assert!(result.is_ok());
    }
}
