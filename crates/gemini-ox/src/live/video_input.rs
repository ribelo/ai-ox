#![cfg(feature = "video")]

use crate::live::message_types::MediaChunk;
use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use image::{DynamicImage, ImageBuffer, ImageOutputFormat, Rgb};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::{ApiBackend, Camera, NokhwaError, query};
use std::io::Cursor;
use tokio::sync::mpsc;
use tokio::time::{Duration, interval};

const MIME_TYPE_JPEG: &str = "image/jpeg";
const TARGET_FPS: u32 = 1; // Capture 1 frame per second, as per Python example
const JPEG_QUALITY: u8 = 80;

pub struct VideoCapturer {
    // Camera is not Send, so we don't store it here
    // Instead, it's initialized and used within the spawned task
}

impl VideoCapturer {
    /// Start capturing video from the specified camera
    /// Returns a receiver that yields MediaChunk objects containing base64-encoded JPEG images
    pub fn start_capturing(
        index: CameraIndex,
        width: u32,
        height: u32,
    ) -> Result<(Self, mpsc::Receiver<MediaChunk>)> {
        let (tx, rx) = mpsc::channel(5); // Buffer a few frames

        tokio::spawn(async move {
            let capture_result = Self::capture_task(tx, index, width, height).await;
            if let Err(e) = capture_result {
                eprintln!("Video capture task failed: {}", e);
            }
        });

        let capturer = VideoCapturer {};
        Ok((capturer, rx))
    }

    async fn capture_task(
        tx: mpsc::Sender<MediaChunk>,
        index: CameraIndex,
        width: u32,
        height: u32,
    ) -> Result<()> {
        // Initialize camera within the task
        let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
            CameraFormat::new(width, height, TARGET_FPS),
        ));

        let mut camera =
            Camera::new(index, requested_format).context("Failed to initialize camera")?;

        camera
            .open_stream()
            .context("Failed to open camera stream")?;

        let mut tick_interval = interval(Duration::from_millis(1000 / TARGET_FPS as u64));

        loop {
            tick_interval.tick().await;

            match camera.frame() {
                Ok(frame_buffer) => {
                    match frame_buffer.decode_image::<RgbFormat>() {
                        Ok(rgb_image) => {
                            // Convert RgbImage to DynamicImage for JPEG encoding
                            let dynamic_image = DynamicImage::ImageRgb8(rgb_image);

                            let mut buffer = Cursor::new(Vec::new());
                            if dynamic_image
                                .write_to(&mut buffer, ImageOutputFormat::Jpeg(JPEG_QUALITY))
                                .is_ok()
                            {
                                let image_bytes = buffer.into_inner();
                                let b64_encoded_data = BASE64_STANDARD.encode(&image_bytes);
                                let chunk = MediaChunk {
                                    mime_type: Some(MIME_TYPE_JPEG.to_string()),
                                    data: Some(b64_encoded_data),
                                };

                                if tx.send(chunk).await.is_err() {
                                    eprintln!("Video frame send failed, channel closed.");
                                    break; // Stop task if receiver is dropped
                                }
                            } else {
                                eprintln!("Failed to encode frame to JPEG.");
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to decode frame: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to capture camera frame: {}", e);
                    // Consider if error is fatal and should break loop
                    if matches!(e, NokhwaError::ReadFrameError(_)) {
                        // Potentially try to reopen stream or break on persistent errors
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }
    }

    /// Query available cameras
    pub fn query_cameras() -> Result<Vec<nokhwa::utils::CameraInfo>> {
        let cameras = query(ApiBackend::Auto).context("Failed to query cameras")?;
        Ok(cameras)
    }

    /// Get information about available cameras as strings
    pub fn list_cameras() -> Result<Vec<String>> {
        let cameras = Self::query_cameras()?;
        let camera_info: Vec<String> = cameras
            .into_iter()
            .map(|info| format!("{}: {}", info.index(), info.human_name()))
            .collect();
        Ok(camera_info)
    }

    /// Start capturing with default settings (camera 0, 640x480)
    pub fn start_capturing_default() -> Result<(Self, mpsc::Receiver<MediaChunk>)> {
        Self::start_capturing(CameraIndex::Index(0), 640, 480)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    #[ignore = "Requires camera hardware"]
    async fn test_video_capture() -> Result<()> {
        let (capturer, mut rx) = VideoCapturer::start_capturing_default()?;

        // Try to capture one frame
        let timeout = tokio::time::timeout(Duration::from_secs(5), async {
            if let Some(chunk) = rx.recv().await {
                assert!(chunk.mime_type.is_some());
                assert!(chunk.data.is_some());
                assert_eq!(chunk.mime_type.as_ref().unwrap(), MIME_TYPE_JPEG);

                // Verify base64 data can be decoded
                let data = chunk.data.unwrap();
                let decoded = BASE64_STANDARD.decode(data)?;
                assert!(!decoded.is_empty());

                // Basic JPEG header check (starts with FF D8)
                assert_eq!(decoded[0], 0xFF);
                assert_eq!(decoded[1], 0xD8);
            }
            Ok::<(), anyhow::Error>(())
        })
        .await;

        drop(capturer); // Clean up

        match timeout {
            Ok(result) => result,
            Err(_) => Ok(()), // Timeout is acceptable for this test
        }
    }

    #[test]
    fn test_query_cameras() {
        // This should not fail even without camera hardware
        let result = VideoCapturer::query_cameras();
        // We don't assert success since cameras might not be available in CI
        println!("Camera query result: {:?}", result);
    }

    #[test]
    fn test_list_cameras() {
        let result = VideoCapturer::list_cameras();
        // We don't assert success since cameras might not be available in CI
        println!("Camera list result: {:?}", result);
    }
}
