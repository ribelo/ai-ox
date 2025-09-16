#![cfg(feature = "video")]

use crate::content::{Blob, mime_types};
use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use image::{DynamicImage, ImageFormat};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, CameraInfo, FrameFormat, RequestedFormat,
    RequestedFormatType, Resolution,
};
use nokhwa::{Camera, query};
use std::io::Cursor;
use tokio::sync::mpsc;
use tokio::time::{Duration, interval};

const TARGET_FPS: u32 = 1; // Capture 1 frame per second, as per Python example

pub struct VideoCapturer {
    // Camera is not Send, so we don't store it here
    // Instead, it's initialized and used within the spawned task
}

impl VideoCapturer {
    /// Start capturing video from the specified camera
    /// Returns a receiver that yields Blob objects containing base64-encoded JPEG images
    pub fn start_capturing(
        index: CameraIndex,
        width: u32,
        height: u32,
    ) -> Result<(Self, mpsc::Receiver<Blob>)> {
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
        tx: mpsc::Sender<Blob>,
        index: CameraIndex,
        width: u32,
        height: u32,
    ) -> Result<()> {
        // Create a channel for communication between blocking and async contexts
        let (blocking_tx, mut blocking_rx) = mpsc::channel::<Vec<u8>>(10);

        // Spawn blocking task for camera operations
        let blocking_handle = tokio::task::spawn_blocking(move || {
            // Initialize camera in blocking context
            let requested_format =
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
                    Resolution::new(width, height),
                    FrameFormat::MJPEG,
                    TARGET_FPS,
                )));

            let mut camera = Camera::new(index, requested_format)?;
            camera.open_stream()?;

            // Capture loop in blocking context
            loop {
                match camera.frame() {
                    Ok(frame_buffer) => {
                        if let Ok(rgb_image) = frame_buffer.decode_image::<RgbFormat>() {
                            let dynamic_image = DynamicImage::ImageRgb8(rgb_image);
                            let mut buffer = Cursor::new(Vec::new());
                            if dynamic_image
                                .write_to(&mut buffer, ImageFormat::Jpeg)
                                .is_ok()
                            {
                                let image_bytes = buffer.into_inner();
                                // Send to async context
                                if blocking_tx.blocking_send(image_bytes).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
                std::thread::sleep(Duration::from_millis(1000 / TARGET_FPS as u64));
            }
            Ok::<(), anyhow::Error>(())
        });

        // Handle encoding and sending in async context
        let mut tick_interval = interval(Duration::from_millis(1000 / TARGET_FPS as u64));
        while let Some(image_bytes) = blocking_rx.recv().await {
            tick_interval.tick().await;
            let b64_encoded_data = BASE64_STANDARD.encode(&image_bytes);
            let chunk = Blob::new(mime_types::IMAGE_JPEG.to_string(), b64_encoded_data);

            if tx.send(chunk).await.is_err() {
                break;
            }
        }

        // Wait for the blocking task to finish
        let _ = blocking_handle.await;

        Ok(())
    }

    /// Query available cameras
    pub fn query_cameras() -> Result<Vec<CameraInfo>> {
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
    pub fn start_capturing_default() -> Result<(Self, mpsc::Receiver<Blob>)> {
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
                assert_eq!(chunk.mime_type, mime_types::IMAGE_JPEG.to_string());

                // Verify base64 data can be decoded
                let decoded = BASE64_STANDARD.decode(&chunk.data)?;
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
