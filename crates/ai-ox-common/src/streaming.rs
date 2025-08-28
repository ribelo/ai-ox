use serde::Deserialize;
use futures_util::StreamExt;
use crate::error::CommonRequestError;

/// Server-Sent Events parser for streaming responses
pub struct SseParser {
    /// The underlying byte stream from the response.
    byte_stream: std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    /// A buffer to store partial event data.
    buffer: Vec<u8>,
}

impl SseParser {
    #[must_use]
    pub fn new(response: reqwest::Response) -> Self {
        Self {
            byte_stream: Box::pin(response.bytes_stream()),
            buffer: Vec::new(),
        }
    }

    /// Get the next parsed event from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if reading from the stream fails or if the event data is invalid.
    pub async fn next_event<T: for<'de> Deserialize<'de>>(&mut self) -> Result<Option<T>, CommonRequestError> {
        loop {
            // Try to process events from current buffer
            if let Some(event) = self.try_parse_event_from_buffer::<T>()? {
                return Ok(Some(event));
            }

            // Read more data if no complete event in buffer
            if let Some(chunk_result) = self.byte_stream.next().await {
                let chunk = chunk_result?;
                self.buffer.extend_from_slice(&chunk);
            } else {
                // Stream ended, process any remaining data
                if !self.buffer.is_empty() {
                    if let Some(event) = self.try_parse_final_event::<T>()? {
                        return Ok(Some(event));
                    }
                }
                return Ok(None);
            }
        }
    }

    /// Try to parse an event from the current buffer
    fn try_parse_event_from_buffer<T: for<'de> Deserialize<'de>>(&mut self) -> Result<Option<T>, CommonRequestError> {
        // Look for complete lines ending with \n
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line_bytes = self.buffer.drain(..=pos).collect::<Vec<u8>>();
            let line = String::from_utf8(line_bytes)?;
            
            if let Some(event) = Self::parse_sse_line::<T>(&line)? {
                return Ok(Some(event));
            }
        }
        
        Ok(None)
    }

    /// Try to parse any remaining data as final event
    fn try_parse_final_event<T: for<'de> Deserialize<'de>>(&mut self) -> Result<Option<T>, CommonRequestError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let line = String::from_utf8(std::mem::take(&mut self.buffer))?;
        Self::parse_sse_line::<T>(&line)
    }

    /// Parse a single SSE line into an event
    fn parse_sse_line<T: for<'de> Deserialize<'de>>(line: &str) -> Result<Option<T>, CommonRequestError> {
        let line = line.trim();
        
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with(':') {
            return Ok(None);
        }

        // Parse SSE format: "data: <json>"
        if line.starts_with("data: ") {
            let json_data = line.trim_start_matches("data: ").trim();
            
            // Skip [DONE] markers and empty data
            if json_data.is_empty() || json_data == "[DONE]" {
                return Ok(None);
            }

            // Parse JSON
            let event: T = serde_json::from_str(json_data)
                .map_err(|e| CommonRequestError::InvalidEventData(format!("JSON parse error: {e}")))?;
            
            return Ok(Some(event));
        }

        // Handle other SSE fields (event, id, retry) - for now, ignore them
        Ok(None)
    }
}

/// Utility function to parse SSE events from a string chunk (for compatibility).
///
/// # Errors
///
/// Returns an error if any line contains invalid event data.
pub fn parse_sse_events<T: for<'de> Deserialize<'de>>(
    chunk: &str,
) -> Result<Vec<T>, CommonRequestError> {
    let mut events = Vec::new();
    
    for line in chunk.lines() {
        if let Some(event) = SseParser::parse_sse_line::<T>(line)? {
            events.push(event);
        }
    }
    
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn test_parse_sse_events_empty() {
        let result: Result<Vec<Value>, _> = parse_sse_events("");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_sse_events_done_message() {
        let sse_data = "data: [DONE]\n";
        let result: Result<Vec<Value>, _> = parse_sse_events(sse_data);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_sse_events_valid_json() {
        let sse_data = "data: {\"test\": \"value\"}\n";
        let result: Result<Vec<Value>, _> = parse_sse_events(sse_data);
        assert!(result.is_ok());
        let events = result.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["test"], "value");
    }

    #[test]
    fn test_parse_sse_events_invalid_json() {
        let sse_data = "data: {invalid json}\n";
        let result: Result<Vec<Value>, _> = parse_sse_events(sse_data);
        assert!(result.is_err());
    }
}