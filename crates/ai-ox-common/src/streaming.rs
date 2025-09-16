use crate::error::CommonRequestError;
use futures_util::StreamExt;
use serde::Deserialize;

/// Server-Sent Events parser for streaming responses
pub struct SseParser {
    byte_stream: std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>,
    >,
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseParser {
    pub fn new(response: reqwest::Response) -> Self {
        Self {
            byte_stream: Box::pin(response.bytes_stream()),
            buffer: Vec::new(),
            data_lines: Vec::new(),
        }
    }

    /// Get the next parsed event from the stream
    pub async fn next_event<T: for<'de> Deserialize<'de>>(
        &mut self,
    ) -> Result<Option<T>, CommonRequestError> {
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
    fn try_parse_event_from_buffer<T: for<'de> Deserialize<'de>>(
        &mut self,
    ) -> Result<Option<T>, CommonRequestError> {
        // Look for complete lines ending with \n
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line_bytes = self.buffer.drain(..=pos).collect::<Vec<u8>>();
            let line = String::from_utf8(line_bytes)?;

            if let Some(event) = self.process_line::<T>(&line)? {
                return Ok(Some(event));
            }
        }

        Ok(None)
    }

    /// Try to parse any remaining data as final event
    fn try_parse_final_event<T: for<'de> Deserialize<'de>>(
        &mut self,
    ) -> Result<Option<T>, CommonRequestError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let line = String::from_utf8(std::mem::take(&mut self.buffer))?;
        if let Some(event) = self.process_line::<T>(&line)? {
            return Ok(Some(event));
        }

        self.finalize_event::<T>()
    }

    fn process_line<T: for<'de> Deserialize<'de>>(
        &mut self,
        line: &str,
    ) -> Result<Option<T>, CommonRequestError> {
        let line = line.trim_end_matches(|c| c == '\n' || c == '\r');
        let trimmed = line.trim_end();

        if trimmed.is_empty() {
            return self.finalize_event();
        }

        if trimmed.starts_with(':') {
            return Ok(None);
        }

        if let Some(rest) = trimmed.strip_prefix("data:") {
            let data = rest.trim_start();

            if data == "[DONE]" {
                self.data_lines.clear();
                return Ok(None);
            }

            if !data.is_empty() {
                self.data_lines.push(data.to_string());
            }

            return Ok(None);
        }

        // Ignore other SSE fields (event, id, retry)
        Ok(None)
    }

    fn finalize_event<T: for<'de> Deserialize<'de>>(
        &mut self,
    ) -> Result<Option<T>, CommonRequestError> {
        if self.data_lines.is_empty() {
            return Ok(None);
        }

        let payload = self.data_lines.join("\n");
        self.data_lines.clear();

        if payload.is_empty() || payload == "[DONE]" {
            return Ok(None);
        }

        let event: T = serde_json::from_str(&payload).map_err(|e| {
            CommonRequestError::InvalidEventData(format!("JSON parse error: {}", e))
        })?;

        Ok(Some(event))
    }
}

/// Utility function to parse SSE events from a string chunk (for compatibility)
pub fn parse_sse_events<T: for<'de> Deserialize<'de>>(
    chunk: &str,
) -> Result<Vec<T>, CommonRequestError> {
    let mut events = Vec::new();
    let mut data_lines = Vec::new();

    for line in chunk.lines() {
        let line = line.trim_end_matches('\r');

        if line.is_empty() {
            if let Some(event) = finalize_data_lines::<T>(&mut data_lines)? {
                events.push(event);
            }
            continue;
        }

        if line.starts_with(':') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("data:") {
            let data = rest.trim_start();

            if data == "[DONE]" {
                data_lines.clear();
                continue;
            }

            if !data.is_empty() {
                data_lines.push(data.to_string());
            }
        }
    }

    if let Some(event) = finalize_data_lines::<T>(&mut data_lines)? {
        events.push(event);
    }

    Ok(events)
}

fn finalize_data_lines<T: for<'de> Deserialize<'de>>(
    data_lines: &mut Vec<String>,
) -> Result<Option<T>, CommonRequestError> {
    if data_lines.is_empty() {
        return Ok(None);
    }

    let payload = data_lines.join("\n");
    data_lines.clear();

    if payload.is_empty() || payload == "[DONE]" {
        return Ok(None);
    }

    let event: T = serde_json::from_str(&payload)
        .map_err(|e| CommonRequestError::InvalidEventData(format!("JSON parse error: {}", e)))?;

    Ok(Some(event))
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
