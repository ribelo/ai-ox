use anthropic_ox::{
    message::Message,
    request::ChatRequest,
    response::StreamEvent,
    Anthropic,
};
use futures_util::StreamExt;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    time::{sleep, Duration},
};

#[tokio::test]
async fn streaming_should_survive_split_sse_chunks() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();

        let mut buffer = Vec::new();
        loop {
            let mut chunk = [0u8; 1024];
            let n = socket.read(&mut chunk).await.unwrap();
            if n == 0 {
                return;
            }
            buffer.extend_from_slice(&chunk[..n]);

            if let Some(pos) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                let header_end = pos + 4;
                let headers = &buffer[..header_end];
                let headers_str = String::from_utf8_lossy(headers).to_lowercase();
                let content_length = headers_str
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .and_then(|len| len.trim().parse::<usize>().ok())
                    .unwrap_or(0);

                let mut body = buffer[header_end..].to_vec();
                while body.len() < content_length {
                    let mut chunk = [0u8; 1024];
                    let n = socket.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    body.extend_from_slice(&chunk[..n]);
                }
                break;
            }
        }

        let response_head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
        socket.write_all(response_head.as_bytes()).await.unwrap();

        let chunk1_data = "data: {\"type\":\"message_stop\"";
        let chunk2_data = "}\n\n";
        let chunk1 = format!("{:x}\r\n{}\r\n", chunk1_data.len(), chunk1_data);
        let chunk2 = format!("{:x}\r\n{}\r\n", chunk2_data.len(), chunk2_data);

        socket.write_all(chunk1.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
        sleep(Duration::from_millis(50)).await;
        socket.write_all(chunk2.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });

    let base_url = format!("http://{}", addr);
    let client = Anthropic::builder()
        .api_key("test-key")
        .base_url(base_url)
        .build();

    let request = ChatRequest::builder()
        .model("claude-test")
        .messages(vec![Message::user(vec!["ping"])])
        .build();

    let mut stream = client.stream(&request);
    let first_event = stream.next().await.expect("expected stream event");
    let _event: StreamEvent = first_event.expect("streaming should not fail on split chunks");

    server.await.unwrap();
}
