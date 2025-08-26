use serde_json;
use anthropic_ox::message::{Content, ThinkingContent};

fn main() {
    let content = Content::Thinking(ThinkingContent::new("Reasoning...".to_string()));
    println!("Actual JSON: {}", serde_json::to_string(&content).unwrap());
}
