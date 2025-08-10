use std::{fmt, path::Path};

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::tool::{ToolResult, ToolUse};

use strum::{Display, EnumString};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

impl ImageSource {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
        let media_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        Ok(ImageSource::Base64 {
            media_type,
            data: base64_data,
        })
    }
}

impl fmt::Display for ImageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageSource::Base64 { media_type, data } => {
                let truncated_data = if data.len() > 20 {
                    format!("{}...", &data[..20])
                } else {
                    data.clone()
                };
                write!(f, "Base64 ({}, {})", media_type, truncated_data)
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Image {
    pub source: ImageSource,
}

impl Image {
    pub fn new(source: ImageSource) -> Self {
        Self { source }
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let source = ImageSource::from_path(path)?;
        Ok(Self::new(source))
    }

    pub fn from_base64(media_type: String, data: String) -> Self {
        let source = ImageSource::Base64 { media_type, data };
        Self::new(source)
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Image: {}", self.source)
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Text {
    pub text: String,
}

impl Text {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn push_str(&mut self, string: &str) {
        self.text.push_str(string);
    }
}

impl From<String> for Text {
    fn from(text: String) -> Self {
        Text { text }
    }
}

impl From<&str> for Text {
    fn from(text: &str) -> Self {
        Text {
            text: text.to_owned(),
        }
    }
}

impl From<&String> for Text {
    fn from(text: &String) -> Self {
        Text { text: text.clone() }
    }
}

impl From<Box<str>> for Text {
    fn from(text: Box<str>) -> Self {
        Text {
            text: text.into_string(),
        }
    }
}

impl From<std::borrow::Cow<'_, str>> for Text {
    fn from(text: std::borrow::Cow<'_, str>) -> Self {
        Text {
            text: text.into_owned(),
        }
    }
}

impl From<serde_json::Value> for Text {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => Text { text: s },
            _ => Text {
                text: value.to_string(),
            },
        }
    }
}

impl From<Text> for String {
    fn from(text: Text) -> Self {
        text.text
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text(Text),
    Image(Image),
    ToolUse(ToolUse),
    ToolResult(ToolResult),
}

impl Content {
    pub fn text<T: Into<String>>(text: T) -> Self {
        Self::Text(Text { text: text.into() })
    }

    pub fn image(source: ImageSource) -> Self {
        Self::Image(Image { source })
    }

    pub fn tool_use(tool_use: ToolUse) -> Self {
        Self::ToolUse(tool_use)
    }

    pub fn tool_result(tool_result: ToolResult) -> Self {
        Self::ToolResult(tool_result)
    }

    pub fn as_text(&self) -> Option<&Text> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_image(&self) -> Option<&Image> {
        if let Self::Image(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_tool_use(&self) -> Option<&ToolUse> {
        if let Self::ToolUse(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResult> {
        if let Self::ToolResult(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_json(&self) -> Option<serde_json::Value> {
        match self {
            Self::Text(text) => serde_json::from_str(&text.text).ok(),
            _ => None,
        }
    }
}

impl<T: Into<Text>> From<T> for Content {
    fn from(text: T) -> Self {
        Content::Text(text.into())
    }
}

impl From<Image> for Content {
    fn from(image: Image) -> Self {
        Content::Image(image)
    }
}

impl From<ToolUse> for Content {
    fn from(tool_use: ToolUse) -> Self {
        Content::ToolUse(tool_use)
    }
}

impl From<ToolResult> for Content {
    fn from(tool_result: ToolResult) -> Self {
        Content::ToolResult(tool_result)
    }
}

impl fmt::Display for Content {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(text) => fmt::Display::fmt(text, f),
            Self::Image(image) => fmt::Display::fmt(image, f),
            Self::ToolUse(tool_use) => fmt::Display::fmt(tool_use, f),
            Self::ToolResult(tool_result) => fmt::Display::fmt(tool_result, f),
        }
    }
}

// For streaming content blocks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

impl Message {
    pub fn new(role: Role, content: Vec<Content>) -> Self {
        Self { role, content }
    }

    pub fn user<T: Into<Content>>(content: Vec<T>) -> Self {
        Self {
            role: Role::User,
            content: content.into_iter().map(Into::into).collect(),
        }
    }

    pub fn assistant<T: Into<Content>>(content: Vec<T>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into_iter().map(Into::into).collect(),
        }
    }

    pub fn add_content<T: Into<Content>>(&mut self, content: T) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl<T: Into<Content>> From<T> for Message {
    fn from(content: T) -> Self {
        Message::user(vec![content])
    }
}

impl From<Vec<Content>> for Message {
    fn from(content: Vec<Content>) -> Self {
        Message::user(content)
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.role)?;
        for (i, content) in self.content.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", content)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn push<T: Into<Message>>(&mut self, message: T) {
        self.0.push(message.into());
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Message> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Message> {
        self.0.iter_mut()
    }

    pub fn last(&self) -> Option<&Message> {
        self.0.last()
    }

    pub fn last_mut(&mut self) -> Option<&mut Message> {
        self.0.last_mut()
    }
}

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

impl<T> From<Vec<T>> for Messages
where
    T: Into<Message>,
{
    fn from(value: Vec<T>) -> Self {
        Messages(value.into_iter().map(Into::into).collect())
    }
}

impl FromIterator<Message> for Messages {
    fn from_iter<T: IntoIterator<Item = Message>>(iter: T) -> Self {
        Messages(iter.into_iter().collect())
    }
}

impl std::ops::Index<usize> for Messages {
    type Output = Message;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Messages {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Messages {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Messages {
    type Item = &'a Message;
    type IntoIter = std::slice::Iter<'a, Message>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut Messages {
    type Item = &'a mut Message;
    type IntoIter = std::slice::IterMut<'a, Message>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}