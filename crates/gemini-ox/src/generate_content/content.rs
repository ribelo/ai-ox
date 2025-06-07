use super::part::*;
use bon::Builder;
use serde::{Deserialize, Serialize};
use std::result::Result;
use thiserror::Error;

/// Represents the producer of the content.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Content produced by the user.
    #[default]
    User,
    /// Content produced by the model.
    Model,
}

/// Errors that can occur when working with content generation.
#[derive(Debug, Error)]
pub enum ContentError {
    /// Error during serialization or deserialization of JSON data.
    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// The base structured datatype containing multi-part content of a message.
///
/// A `Content` includes a `role` field designating the producer of the `Content`
/// and a `parts` field containing multi-part data that contains the content of the message turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Builder)]
pub struct Content {
    /// Ordered `Parts` that constitute a single message. Parts may have different MIME types.
    #[builder(field = Vec::new())] // Initialize with an empty Vec
    pub parts: Vec<Part>,
    /// Optional. The producer of the content. Must be either 'user' or 'model'.
    /// Useful to set for multi-turn conversations, otherwise can be left blank or unset.
    #[builder(default, into)]
    pub role: Role,
}

impl Content {
    /// Creates a new `Content` with the given role and parts.
    pub fn new(role: Role, parts: impl IntoIterator<Item = impl Into<Part>>) -> Self {
        Self {
            role,
            parts: parts.into_iter().map(Into::into).collect(),
        }
    }

    /// Creates a `Content` with `Role::User` containing a single `Text` part.
    pub fn text(text: impl Into<Text>) -> Self {
        Self {
            parts: vec![Part::new(text.into())],
            role: Role::User,
        }
    }

    /// Creates a `Content` with `Role::User` containing a single `InlineData` (Blob) part,
    /// constructed from the provided data (typically base64 encoded) and mime type.
    #[must_use]
    pub fn inline_data(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        let blob = Blob::new(mime_type, data);
        Self {
            parts: vec![Part::new(blob)],
            role: Role::User,
        }
    }

    /// Creates a `Content` with `Role::Model` containing a single `FunctionCall` part,
    /// constructed from a name and optional serializable arguments.
    ///
    /// # Errors
    /// Returns a `ContentError::SerializationError` if the arguments cannot be serialized.
    #[must_use = "function_call can fail during serialization"]
    pub fn function_call(
        name: impl Into<String>,
        args: Option<impl Serialize>,
    ) -> Result<Self, ContentError> {
        let args_value = match args {
            Some(a) => Some(serde_json::to_value(a).map_err(ContentError::SerializationError)?),
            None => None,
        };

        let function_call = FunctionCall {
            id: None,
            name: name.into(),
            args: args_value,
        };

        Ok(Self {
            parts: vec![Part::new(function_call)],
            role: Role::Model,
        })
    }

    /// Creates a `Content` with `Role::User` containing a single `FunctionResponse` part,
    /// constructed from a name and serializable response object.
    ///
    /// # Errors
    /// Returns a `ContentError::SerializationError` if the response cannot be serialized.
    #[must_use = "function_response can fail during serialization"]
    pub fn function_response(
        name: impl Into<String>,
        response: impl Serialize,
    ) -> Result<Self, ContentError> {
        let response_value =
            serde_json::to_value(response).map_err(ContentError::SerializationError)?;

        let func_response = FunctionResponse {
            id: None,
            name: name.into(),
            response: response_value,
            will_continue: None,
            scheduling: None,
        };

        Ok(Self {
            parts: vec![Part::new(func_response)],
            role: Role::User,
        })
    }

    /// Creates a `Content` with `Role::User` containing a single `FileData` part,
    /// constructed from an optional mime type and a file URI.
    #[must_use]
    pub fn file_data(file_uri: impl Into<String>, mime_type: Option<impl Into<String>>) -> Self {
        let file_data = FileData::new_with_optional_mime_type(file_uri, mime_type);
        Self {
            parts: vec![Part::new(file_data)],
            role: Role::User,
        }
    }
}

impl<S: content_builder::State> ContentBuilder<S> {
    /// Sets the parts of the content, consuming an iterator of items convertible to `Part`.
    pub fn parts(mut self, parts: impl IntoIterator<Item = impl Into<Part>>) -> Self {
        self.parts = parts.into_iter().map(Into::into).collect();
        self
    }

    /// Adds a single part to the content.
    pub fn part(mut self, part: impl Into<Part>) -> Self {
        self.parts.push(part.into());
        self
    }

    /// Adds a single text part to the content.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(Text::new(text).into());
        self
    }

    /// Adds a single blob part to the content.
    pub fn blob(mut self, data: impl Into<String>, mime: impl Into<String>) -> Self {
        self.parts.push(Blob::new(mime, data).into());
        self
    }

    /// Adds a single function response part to the content.
    ///
    /// # Errors
    /// Returns an error if the response cannot be serialized to JSON.
    pub fn function_response(
        mut self,
        name: impl Into<String>,
        response: impl Serialize,
    ) -> Result<Self, ContentError> {
        let response_value =
            serde_json::to_value(response).map_err(ContentError::SerializationError)?;

        self.parts.push(
            FunctionResponse {
                id: None,
                name: name.into(),
                response: response_value,
                will_continue: None,
                scheduling: None,
            }
            .into(),
        );

        Ok(self)
    }

    /// Adds a single function call part to the content.
    ///
    /// # Errors
    /// Returns an error if the arguments cannot be serialized to JSON.
    pub fn function_call(
        mut self,
        name: impl Into<String>,
        args: Option<impl Serialize>,
    ) -> Result<Self, ContentError> {
        let args_value = match args {
            Some(a) => Some(serde_json::to_value(a).map_err(ContentError::SerializationError)?),
            None => None,
        };

        self.parts.push(
            FunctionCall {
                id: None,
                name: name.into(),
                args: args_value,
            }
            .into(),
        );

        Ok(self)
    }

    /// Adds a single file data part to the content.
    pub fn file_data(
        mut self,
        file_uri: impl Into<String>,
        mime_type: Option<impl Into<String>>,
    ) -> Self {
        self.parts
            .push(FileData::new_with_optional_mime_type(file_uri, mime_type).into());
        self
    }
}

impl Content {
    /// Returns a reference to the vector of parts in the content.
    #[must_use]
    pub fn parts(&self) -> &Vec<Part> {
        &self.parts
    }

    /// Returns a mutable reference to the vector of parts in the content.
    pub fn parts_mut(&mut self) -> &mut Vec<Part> {
        &mut self.parts
    }

    /// Adds a new part to the end of the content's parts vector.
    pub fn push(&mut self, part: impl Into<Part>) {
        self.parts.push(part.into());
    }

    /// Returns `Some(&Self)` if the content's role is `User`, otherwise returns `None`.
    #[must_use]
    pub fn as_user(&self) -> Option<&Self> {
        if self.role == Role::User {
            Some(self)
        } else {
            None
        }
    }

    /// Returns `Some(&Self)` if the content's role is `Model`, otherwise returns `None`.
    #[must_use]
    pub fn as_model(&self) -> Option<&Self> {
        if self.role == Role::Model {
            Some(self)
        } else {
            None
        }
    }
}

/// Creates a `Content` with `Role::User` from an iterator of `Part`s.
impl FromIterator<Part> for Content {
    fn from_iter<T: IntoIterator<Item = Part>>(iter: T) -> Self {
        Self::builder().role(Role::User).parts(iter).build()
    }
}

/// Creates a `Content` with `Role::User` containing a single `Text` part from a string slice.
impl From<&str> for Content {
    fn from(value: &str) -> Self {
        Content::builder()
            .role(Role::User)
            .parts(vec![Part::new(Text::from(value))])
            .build()
    }
}

/// Creates a `Content` with `Role::User` containing a single `Text` part from a `String`.
impl From<String> for Content {
    fn from(value: String) -> Self {
        Content::builder()
            .role(Role::User)
            .parts(vec![Part::new(Text::from(value))])
            .build()
    }
}

/// Creates a `Content` with `Role::User` containing a single `FileData` part.
impl From<FileData> for Content {
    fn from(value: FileData) -> Self {
        Content::builder()
            .role(Role::User)
            .parts(vec![Part::new(value)])
            .build()
    }
}

impl From<Content> for Vec<Content> {
    fn from(value: Content) -> Self {
        vec![value]
    }
}

/// Combines adjacent Text parts within sequences of Content items that share the same Role.
///
/// This function takes a collection of Content items and returns a new Vec<Content> where:
/// 1. Content items with the same role are preserved as separate items
/// 2. Within each Content item, adjacent Text parts are merged
/// 3. Empty Content items are skipped
///
/// # Examples
///
/// ```
/// use gemini_ox::generate_content::content::{Content, Role, combine_content_list}; // Added imports (Removed Part)
///
/// // Combining content with different roles preserves the separate roles
/// let user_content = Content::new(Role::User, vec!["Hello"]);
/// let model_content = Content::new(Role::Model, vec!["Hi there"]);
/// let result = combine_content_list(vec![user_content, model_content]);
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].role, Role::User);
/// assert_eq!(result[1].role, Role::Model);
///
/// // Combining two text parts with the same role merges the text within one Content
/// let content1 = Content::text("Hello");
/// let content2 = Content::text(" world");
/// let result = combine_content_list(vec![content1, content2]);
/// assert_eq!(result.len(), 1);
/// assert_eq!(result[0].parts.len(), 1);
/// assert_eq!(result[0].parts[0].as_text().unwrap().to_string(), "Hello world");
/// ```
pub fn combine_content_list(content_list: impl IntoIterator<Item = Content>) -> Vec<Content> {
    let mut result: Vec<Content> = Vec::new();
    let mut current_parts: Vec<Part> = Vec::new();
    let mut current_role: Option<Role> = None;

    for content in content_list {
        if content.parts.is_empty() {
            continue;
        }

        let role_changed = current_role != Some(content.role);

        if role_changed && !current_parts.is_empty() {
            // Finalize the previous block before starting a new one
            result.push(Content {
                role: current_role.unwrap(), // Safe because current_parts non-empty
                parts: std::mem::take(&mut current_parts),
            });
        }

        if role_changed || current_role.is_none() {
            current_role = Some(content.role); // Start new block (or first block)
        }

        // Now merge/add parts to current_parts
        for new_part in content.parts {
            let mut merged = false;
            if let Some(last_part) = current_parts.last_mut() {
                if let Some(last_text) = last_part.as_mut_text() {
                    if let PartData::Text(new_text_data) = &new_part.data {
                        last_text.push_text(new_text_data);
                        merged = true;
                    }
                }
            }
            if !merged {
                current_parts.push(new_part);
            }
        }
    }

    // Add the last processed block, if any
    if !current_parts.is_empty() {
        result.push(Content {
            role: current_role.unwrap(), // Safe because current_parts non-empty
            parts: current_parts,
        });
    }

    result
}
