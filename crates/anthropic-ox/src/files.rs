use serde::{Deserialize, Serialize};

/// Represents a file to be uploaded.
#[derive(Debug, Clone)]
pub struct FileUploadRequest {
    /// The raw byte content of the file.
    pub content: Vec<u8>,
    /// The name of the file.
    pub filename: String,
    /// The MIME type of the file.
    pub mime_type: String,
}

/// Information about a file stored on the server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileInfo {
    /// The unique identifier for the file.
    pub id: String,
    /// The type of the object, which is always "file".
    #[serde(rename = "type")]
    pub object_type: String,
    /// The name of the file.
    pub filename: String,
    /// The MIME type of the file.
    pub mime_type: String,
    /// The size of the file in bytes.
    pub size_bytes: u64,
    /// The creation date of the file.
    pub created_at: String,
    /// Whether the file is downloadable.
    pub downloadable: bool,
}

/// A response containing a list of files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileListResponse {
    /// The list of files.
    pub data: Vec<FileInfo>,
    /// Indicates if there are more files to fetch.
    pub has_more: bool,
    /// The ID of the first file in the list.
    pub first_id: Option<String>,
    /// The ID of the last file in the list.
    pub last_id: Option<String>,
}
