#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    strum::EnumString,
    strum::Display,
    strum::IntoStaticStr,
)]
pub enum Model {
    // Claude 3.5 models (latest)
    #[strum(to_string = "claude-3-5-sonnet-20241022")]
    Claude35Sonnet20241022,
    #[strum(to_string = "claude-3-5-sonnet-latest")]
    Claude35SonnetLatest,
    #[strum(to_string = "claude-3-5-haiku-20241022")]
    Claude35Haiku20241022,
    #[strum(to_string = "claude-3-5-haiku-latest")]
    Claude35HaikuLatest,

    // Claude 3 models
    #[strum(to_string = "claude-3-opus-20240229")]
    Claude3Opus20240229,
    #[strum(to_string = "claude-3-opus-latest")]
    Claude3OpusLatest,
    #[strum(to_string = "claude-3-sonnet-20240229")]
    Claude3Sonnet20240229,
    #[strum(to_string = "claude-3-haiku-20240307")]
    Claude3Haiku20240307,

    // Legacy Claude 2 models
    #[strum(to_string = "claude-2.1")]
    Claude21,
    #[strum(to_string = "claude-2.0")]
    Claude20,

    // Instant models
    #[strum(to_string = "claude-instant-1.2")]
    ClaudeInstant12,
}

impl From<Model> for String {
    fn from(model: Model) -> Self {
        model.to_string()
    }
}
