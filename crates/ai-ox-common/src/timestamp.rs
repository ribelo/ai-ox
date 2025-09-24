use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Unified timestamp type shared across all providers.
/// Internally stores as DateTime<Utc> for consistency and precision.
/// Provides conversions to/from Unix timestamps and ISO strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(DateTime<Utc>);

impl Timestamp {
    /// Create a new timestamp from the current time
    #[must_use]
    pub fn now() -> Self {
        Self(Utc::now())
    }

    /// Create a timestamp from a Unix timestamp (u64, seconds since epoch)
    #[must_use]
    pub fn from_unix_timestamp(secs: u64) -> Self {
        let nanos = (secs as i128 * 1_000_000_000).clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        Self(Utc.timestamp_nanos(nanos))
    }

    /// Create a timestamp from a Unix timestamp (i64, seconds since epoch)
    #[must_use]
    pub fn from_unix_timestamp_i64(secs: i64) -> Self {
        let nanos = (secs as i128 * 1_000_000_000).clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        Self(Utc.timestamp_nanos(nanos))
    }

    /// Create a timestamp from an ISO 8601 string
    pub fn from_iso_string(s: &str) -> Result<Self, chrono::ParseError> {
        DateTime::parse_from_rfc3339(s).map(|dt| Self(dt.with_timezone(&Utc)))
    }

    /// Convert to Unix timestamp (u64, seconds since epoch)
    #[must_use]
    pub fn to_unix_timestamp(&self) -> u64 {
        self.0.timestamp() as u64
    }

    /// Convert to Unix timestamp (i64, seconds since epoch)
    #[must_use]
    pub fn to_unix_timestamp_i64(&self) -> i64 {
        self.0.timestamp()
    }

    /// Convert to ISO 8601 string
    #[must_use]
    pub fn to_iso_string(&self) -> String {
        self.0.to_rfc3339()
    }

    /// Get the inner DateTime<Utc>
    #[must_use]
    pub fn inner(&self) -> DateTime<Utc> {
        self.0
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::now()
    }
}

impl From<DateTime<Utc>> for Timestamp {
    fn from(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }
}

impl From<Timestamp> for DateTime<Utc> {
    fn from(ts: Timestamp) -> Self {
        ts.0
    }
}

impl From<u64> for Timestamp {
    fn from(secs: u64) -> Self {
        Self::from_unix_timestamp(secs)
    }
}

impl From<Timestamp> for u64 {
    fn from(ts: Timestamp) -> Self {
        ts.to_unix_timestamp()
    }
}

impl From<i64> for Timestamp {
    fn from(secs: i64) -> Self {
        Self::from_unix_timestamp_i64(secs)
    }
}

impl From<Timestamp> for i64 {
    fn from(ts: Timestamp) -> Self {
        ts.to_unix_timestamp_i64()
    }
}
