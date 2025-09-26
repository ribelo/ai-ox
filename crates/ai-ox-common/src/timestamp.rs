use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Deserializer, Serialize, de};

/// Unified timestamp type shared across all providers.
/// Internally stores as DateTime<Utc> for consistency and precision.
/// Provides conversions to/from Unix timestamps and ISO strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Timestamp(DateTime<Utc>);

impl<'de> Deserialize<'de> for Timestamp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TimestampVisitor;

        impl<'de> de::Visitor<'de> for TimestampVisitor {
            type Value = Timestamp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a UNIX timestamp (int) or RFC 3339 string")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Timestamp::from_unix_timestamp_i64(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Timestamp::from_unix_timestamp(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if !value.is_finite() {
                    return Err(E::custom("floating point timestamp is not finite"));
                }
                Ok(Timestamp::from_unix_timestamp_i64(value.trunc() as i64))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Timestamp::from_iso_string(value)
                    .map_err(|err| E::custom(format!("invalid RFC 3339 timestamp: {err}")))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_any(TimestampVisitor)
    }
}

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

#[cfg(test)]
mod tests {
    use super::Timestamp;

    #[test]
    fn deserializes_from_integer_seconds() {
        let json = "1758887156";
        let ts: Timestamp = serde_json::from_str(json).expect("integer timestamps should parse");
        assert_eq!(ts.to_unix_timestamp_i64(), 1_758_887_156);
    }

    #[test]
    fn deserializes_from_rfc3339_string() {
        let json = "\"2025-09-26T11:45:56Z\"";
        let ts: Timestamp = serde_json::from_str(json).expect("RFC3339 string should parse");
        assert_eq!(ts.to_unix_timestamp_i64(), 1_758_887_156);
    }
}
