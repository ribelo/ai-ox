use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::OpenRouter;

/// Represents a model pricing structure.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ModelPricing {
    /// Cost per token for prompt
    #[serde(deserialize_with = "deserialize_string_or_number")]
    pub prompt: f64,
    /// Cost per token for completion
    #[serde(deserialize_with = "deserialize_string_or_number")]
    pub completion: f64,
}

// Helper function to deserialize either a string or a number to f64
fn deserialize_string_or_number<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StringOrNumberVisitor;

    impl<'de> serde::de::Visitor<'de> for StringOrNumberVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a number or string containing a number")
        }

        fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(value as f64)
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(value as f64)
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            value.parse::<f64>().map_err(serde::de::Error::custom)
        }
    }

    deserializer.deserialize_any(StringOrNumberVisitor)
}

/// Represents a model available through OpenRouter API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Model {
    /// Unique identifier for the model
    pub id: String,
    /// Human-readable name of the model
    pub name: String,
    /// Description of the model and its capabilities
    pub description: String,
    /// Pricing information for using the model
    pub pricing: ModelPricing,
}

/// Response structure for the models endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelsResponseRaw {
    data: Vec<Model>,
}

/// Request struct for model listing API
#[derive(Debug, Serialize, Deserialize, Clone, Default, Builder)]
pub struct ModelsRequest {}

impl ModelsRequest {
    /// Retrieves the list of available models from the API.
    pub async fn fetch(&self, client: &reqwest::Client) -> Result<Vec<Model>, reqwest::Error> {
        let response = client
            .get("https://openrouter.ai/api/v1/models")
            .send()
            .await?
            .json::<ModelsResponseRaw>()
            .await?;

        Ok(response.data)
    }
}

impl OpenRouter {
    /// Retrieves the list of available models from the API.
    pub async fn models(&self) -> Result<Vec<Model>, reqwest::Error> {
        let request = ModelsRequest::default();
        request.fetch(&self.client).await
    }
}
