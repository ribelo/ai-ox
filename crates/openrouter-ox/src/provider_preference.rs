use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderPreferences {
    #[serde(default)]
    pub allow_fallbacks: Option<bool>,
    #[serde(default)]
    pub require_parameters: Option<bool>,
    #[serde(default)]
    pub data_collection: Option<DataCollection>,
    #[serde(default)]
    pub order: Option<Vec<Provider>>,
    #[serde(default)]
    pub ignore: Option<Vec<Provider>>,
    #[serde(default)]
    pub quantizations: Option<Vec<Quantization>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataCollection {
    Deny,
    Allow,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    #[serde(rename = "Google AI Studio")]
    GoogleAiStudio,
    #[serde(rename = "Amazon Bedrock")]
    AmazonBedrock,
    Groq,
    SambaNova,
    Cohere,
    Mistral,
    Together,
    #[serde(rename = "Together 2")]
    Together2,
    Fireworks,
    DeepInfra,
    Lepton,
    Novita,
    Avian,
    Lambda,
    Azure,
    Modal,
    AnyScale,
    Replicate,
    Perplexity,
    Recursal,
    OctoAI,
    DeepSeek,
    Infermatic,
    AI21,
    Featherless,
    Inflection,
    xAI,
    Cloudflare,
    #[serde(rename = "SF Compute")]
    SfCompute,
    Minimax,
    Nineteen,
    #[serde(rename = "01.AI")]
    ZeroOneAI,
    HuggingFace,
    Mancer,
    #[serde(rename = "Mancer 2")]
    Mancer2,
    Hyperbolic,
    #[serde(rename = "Hyperbolic 2")]
    Hyperbolic2,
    #[serde(rename = "Lynn 2")]
    Lynn2,
    Lynn,
    Reflection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    Int4,
    Int8,
    Fp6,
    Fp8,
    Fp16,
    Bf16,
    Fp32,
    Unknown,
}
