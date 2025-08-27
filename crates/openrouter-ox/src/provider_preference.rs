use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sort {
    Price,
    Throughput,
    Latency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxPrice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderPreferences {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<Provider>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<Provider>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<Provider>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<Sort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,
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
    AionLabs,
    Alibaba,
    AtlasCloud,
    Atoma,
    BaseTen,
    Cerebras,
    Chutes,
    CrofAI,
    Crusoe,
    Enfer,
    Friendli,
    GMICloud,
    Inception,
    InferenceNet,
    InoCloud,
    Kluster,
    Liquid,
    Meta,
    #[serde(rename = "Moonshot AI")]
    MoonshotAI,
    Morph,
    NCompass,
    Nebius,
    NextBit,
    OpenInference,
    Parasail,
    Phala,
    SiliconFlow,
    Stealth,
    Switchpoint,
    Targon,
    Ubicloud,
    Venice,
    WandB,
    #[serde(rename = "Z.AI")]
    ZAI,
    #[serde(rename = "Cent-ML")]
    CentML,
    #[serde(rename = "SambaNova 2")]
    SambaNova2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    Int4,
    Int8,
    Fp4,
    Fp6,
    Fp8,
    Fp16,
    Bf16,
    Fp32,
    Unknown,
}
