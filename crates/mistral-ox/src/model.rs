use serde::{Deserialize, Serialize};
use strum::{Display, EnumString, IntoStaticStr};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    Serialize,
    Deserialize,
    EnumString,
    Display,
    IntoStaticStr,
)]
pub enum Model {
    // Open source models
    #[strum(to_string = "mistral-embed")]
    MistralEmbed,
    #[strum(to_string = "open-mistral-7b")]
    OpenMistral7b,
    #[strum(to_string = "open-mixtral-8x7b")]
    OpenMixtral8x7b,
    #[strum(to_string = "open-mixtral-8x22b")]
    OpenMixtral8x22b,
    
    // Commercial models
    #[strum(to_string = "mistral-tiny")]
    MistralTiny,
    #[strum(to_string = "mistral-small")]
    MistralSmall,
    #[strum(to_string = "mistral-small-2402")]
    MistralSmall2402,
    #[strum(to_string = "mistral-small-2409")]
    MistralSmall2409,
    #[strum(to_string = "mistral-small-latest")]
    MistralSmallLatest,
    #[strum(to_string = "mistral-medium")]
    MistralMedium,
    #[strum(to_string = "mistral-medium-latest")]
    MistralMediumLatest,
    #[strum(to_string = "mistral-large")]
    MistralLarge,
    #[strum(to_string = "mistral-large-2402")]
    MistralLarge2402,
    #[strum(to_string = "mistral-large-2407")]
    MistralLarge2407,
    #[strum(to_string = "mistral-large-2411")]
    MistralLarge2411,
    #[strum(to_string = "mistral-large-latest")]
    MistralLargeLatest,
    
    // Codestral models
    #[strum(to_string = "codestral-latest")]
    CodestralLatest,
    #[strum(to_string = "codestral-2405")]
    Codestral2405,
    #[strum(to_string = "codestral-2501")]
    Codestral2501,
    #[strum(to_string = "codestral-mamba-latest")]
    CodestralMambaLatest,
    
    // Pixtral models
    #[strum(to_string = "pixtral-12b")]
    Pixtral12b,
    #[strum(to_string = "pixtral-12b-2409")]
    Pixtral12b2409,
    #[strum(to_string = "pixtral-large-latest")]
    PixtralLargeLatest,
    
    // Magistral models
    #[strum(to_string = "magistral-medium-2506")]
    MagistralMedium2506,
    
    // Special models
    #[strum(to_string = "mistral-ocr-2505")]
    MistralOcr2505,
    
    // Voxtral models
    #[strum(to_string = "voxtral-small")]
    VoxtralSmall,
    #[strum(to_string = "voxtral-mini-2507")]
    VoxtralMini2507,
    #[strum(to_string = "voxtral-mini-transcribe")]
    VoxtralMiniTranscribe,
}

impl From<Model> for String {
    fn from(model: Model) -> Self {
        model.to_string()
    }
}