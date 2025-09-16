use std::convert::TryInto;

use crate::{
    content::message::Message,
    errors::GenerateContentError,
    model::request::ModelRequest,
    tool::{FunctionMetadata, Tool},
};

use gemini_ox::{
    content::Content as GeminiContent,
    generate_content::request::GenerateContentRequest as GeminiRequest,
};

pub fn model_request_to_gemini_request(
    request: &ModelRequest,
    model: impl Into<String>,
) -> Result<GeminiRequest, GenerateContentError> {
    let contents = request
        .messages
        .iter()
        .cloned()
        .map(|message| message.try_into())
        .collect::<Result<Vec<GeminiContent>, _>>()?;

    let system_instruction = if let Some(system_message) = &request.system_message {
        Some(system_message.clone().try_into()?)
    } else {
        None
    };

    let tools = if let Some(tools) = &request.tools {
        let gemini_tools = convert_tools_to_gemini_values(tools);
        if gemini_tools.is_empty() {
            None
        } else {
            Some(
                gemini_tools
                    .into_iter()
                    .map(|tool| serde_json::to_value(tool).unwrap())
                    .collect(),
            )
        }
    } else {
        None
    };

    Ok(GeminiRequest {
        contents,
        tools,
        model: model.into(),
        tool_config: None,
        safety_settings: None,
        system_instruction,
        generation_config: None,
        cached_content: None,
    })
}

pub fn gemini_request_to_model_request(
    request: &GeminiRequest,
) -> Result<ModelRequest, GenerateContentError> {
    let messages = request
        .contents
        .iter()
        .cloned()
        .map(|content| content.try_into())
        .collect::<Result<Vec<Message>, _>>()?;

    let system_message = if let Some(system_instruction) = &request.system_instruction {
        Some(system_instruction.clone().try_into()?)
    } else {
        None
    };

    let tools = if let Some(tool_values) = &request.tools {
        let converted = convert_gemini_values_to_tools(tool_values)?;
        if converted.is_empty() {
            None
        } else {
            Some(converted)
        }
    } else {
        None
    };

    Ok(ModelRequest {
        messages,
        tools,
        system_message,
    })
}

fn convert_tools_to_gemini_values(tools: &[Tool]) -> Vec<gemini_ox::tool::Tool> {
    let mut gemini_tools = Vec::new();
    for tool in tools {
        match tool {
            Tool::FunctionDeclarations(functions) => {
                let functions = functions
                    .iter()
                    .map(|func| gemini_ox::tool::FunctionMetadata {
                        name: func.name.clone(),
                        description: func.description.clone(),
                        parameters: func.parameters.clone(),
                    })
                    .collect();
                gemini_tools.push(gemini_ox::tool::Tool::FunctionDeclarations(functions));
            }
            #[cfg(feature = "gemini")]
            Tool::GeminiTool(inner) => {
                gemini_tools.push(inner.clone());
            }
        }
    }
    gemini_tools
}

fn convert_gemini_values_to_tools(
    values: &[serde_json::Value],
) -> Result<Vec<Tool>, GenerateContentError> {
    let mut result = Vec::new();

    for value in values {
        let parsed: gemini_ox::tool::Tool =
            serde_json::from_value(value.clone()).map_err(|err| {
                GenerateContentError::message_conversion(&format!(
                    "Failed to parse Gemini tool definition: {}",
                    err
                ))
            })?;

        match parsed {
            gemini_ox::tool::Tool::FunctionDeclarations(functions) => {
                let converted = functions
                    .into_iter()
                    .map(|func| FunctionMetadata {
                        name: func.name,
                        description: func.description,
                        parameters: func.parameters,
                    })
                    .collect();
                result.push(Tool::FunctionDeclarations(converted));
            }
            other => {
                #[cfg(feature = "gemini")]
                {
                    result.push(Tool::GeminiTool(other));
                }
                #[cfg(not(feature = "gemini"))]
                {
                    let _ = other;
                }
            }
        }
    }

    Ok(result)
}
