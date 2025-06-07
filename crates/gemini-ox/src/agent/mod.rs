mod context;
mod error;
mod events;
mod traits;

use std::{marker::PhantomData, pin::Pin, sync::Arc};

use async_stream::try_stream;
use bon::Builder;
use context::RunContext;
use derive_more::Deref;
use error::AgentError;
use events::AgentEvent;
use futures_util::{Stream, StreamExt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
    Gemini, GeminiRequestError, GenerationConfig,
    generate_content::{
        content::{self, Content},
        part::FunctionCall,
        response::GenerateContentResponse,
        usage::UsageMetadata,
    },
    tool::{FunctionCallError, Tool, ToolBox},
};
// pub use typed_agent::SimpleTypedAgent;

#[derive(Clone, Builder)]
pub struct Agent<T: Clone + Send + Sync + 'static> {
    #[builder(start_fn)]
    pub state: Option<T>,
    #[builder(field)]
    pub instruction: Option<Arc<dyn Fn(&T) -> String + Send + Sync>>,
    pub(crate) gemini: Gemini,
    #[builder(into)]
    pub name: String,
    #[builder(into)]
    pub description: Option<String>,
    #[builder(into)]
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    #[builder(default = 12)]
    pub max_iterations: u32,
}

// /// A wrapper struct that holds a Gemini client and an Agent implementation.
// /// It facilitates executing the agent's logic by providing the necessary client.
// /// This acts as a simple pass-through executor, delegating all Agent trait methods
// /// to the contained agent instance.
// #[derive(Clone, Debug, Builder, Deref)] // Added Clone
// pub struct GeminiAgent<A: Agent> {
//     #[builder(field)]
//     pub(crate) tools: Option<ToolBox>,
//     pub(crate) gemini: Gemini,
//     #[deref]
//     pub(crate) agent: A,
// }

// impl<S: gemini_agent_builder::State, A: Agent> GeminiAgentBuilder<A, S> {
//     pub fn tools(mut self, toolbox: ToolBox) -> Self {
//         self.tools = Some(toolbox);
//         self
//     }
//     pub fn tool<T: Tool>(mut self, tool: T) -> Self {
//         self.tools.get_or_insert_default().add(tool);
//         self
//     }
// }

// impl<S: gemini_agent_builder::IsComplete, A: Agent> GeminiAgentBuilder<A, S> {
//     pub fn typed<O: JsonSchema + DeserializeOwned>(self) -> GeminiTypedAgent<A, O> {
//         GeminiTypedAgent {
//             agent: self.build(),
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<A: Agent> GeminiAgent<A> {
//     fn generation_config(&self) -> GenerationConfig {
//         GenerationConfig::builder()
//             .maybe_max_output_tokens(self.agent.max_tokens())
//             .maybe_temperature(self.agent.temperature())
//             .maybe_top_p(self.agent.top_p())
//             .maybe_top_k(self.agent.top_k())
//             .maybe_stop_sequences(self.agent.stop_sequences())
//             .build()
//     }
//     /// Gets a single, complete `GenerateContentResponse` from one call to the LLM.
//     /// Makes a single, direct call to the underlying model API.
//     /// Returns a `Result<GenerateContentResponse, AgentError>`.
//     pub async fn generate(
//         &self,
//         content: impl Into<Vec<Content>> + Send,
//     ) -> Result<GenerateContentResponse, AgentError> {
//         let config = self.generation_config();
//         // Build and send the request
//         self.gemini
//             .generate_content()
//             .maybe_system_instruction(self.agent.instructions())
//             .model(self.agent.model())
//             .content_list(content.into())
//             .generation_config(config)
//             .maybe_tools(self.tools.clone())
//             .maybe_tool_config(self.agent.tool_config())
//             .build()
//             .send()
//             .await
//             .map_err(AgentError::ApiError)
//     }

//     /// Streams the parts (chunks) of a single `GenerateContentResponse` from one call to the LLM.
//     /// Makes a single, streaming call to the underlying model API.
//     /// Yields `Result<GenerateContentResponse, AgentError>` items representing the chunks.
//     pub fn stream_response<'a>(
//         &'a self,
//         content: impl Into<Vec<Content>> + Send,
//     ) -> Pin<Box<dyn Stream<Item = Result<GenerateContentResponse, AgentError>> + Send + 'a>> {
//         let config = self.generation_config();

//         // Clone/copy necessary data for the async block
//         // We need gemini to live long enough for the stream call. Pass by ref.
//         let model = self.agent.model().to_string(); // Owned String
//         let content: Vec<Content> = content.into(); // Owned Vec
//         let tools = self.tools.clone(); // Owned Option<ToolBox>

//         // Ensure the request builder lives long enough for the stream() call
//         // The stream itself needs 'a lifetime because it might depend on self or gemini implicitly
//         // via the request structure, even if parts are cloned internally by stream().
//         let stream = try_stream! {
//             let request = self.gemini // gemini reference is captured by the stream closure
//                 .generate_content()
//                 .maybe_system_instruction(self.agent.instructions())
//                 .model(model) // Model string is moved
//                 .content_list(content) // Content vec is moved
//                 .generation_config(config) // Config is moved (Copy or Clone assumed)
//                 .maybe_tools(tools) // Tools Option<ToolBox> is moved
//                 .maybe_tool_config(self.agent.tool_config())
//                 .build(); // Builds the request, using captured vars

//             // Create the inner stream from the request.
//             // The stream returned here borrows `request` for its lifetime ('req_lifetime)
//             // and request borrows `gemini` for 'a.
//             // `request.stream` has signature stream<'req>(&'req self) -> Pin<Box<Stream + Send + 'req>>
//             let mut inner_stream = request.stream();

//             // Iterate over the inner stream and yield its items.
//             // The 'yield' suspends the state machine, preserving the state of `request` and `inner_stream`.
//             while let Some(result) = inner_stream.next().await {
//                 yield result?; // Yield Ok values, propagate Err. The `?` handles the Result.
//             }
//         };

//         // Pin the resulting stream. The lifetime 'a is correctly propagated
//         // because the try_stream! captures `gemini` (with lifetime 'a) and `self` (implicitly, also 'a).
//         Box::pin(stream)
//     }

//     /// Executes the agent process, potentially involving multiple turns (LLM calls) and tool calls,
//     /// until a final `GenerateContentResponse` is produced or an error occurs.
//     /// Returns the final `Result<GenerateContentResponse, AgentError>`.
//     pub async fn execute(
//         &self,
//         content: impl Into<Vec<Content>> + Send,
//     ) -> Result<GenerateContentResponse, AgentError> {
//         let mut content_list: Vec<Content> = content.into();

//         for _ in 0..self.agent.max_iterations() {
//             // Make an API call using get_response
//             // Note: get_response takes Gemini by value, so we clone it.
//             let resp = self.generate(content_list.clone()).await?;

//             // Add the model's response (including potential function *calls*) to the history
//             // The `From<GenerateContentResponse> for Content` impl is used here.
//             content_list.push(resp.clone().into());

//             // Check if the agent has tools AND if the response contains function calls
//             if let Some(tools) = &self.tools {
//                 match resp.invoke_functions(tools).await {
//                     Ok(Some(tool_result_content)) => {
//                         // Add the function execution results to the history
//                         // The invoke_functions should ideally return Content with Role::Function/Tool
//                         content_list.push(tool_result_content);
//                         // Continue the loop to get the next response from the model
//                     }
//                     Ok(None) => return Ok(resp), // No function calls, execution is finished
//                     Err(tool_error) => {
//                         // An error occurred during tool execution
//                         return Err(AgentError::FunctionCallError(tool_error));
//                     }
//                 }
//             } else {
//                 // Agent has no tools configured. The first response is the final one.
//                 return Ok(resp);
//             }
//         }

//         // If the loop completes without returning, it means max_iterations was reached
//         Err(AgentError::MaxIterationsReached {
//             limit: self.agent.max_iterations(),
//         })
//     }

//     /// Streams events related to the entire agent execution process.
//     /// This may include multiple turns, yielding events like `AgentStart`, `AgentResponse` (LLM chunks),
//     /// `StreamEnd` (per LLM call), tool calls, `AgentFinish`, or `AgentError`.
//     /// Yields `Result<AgentEvent, AgentError>` items.
//     pub fn stream_events<'a>(
//         &'a self,
//         content: impl Into<Vec<Content>> + Send + 'static,
//     ) -> Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'a>> {
//         // Stream lifetime tied to 'a (self)
//         Box::pin(try_stream! {
//             yield AgentEvent::AgentStart; // Yield start event once

//             let mut agent_content: Vec<Content> = content.into();
//             let tools = self.tools.clone(); // Clone tools once if they exist

//             for _iteration in 0..self.agent.max_iterations() {
//                 // Note: No explicit ModelRequest event defined in AgentEvent enum

//                 // Get the stream for this turn using stream_response
//                 let mut response_stream = self.stream_response(agent_content.clone()); // stream_response takes Gemini by value

//                 let mut current_turn_function_calls: Vec<FunctionCall> = Vec::new();
//                 let mut current_turn_content: Vec<Content> = Vec::new();
//                 let mut current_turn_usage: UsageMetadata = UsageMetadata::default();
//                 // We collect all chunks to reliably get the final state, usage, and potential function calls

//                 while let Some(response_result) = response_stream.next().await {
//                     match response_result {
//                         Ok(response) => {
//                             current_turn_content.extend(response.content_owned()); // Store chunk for aggregation
//                             if let Some(ref usage_metadata) = response.usage_metadata {
//                                 current_turn_usage += usage_metadata.clone();
//                             }
//                             // Extract and store function calls from this chunk
//                             let calls_in_chunk = response
//                                 .candidates
//                                 .iter()
//                                 .flat_map(|candidate| candidate.content.parts()) // Get all parts from all candidates
//                                 .filter_map(|part| part.as_function_call()) // Filter for function calls
//                                 .cloned(); // Clone the FunctionCall references
//                             current_turn_function_calls.extend(calls_in_chunk);
//                             yield AgentEvent::AgentResponse  { response }; // Yield the LLM chunk
//                         }
//                         Err(error) => {
//                             yield AgentEvent::AgentError { error: error.to_string() };
//                             Err(error)?; // Propagate error within try_stream! to terminate it
//                         }
//                     }
//                 }

//                 // Yield the end of the model's response stream for this turn
//                 yield AgentEvent::StreamEnd { usage: Some(current_turn_usage) };

//                 // Add the model's complete response content to the history for the next turn
//                 // This uses the From<GenerateContentResponse> for Content conversion.
//                 agent_content.extend(content::combine_content_list(current_turn_content));

//                 // Check for function calls using the functions collected during the stream and if tools are configured
//                 if let Some(ref agent_tools) = tools {
//                     // Use the function calls collected during the stream processing for this turn
//                     if current_turn_function_calls.is_empty() {
//                         // No function calls were detected in this turn's stream, agent's iterative execution is complete.
//                         yield AgentEvent::AgentFinish;
//                         return; // Exit the try_stream successfully
//                     }
//                     // Invoke the detected functions using the collected calls
//                     let mut join_set = tokio::task::JoinSet::new();
//                     for fc in current_turn_function_calls { // Consume collected calls
//                         let tools_clone = agent_tools.clone();
//                         join_set.spawn(async move {
//                             tools_clone.invoke(fc).await // Execute the tool invocation
//                         });
//                     }

//                     // Collect results
//                     let mut tool_invocation_results: Vec<Content> = Vec::new();
//                     let mut tool_error_occurred: Option<FunctionCallError> = None;

//                     while let Some(result) = join_set.join_next().await {
//                         match result {
//                             Ok(Ok(content)) => {
//                                 // Task completed, tool succeeded, collect content
//                                 tool_invocation_results.push(content);
//                             }
//                             Ok(Err(tool_error)) => {
//                                 // Task completed, but tool invocation failed
//                                 tool_error_occurred = Some(tool_error);
//                                 break; // Break on first error
//                             }
//                             Err(join_error) => {
//                                 // Task failed to execute (panic, cancellation)
//                                 tool_error_occurred = Some(FunctionCallError::ExecutionFailed(join_error.to_string()));
//                                 break;
//                             }
//                         }
//                     }

//                     // Handle the outcome of tool invocations
//                     match tool_error_occurred {
//                         Some(tool_error) => {
//                             yield AgentEvent::AgentError { error: tool_error.to_string() };
//                             // Propagate the tool error to terminate the stream.
//                             Err(AgentError::FunctionCallError(tool_error))?;
//                         }
//                         None => {
//                             // Add tool results to history for the next turn
//                             agent_content.extend(tool_invocation_results);
//                             // Continue the loop
//                         }
//                     }
//                 } else {
//                     // No tools are configured for this agent. The first complete model response is the final response.
//                     yield AgentEvent::AgentFinish;
//                     return; // Exit the try_stream successfully
//                 }
//             }

//             // If the loop completes without returning, it means max_iterations was reached.
//             let max_iterations_error = AgentError::MaxIterationsReached { limit: self.agent.max_iterations() };
//             yield AgentEvent::AgentError { error: max_iterations_error.to_string() };
//             Err(max_iterations_error)?; // Return the error
//         })
//     }
// }

// /// A wrapper struct similar to `AgentExecutor`, but specifically for `TypedAgent` implementations.
// /// It holds a Gemini client and a TypedAgent, facilitating the execution of typed agent workflows.
// /// Delegates `Agent` and `TypedAgent` methods to the contained agent.
// #[derive(Clone, Debug, Deref)]
// pub struct GeminiTypedAgent<A: Agent, O: JsonSchema + DeserializeOwned> {
//     #[deref]
//     agent: GeminiAgent<A>,
//     _phantom: PhantomData<O>,
// }

// impl<T, O> GeminiTypedAgent<T, O>
// where
//     T: Agent,
//     O: JsonSchema + DeserializeOwned,
// {
//     fn generate_schema(&self) -> Result<Value, AgentError> {
//         let settings = schemars::r#gen::SchemaSettings::openapi3().with(|s| {
//             s.inline_subschemas = true;
//             s.meta_schema = None;
//         });
//         let sgen = schemars::r#gen::SchemaGenerator::new(settings);
//         // into_root_schema_for does not return a Result, assuming schemars handles internal errors
//         let root_schema = sgen.into_root_schema_for::<O>();

//         // Handle potential JSON serialization error
//         let mut json_schema = serde_json::to_value(root_schema).map_err(|e| {
//             AgentError::SchemaGenerationFailed(format!("Failed to serialize root schema: {}", e))
//         })?;

//         // Safely access as mutable object and remove title if it exists
//         match json_schema.as_object_mut() {
//             Some(obj) => {
//                 obj.remove("title");
//             }
//             None => {
//                 return Err(AgentError::SchemaGenerationFailed(
//                     "Generated schema root is not a JSON object, cannot remove 'title'".to_string(),
//                 ));
//             }
//         }

//         Ok(json_schema)
//     }

//     /// Gets a single, structured response of type `Output` from one call to the LLM.
//     /// Makes a single, direct call to the underlying model API, configured to return JSON
//     /// matching the `Output` type's schema. Parses the response into `Output`.
//     /// Returns `Result<Self::Output, AgentError>`.
//     pub async fn generate_typed(
//         &self,
//         content: impl Into<Vec<Content>> + Send,
//     ) -> Result<O, AgentError> {
//         // Prepare generation config, enforcing JSON output with the schema.
//         let mut generation_config = self.agent.generation_config();
//         generation_config.response_mime_type = Some("application/json".to_string());
//         generation_config.response_schema = Some(self.generate_schema()?);

//         // Build and send the request
//         let response = self
//             .agent
//             .gemini
//             .generate_content()
//             .maybe_system_instruction(self.agent.instructions())
//             .model(self.agent.model())
//             .content_list(content.into())
//             .generation_config(generation_config)
//             // TypedAgent methods typically don't use tools directly in the *final* typed call,
//             // relying instead on the LLM adhering to the JSON schema.
//             // Tools might be used in the `execute_typed` flow before this final call.
//             .build()
//             .send()
//             .await
//             .map_err(AgentError::ApiError)?; // Convert API comms error

//         // --- Safely extract and parse the JSON response ---
//         let first_candidate = response.candidates.first().ok_or_else(|| {
//             GeminiRequestError::UnexpectedResponse("API response missing candidate".into())
//         })?;

//         let first_part = first_candidate.content.parts().first().ok_or_else(|| {
//             GeminiRequestError::UnexpectedResponse("Candidate content missing parts".into())
//         })?;

//         let text_content = first_part.as_text().ok_or_else(|| {
//             GeminiRequestError::UnexpectedResponse(format!(
//                 "Expected text part containing JSON, found other type: {first_part:?}",
//             ))
//         })?;

//         // Parse the text content as JSON into the target type `Self::Output`.
//         serde_json::from_str(text_content).map_err(|e| AgentError::ResponseParsingFailed {
//             source: e,
//             response_text: text_content.to_string(),
//         }) // More specific agent error
//     }

//     /// Executes the agent process, potentially involving multiple turns (LLM calls) and tool calls,
//     /// with the final goal of producing a structured response of type `Output`.
//     /// If the agent has tools, it uses the multi-turn `execute` method first, then makes a final
//     /// call with `get_typed_response` to get the structured output. If no tools are present,
//     /// it directly calls `get_typed_response`.
//     /// Returns `Result<Self::Output, AgentError>`.
//     pub async fn execute_typed(
//         &self,
//         content: impl Into<Vec<Content>> + Send,
//     ) -> Result<O, AgentError> {
//         let mut content = content.into();
//         if self.tools.is_none() {
//             // No tools, directly ask for the typed response in a single turn.
//             self.generate_typed(content).await
//         } else {
//             // Has tools, run the multi-turn execution first.
//             let response = self.execute(content.clone()).await?;
//             // Take the final response content from the multi-turn execution...
//             content.extend(response.content_owned());

//             self.generate_typed(content).await
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use futures_util::StreamExt;

//     use crate::{
//         Gemini, Model,
//         agent::{GeminiAgent, GeminiTypedAgent, events::AgentEvent},
//         generate_content::content::Content,
//         tool::{FunctionCallError, Tool, ToolBox},
//     };

//     // Import the necessary builders and traits
//     use super::Agent;

//     #[derive(Debug, schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
//     struct CalculatorToolInput {
//         a: f64,
//         b: f64,
//     }
//     #[derive(Default, Clone)]
//     struct CalculatorTool;

//     impl Tool for CalculatorTool {
//         type Input = CalculatorToolInput;
//         type Error = FunctionCallError;

//         fn name(&self) -> &'static str {
//             "CalculatorTool"
//         }
//         fn description(&self) -> Option<&'static str> {
//             Some("Simple adding tool")
//         }
//         async fn invoke(&self, input: Self::Input) -> Result<Content, Self::Error> {
//             dbg!(&input);
//             Ok(Content::function_response(
//                 self.name(),
//                 serde_json::json!({ "result": input.a + input.b }),
//             ).unwrap()) // Return JSON as function response often expects
//         }
//     }

//     #[derive(Debug, Clone)]
//     struct CalculatorAgent; // Removed tools field as it's managed by the wrapper

//     impl Agent for CalculatorAgent {
//         fn name(&self) -> &'static str {
//             "CalculatorAgent"
//         }
//         // fn description(&self) -> Option<&str> {
//         //     Some("An agent that performs calculator operations")
//         // }
//         fn instructions(&self) -> Option<Content> {
//             Some(Content::text(
//                 "You are a calculator. Use the available tools to perform calculations based on the user query.".to_string(),
//             ))
//         }
//         fn model(&self) -> Model {
//             Model::Gemini20Flash // Use default or a specific capable model like Flash / Pro
//         }
//     }

//     #[derive(Debug, schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
//     #[serde(rename_all = "camelCase")] // Match Gemini Function Calling expectations
//     pub struct CalculationResult {
//         pub final_answer: f64, // Field name likely guided by LLM instruction
//     }

//     // Commented out OrchestratorAgent parts as they were incomplete and not used in fixed tests
//     // #[derive(Debug, Clone)]
//     // struct OrchestratorAgent {
//     //     tools: crate::tool::ToolBox,
//     // }
//     // ... (rest of OrchestratorAgent implementation) ...

//     // Helper to create Gemini client from env var
//     fn create_gemini() -> Gemini {
//         let api_key = std::env::var("GOOGLE_AI_API_KEY")
//             .expect("GOOGLE_AI_API_KEY environment variable not set");
//         Gemini::new(api_key)
//     }

//     #[tokio::test]
//     #[ignore] // Ignored by default to avoid making API calls unless explicitly run
//     async fn test_run_simple_agent() {
//         let gemini = create_gemini();
//         let tools = ToolBox::builder().tool(CalculatorTool).build();
//         let agent_impl = CalculatorAgent; // Create the agent implementation (no state needed here)
//         let agent_executor = GeminiAgent::builder() // Use the GeminiAgent builder
//             .gemini(gemini.clone())
//             .agent(agent_impl) // Pass the agent implementation
//             .tools(tools) // Pass the tools to the wrapper
//             .build();

//         let content = Content::text("What is 1 plus 2? Use the calculator tool.");

//         // Use execute with the executor
//         let resp = agent_executor.execute(content).await;
//         dbg!(&resp);
//         match resp {
//             Ok(r) => {
//                 println!("Simple Agent Response: {:?}", r);
//                 // Add assertions based on expected outcome (e.g., text contains "3")
//                 let response_text = r.content()[0].parts[0].as_text().unwrap();
//                 assert!(
//                     response_text.contains('3'),
//                     "Response should contain the result '3'"
//                 );
//             }
//             Err(e) => panic!("Simple Agent test failed: {:?}", e),
//         }
//     }

//     #[tokio::test]
//     #[ignore] // Ignored by default to avoid making API calls unless explicitly run
//     async fn test_run_stream_simple_agent() {
//         let gemini = create_gemini();
//         let tools = ToolBox::builder().tool(CalculatorTool).build();
//         let agent_impl = CalculatorAgent;
//         let agent_executor = GeminiAgent::builder()
//             .gemini(gemini.clone())
//             .agent(agent_impl)
//             .tools(tools)
//             .build();

//         let content = Content::text("Add 1 and 2 using the tool provided.");

//         // Use stream_events with the executor
//         let mut stream = agent_executor.stream_events(content);
//         let mut event_count = 0;
//         let mut saw_finish = false;
//         while let Some(event_result) = stream.next().await {
//             match event_result {
//                 Ok(event) => {
//                     println!("Stream Event: {:#?}", event);
//                     if matches!(event, AgentEvent::AgentFinish) {
//                         saw_finish = true;
//                     }
//                     event_count += 1;
//                 }
//                 Err(e) => panic!("Stream event error: {:?}", e),
//             }
//         }
//         assert!(event_count > 1, "Should receive multiple events"); // Start, Response(s), End, Finish
//         assert!(saw_finish, "Should receive AgentFinish event");
//     }

//     #[tokio::test]
//     #[ignore] // Ignored by default to avoid making API calls unless explicitly run
//     async fn test_simple_typed_agent_once() {
//         let gemini = create_gemini();
//         // Typed agent generation often works best *without* tools enabled for the final generation step,
//         // relying on the schema constraint. Tools are used in the multi-step execute_typed.
//         // let tools = crate::tool::ToolBox::builder().tool(CalculatorTool{}).build();
//         let agent_impl = CalculatorAgent;
//         let typed_agent_executor = GeminiAgent::builder() // Specify Output type
//             .gemini(gemini.clone())
//             .agent(agent_impl)
//             // .tools(tools) // <-- Typically omit tools for direct generate_typed
//             .typed::<CalculationResult>();

//         let message =
//             Content::text("Output the result of 2 + 2 as JSON matching the required schema.");
//         // Use generate_typed with the typed executor
//         let resp = typed_agent_executor.generate_typed(message).await;
//         match resp {
//             Ok(r) => {
//                 println!("Simple Typed Agent Response: {:?}", r);
//                 // 'r' is the parsed CalculationResult
//                 assert!(
//                     (r.final_answer - 4.0).abs() < f64::EPSILON,
//                     "Expected final_answer to be 4.0, but got {}",
//                     r.final_answer
//                 );
//             }
//             Err(e) => panic!("Simple Typed Agent test failed: {:?}", e),
//         }
//     }

//     #[tokio::test]
//     #[ignore] // Ignored by default to avoid making API calls unless explicitly run
//     async fn test_simple_typed_agent_run() {
//         let gemini = create_gemini();
//         // For execute_typed, we *do* provide tools for the initial execution steps
//         let tools = crate::tool::ToolBox::builder().tool(CalculatorTool).build();
//         let typed_agent_executor = GeminiAgent::builder() // Specify Output type
//             .gemini(gemini.clone())
//             .agent(CalculatorAgent)
//             .tools(tools) // Provide tools for the initial execution steps
//             .typed::<CalculationResult>();

//         let message = Content::text("Use the calculator tool to find 2 + 2");

//         // Use execute_typed with the executor. Needs .await because it returns an impl Future.
//         let resp = typed_agent_executor.execute_typed(message).await;
//         match resp {
//             Ok(r) => {
//                 println!("Simple Typed Agent Response (Run): {:?}", r);
//                 // 'r' is the parsed CalculationResult after potential tool use
//                 assert!(
//                     (r.final_answer - 4.0).abs() < f64::EPSILON,
//                     "Expected final_answer to be 4.0, but got {}",
//                     r.final_answer
//                 );
//             }
//             Err(e) => panic!("Simple Typed Agent (Run) test failed: {:?}", e),
//         }
//     }

//     // // Commented out previous test attempts and orchestrator tests
//     // // ...
// }
