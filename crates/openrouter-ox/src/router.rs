// use std::{
//     collections::HashMap,
//     sync::{Arc, RwLock},
// };

// use async_trait::async_trait;
// use bon::Builder;
// use derive_more::Deref;
// use indoc::indoc;
// use schemars::JsonSchema;
// use serde::{Deserialize, Serialize};

// use crate::{
//     agent::{Agent, AnyAgent, BaseAgent},
//     message::{AssistantMessage, Message, Messages, SystemMessage, UserMessage},
//     request::ResponseFormat,
//     response::ChatCompletionResponse,
//     tool::{Tool, ToolBox, ToolsList},
//     ApiRequestError, OpenRouter,
// };

// static DEFAULT_ROUTER_SYSTEM_PROMPT: &str = indoc! {"
//     You are a specialized router AI. Your role is to:
//     - Evaluate and score agents for handling queries
//     - Assign scores between 0.0 and 1.0 based on suitability

//     You must consider:
//     - Agent expertise and capabilities
//     - Query requirements
//     - Relative strengths between agents

//     Score Guidelines:
//     - 1.0: Perfect match
//     - 0.0: Completely unsuitable
//     - Intermediate scores reflect partial matches
// "};

// static DEFAULT_ROUTER_PROMPT: &str = indoc! {"
//     Review request and agents:

//     <agents_list>
//     {AGENTS_LIST}
//     </agents_list>

//     Score Guidelines:
//     - 0.0-0.2: Unsuitable
//     - 0.3-0.4: Minimal fit
//     - 0.5-0.6: Moderate
//     - 0.7-0.8: Strong
//     - 0.9-1.0: Excellent

//     Important: Each agent MUST receive a unique score - no two agents can have the exact same score.

//     Score all agents relative to requirements.
// "};

// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Deref)]
// pub struct AgentScores {
//     pub scores: Vec<AgentScore>,
// }

// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
// pub struct AgentScore {
//     pub agent_name: String,
//     pub score: f64,
// }

// #[derive(Clone, Default, Builder)]
// pub struct Agents {
//     #[builder(field)]
//     agents: Arc<RwLock<HashMap<String, Arc<dyn AnyAgent>>>>,
// }

// impl std::fmt::Debug for Agents {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let agents = self.agents.read().unwrap();
//         let agents_map: HashMap<&String, &str> =
//             agents.iter().map(|(k, _)| (k, "Arc<AnyAgent>")).collect();
//         f.debug_struct("Agents")
//             .field("agents", &agents_map)
//             .finish()
//     }
// }

// impl<S: agents_builder::State> AgentsBuilder<S> {
//     pub fn agent<T: Agent + 'static>(self, tool: T) -> Self {
//         let name = tool.name().to_string();
//         self.agents.write().unwrap().insert(name, Arc::new(tool));
//         self
//     }
// }

// impl Agents {
//     pub fn as_list(&self) -> AgentsList {
//         let metadata = self
//             .agents
//             .read()
//             .unwrap()
//             .iter()
//             .map(|(name, agent)| AgentMetadata {
//                 name: name.clone(),
//                 description: agent.description().to_string(),
//             })
//             .collect();
//         AgentsList(metadata)
//     }
//     pub fn get(&self, name: &str) -> Option<Arc<dyn AnyAgent>> {
//         self.agents.read().unwrap().get(name).cloned()
//     }
//     pub async fn route(
//         &self,
//         agent_name: &str,
//         messages: impl Into<Messages> + Send,
//     ) -> Result<ChatCompletionResponse, ApiRequestError> {
//         let agent = self.get(agent_name).unwrap().clone();
//         agent.send_any(messages.into()).await
//     }
// }

// impl Serialize for Agents {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::ser::Serializer,
//     {
//         self.as_list().serialize(serializer)
//     }
// }

// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
// pub struct AgentMetadata {
//     name: String,
//     description: String,
// }

// #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
// pub struct AgentsList(Vec<AgentMetadata>);

// #[async_trait]
// pub trait Router: Send + Sync + Clone {
//     fn openrouter(&self) -> &OpenRouter;
//     fn name(&self) -> &str;
//     fn system_prompt(&self) -> &str {
//         DEFAULT_ROUTER_SYSTEM_PROMPT
//     }
//     fn model(&self) -> &str;
//     fn max_tokens(&self) -> Option<u32> {
//         None
//     }
//     fn stop_sequences(&self) -> Option<&Vec<String>> {
//         None
//     }
//     fn temperature(&self) -> Option<f64> {
//         None
//     }
//     fn top_p(&self) -> Option<f64> {
//         None
//     }
//     fn top_k(&self) -> Option<usize> {
//         None
//     }
//     fn agents(&self) -> &Agents;
//     fn get_agent(&self, name: &str) -> Option<Arc<dyn AnyAgent>> {
//         self.agents().get(name)
//     }
//     async fn choose(
//         &self,
//         messages: impl Into<Messages> + Send,
//     ) -> Result<AgentScores, ApiRequestError> {
//         // Convert agents to JSON string
//         let agents_list = serde_json::to_string_pretty(&self.agents())?;

//         // Build messages vector
//         let mut messages = messages.into();
//         messages.insert(0, SystemMessage::new(vec![self.system_prompt()]));
//         let routing_prompt = DEFAULT_ROUTER_PROMPT.replace("{AGENTS_LIST}", &agents_list);
//         messages.push(UserMessage::new(vec![routing_prompt]));

//         // Send request to OpenRouter
//         let response = self
//             .openrouter()
//             .chat_completion()
//             .model(self.model())
//             .messages(messages)
//             .maybe_max_tokens(self.max_tokens())
//             .maybe_temperature(self.temperature())
//             .maybe_top_p(self.top_p())
//             .maybe_stop(self.stop_sequences().cloned())
//             .response_format::<AgentScores>()
//             .build()
//             .send()
//             .await?;

//         let scores: AgentScores = serde_json::from_str(
//             &response.choices[0].message.content[0]
//                 .as_text()
//                 .unwrap()
//                 .text,
//         )?;
//         Ok(scores)
//     }

//     async fn send(
//         &self,
//         agent_name: &str,
//         messages: impl Into<Messages> + Send,
//     ) -> Result<ChatCompletionResponse, ApiRequestError> {
//         self.agents().route(agent_name, messages).await
//     }

//     async fn route(
//         &self,
//         messages: impl Into<Messages> + Send,
//     ) -> Result<ChatCompletionResponse, ApiRequestError> {
//         // Get agent scores and find best match
//         let messages = messages.into();
//         let scores = self.choose(messages.clone()).await?;
//         let best_agent = scores
//             .scores
//             .iter()
//             .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
//             .expect("No agents found");

//         self.agents().route(&best_agent.agent_name, messages).await
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[tokio::test]
//     async fn test_router() {
//         #[derive(Debug, Clone)]
//         pub struct AdderAgent {
//             openrouter: OpenRouter,
//         }
//         impl BaseAgent for AdderAgent {
//             fn openrouter(&self) -> &OpenRouter {
//                 &self.openrouter
//             }

//             fn description(&self) -> &str {
//                 "Adds two numbers together"
//             }

//             fn instructions(&self) -> Option<String> {
//                 Some("Given two numbers, add them together".to_string())
//             }

//             fn model(&self) -> &str {
//                 "openai/gpt-4o"
//             }
//         }
//         #[derive(Debug, Clone)]
//         pub struct SubtracterAgent {
//             openrouter: OpenRouter,
//         }
//         impl BaseAgent for SubtracterAgent {
//             fn openrouter(&self) -> &OpenRouter {
//                 &self.openrouter
//             }

//             fn description(&self) -> &str {
//                 "Subtracts two numbers from each other"
//             }

//             fn instructions(&self) -> Option<String> {
//                 Some(
//                     "Given two numbers, subtract the second number from the first number"
//                         .to_string(),
//                 )
//             }

//             fn model(&self) -> &str {
//                 "openai/gpt-4o"
//             }
//         }
//         #[derive(Debug, Clone)]
//         pub struct CalculatorAgent {
//             openrouter: OpenRouter,
//         }
//         impl BaseAgent for CalculatorAgent {
//             fn openrouter(&self) -> &OpenRouter {
//                 &self.openrouter
//             }

//             fn description(&self) -> &str {
//                 "Performs basic calculator operations including addition, subtraction, multiplication and division"
//             }

//             fn instructions(&self) -> Option<String> {
//                 Some("You are a calculator agent capable of performing basic math operations. When given a question about calculations, provide a clear, step-by-step breakdown of the solution and the final result. Focus on addition, subtraction, multiplication, and division.".to_string())
//             }

//             fn model(&self) -> &str {
//                 "openai/gpt-4o"
//             }
//         }
//         #[derive(Debug, Clone)]
//         pub struct SimpleRouter {
//             agents: Agents,
//             openrouter: OpenRouter,
//         }
//         impl Router for SimpleRouter {
//             fn openrouter(&self) -> &OpenRouter {
//                 &self.openrouter
//             }

//             fn name(&self) -> &str {
//                 "SimpleRouter"
//             }

//             fn system_prompt(&self) -> &str {
//                 "This is a simple router"
//             }

//             fn model(&self) -> &str {
//                 "openai/gpt-4o"
//             }

//             fn agents(&self) -> &Agents {
//                 &self.agents
//             }
//         }

//         let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
//         let openrouter = OpenRouter::builder().api_key(api_key).build();
//         let adder_agent = AdderAgent {
//             openrouter: openrouter.clone(),
//         };
//         let subtracter_agent = SubtracterAgent {
//             openrouter: openrouter.clone(),
//         };
//         let calculator_agent = CalculatorAgent {
//             openrouter: openrouter.clone(),
//         };
//         let agents = Agents::builder()
//             .agent(adder_agent)
//             .agent(subtracter_agent)
//             .agent(calculator_agent)
//             .build();
//         let router = SimpleRouter {
//             agents: agents,
//             openrouter,
//         };
//         let messages: Messages = UserMessage::new(vec!["What is 2 + 2?"]).into();
//         let resp = router.choose(messages).await.unwrap();
//         dbg!(resp);
//     }
// }
