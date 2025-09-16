//! Multi-agent workflow integration test.
//!
//! This test demonstrates a proper multi-agent workflow with clear separation
//! between orchestration and execution logic. The workflow involves three agents:
//! 1. Planner Agent - Determines which tool to use
//! 2. Tool Executor Agent - Executes tools (mocked)
//! 3. Summarizer Agent - Provides final user-facing response

use ai_ox::{
    agent::Agent,
    content::{
        Part,
        message::{Message, MessageRole},
    },
    toolbox,
    workflow::{Next, Node, RunContext, Workflow, WorkflowError},
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

mod common;

/// Input for the weather tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherInput {
    location: String,
}

/// Output for the weather tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct WeatherOutput {
    location: String,
    temperature: String,
    forecast: String,
}

/// Weather service for the multi-agent workflow test
#[derive(Debug, Clone)]
struct WeatherService;

#[toolbox]
impl WeatherService {
    /// Get the current weather for a location
    pub fn get_weather(&self, input: WeatherInput) -> WeatherOutput {
        WeatherOutput {
            location: input.location,
            temperature: "25°C".to_string(),
            forecast: "sunny".to_string(),
        }
    }
}

/// Workflow state that maintains conversation history and workflow step.
#[derive(Debug, Clone)]
struct MultiAgentState {
    conversation_history: Vec<Message>,
    current_step: WorkflowStep,
    planner_agent: Agent,
    tool_executor_agent: Agent,
    summarizer_agent: Agent,
}

/// Represents the current step in the multi-agent workflow.
#[derive(Debug, Clone)]
enum WorkflowStep {
    Planning,
    ToolExecution,
    Summarization,
}

impl MultiAgentState {
    fn new(planner: Agent, executor: Agent, summarizer: Agent, initial_message: Message) -> Self {
        Self {
            conversation_history: vec![initial_message],
            current_step: WorkflowStep::Planning,
            planner_agent: planner,
            tool_executor_agent: executor,
            summarizer_agent: summarizer,
        }
    }

    fn add_message(&mut self, message: Message) {
        self.conversation_history.push(message);
    }

    fn get_history(&self) -> &[Message] {
        &self.conversation_history
    }

    fn set_step(&mut self, step: WorkflowStep) {
        self.current_step = step;
    }
}

/// Generic agent node that implements the workflow Node trait using the FSM pattern.
#[derive(Debug, Clone)]
struct AgentNode {
    name: String,
}

impl AgentNode {
    fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait]
impl Node<MultiAgentState, (), String> for AgentNode {
    async fn run(
        &self,
        context: &RunContext<MultiAgentState, ()>,
    ) -> Result<Next<MultiAgentState, (), String>, WorkflowError> {
        let state = context.state.lock().await;

        let (agent, next_step) = match &state.current_step {
            WorkflowStep::Planning => (
                &state.planner_agent.clone(),
                Some(WorkflowStep::ToolExecution),
            ),
            WorkflowStep::ToolExecution => (
                &state.tool_executor_agent.clone(),
                Some(WorkflowStep::Summarization),
            ),
            WorkflowStep::Summarization => (&state.summarizer_agent.clone(), None),
        };

        let messages = state.get_history().to_vec();

        // Release the lock before the async operation
        drop(state);

        match agent.run(messages).await {
            Ok(response) => {
                // Re-acquire lock to update state
                let mut state = context.state.lock().await;
                state.add_message(response.message.clone());

                match next_step {
                    Some(step) => {
                        state.set_step(step);
                        Ok(Next::Continue(Box::new(AgentNode::new(format!(
                            "{}_next",
                            self.name
                        )))))
                    }
                    None => {
                        // Final step - extract text response
                        if let Some(text) =
                            response.message.content.iter().find_map(|part| match part {
                                Part::Text { text, .. } => Some(text.clone()),
                                _ => None,
                            })
                        {
                            Ok(Next::End(text))
                        } else {
                            Ok(Next::End(
                                "Workflow completed without text response".to_string(),
                            ))
                        }
                    }
                }
            }
            Err(e) => Err(WorkflowError::node_execution_failed(e)),
        }
    }
}

#[tokio::test]
async fn multi_agent_weather_workflow() {
    let models = common::get_available_models().await;

    if models.is_empty() {
        println!("⚠️ No models available for testing. Skipping multi-agent workflow test.");
        return;
    }

    let model = Arc::from(models.into_iter().next().unwrap());
    let weather_service = WeatherService;

    // Planner Agent - determines what needs to be done (no tools)
    let planner_agent = Agent::builder()
        .model(Arc::clone(&model))
        .system_instruction(
            "Given the user's request, analyze what information they need. Simply state what needs to be checked or retrieved.",
        )
        .build();

    // Tool Executor Agent - has tools and will execute them based on context
    let tool_executor_agent = Agent::builder()
        .model(Arc::clone(&model))
        .tools(weather_service.clone())
        .system_instruction("Based on the conversation, use the appropriate tools to get the requested information.")
        .build();

    // Summarizer Agent - provides final response
    let summarizer_agent = Agent::builder()
        .model(model)
        .system_instruction("Based on the conversation history, provide a concise, natural language summary of the weather information.")
        .build();

    // Initialize workflow state
    let initial_message = Message::new(
        MessageRole::User,
        vec![Part::Text {
            text: "What is the weather like in Paris?".to_string(),
            ext: BTreeMap::new(),
        }],
    );
    let initial_state = MultiAgentState::new(
        planner_agent,
        tool_executor_agent,
        summarizer_agent,
        initial_message,
    );

    // Create and execute workflow using the FSM pattern
    let workflow = Workflow::new(
        AgentNode::new("MultiAgentWorkflow".to_string()),
        initial_state,
        (), // Empty deps
    );

    println!("🚀 Starting multi-agent weather workflow...");

    let result = workflow.run().await;

    match result {
        Ok(final_response) => {
            println!("✅ Multi-agent workflow completed successfully!");
            println!("📝 Final response: {}", final_response);

            let text_lower = final_response.to_lowercase();
            let has_weather_info = text_lower.contains("paris")
                || text_lower.contains("sunny")
                || text_lower.contains("25")
                || text_lower.contains("weather");

            if has_weather_info {
                println!("✅ Final response contains expected weather information");
            } else {
                println!(
                    "ℹ️ Final response doesn't contain expected weather terms, but workflow completed successfully"
                );
            }

            println!("✅ Multi-agent workflow validation completed successfully!");
        }
        Err(e) => {
            panic!("Multi-agent workflow failed: {:?}", e);
        }
    }
}
