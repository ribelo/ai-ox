//! Multi-agent feedback workflow test using dependency injection.
//!
//! This test demonstrates the proper use of the new dependency injection workflow system.
//! The workflow implements an email feedback loop with three nodes:
//! 1. WriteEmailNode - Generates initial email draft
//! 2. FeedbackNode - Reviews email and decides if approved or needs rewrite
//! 3. RewriteEmailNode - Rewrites email based on feedback
//!
//! Key features demonstrated:
//! - Clean separation between dependencies (agents) and mutable state
//! - Simple node structs with no agent fields
//! - Dependency injection via RunContext.deps
//! - Type-safe workflow with proper error handling

use std::sync::Arc;

use ai_ox::{
    agent::Agent,
    content::{
        message::{Message, MessageRole},
        part::Part,
    },
    workflow::{Next, Node, RunContext, Workflow, WorkflowError},
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod common;

// Data models for the workflow
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct User {
    name: String,
    email: String,
    interests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Email {
    subject: String,
    body: String,
}

/// Status of the email review
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
enum ReviewStatus {
    #[serde(rename = "approved")]
    Approved,
    #[serde(rename = "needs_rewrite")]
    NeedsRewrite,
}

/// Represents the structured response from the Feedback Agent.
///
/// This struct indicates whether an email draft is approved or if it requires
/// further revisions, along with any specific feedback.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct FeedbackResponse {
    /// Status of the review
    status: ReviewStatus,
    /// Optional feedback message when status is "needs_rewrite"
    #[serde(skip_serializing_if = "Option::is_none")]
    feedback: Option<String>,
}

// Dependency injection: shared, stateless dependencies
#[derive(Debug, Clone)]
struct WorkflowDeps {
    writer_agent: Agent,
    feedback_agent: Agent,
}

// Workflow state: mutable state that changes during execution
#[derive(Debug, Clone)]
struct WorkflowState {
    user: User,
    email: Email,
}

// Node structs: simple, empty structs that contain no fields
#[derive(Debug, Clone)]
struct WriteEmailNode;

#[derive(Debug, Clone)]
struct FeedbackNode;

#[derive(Debug, Clone)]
struct RewriteEmailNode {
    feedback: String, // Only field needed - the feedback text from the review
}

// Node implementations using dependency injection pattern
#[async_trait]
impl Node<WorkflowState, WorkflowDeps, Email> for WriteEmailNode {
    async fn run(
        &self,
        context: &RunContext<WorkflowState, WorkflowDeps>,
    ) -> Result<Next<WorkflowState, WorkflowDeps, Email>, WorkflowError> {
        let state = context.state.lock().await;
        let user = state.user.clone();
        drop(state);

        let prompt = format!(
            "You are an expert copywriter. Write a welcome email to the user, incorporating their interests. \
            User: {} ({})\nInterests: {}\n\
            Respond with a JSON object containing 'subject' and 'body'.",
            user.name,
            user.email,
            user.interests.join(", ")
        );

        let message = Message::new(MessageRole::User, vec![Part::Text { text: prompt }]);

        // Access writer agent from dependencies
        match context
            .deps
            .writer_agent
            .generate_typed::<Email>(vec![message])
            .await
        {
            Ok(response) => {
                let email = response.data;
                // Update state with generated email
                let mut state = context.state.lock().await;
                state.email = email;
                drop(state);

                // Continue to feedback node
                Ok(FeedbackNode.into())
            }
            Err(e) => Err(WorkflowError::node_execution_failed(e)),
        }
    }
}

#[async_trait]
impl Node<WorkflowState, WorkflowDeps, Email> for FeedbackNode {
    async fn run(
        &self,
        context: &RunContext<WorkflowState, WorkflowDeps>,
    ) -> Result<Next<WorkflowState, WorkflowDeps, Email>, WorkflowError> {
        let state = context.state.lock().await;
        let user = state.user.clone();
        let email = state.email.clone();
        drop(state);

        let prompt = format!(
            "You are a senior editor. Review the email. If it properly incorporates the user's interests, \
            respond with status 'approved'. If not, respond with status 'needs_rewrite' and provide specific feedback.\n\
            User interests: {}\nEmail subject: {}\nEmail body: {}\n\
            Respond with a JSON object containing 'status' and optionally 'feedback'.",
            user.interests.join(", "),
            email.subject,
            email.body
        );

        let message = Message::new(MessageRole::User, vec![Part::Text { text: prompt }]);

        // Access feedback agent from dependencies
        match context
            .deps
            .feedback_agent
            .generate_typed::<FeedbackResponse>(vec![message])
            .await
        {
            Ok(response) => match response.data.status {
                ReviewStatus::Approved => Ok(Next::End(email)),
                ReviewStatus::NeedsRewrite => {
                    let feedback = response.data.feedback.unwrap_or_else(|| {
                        "Please improve the email to better incorporate the user's interests".to_string()
                    });
                    // Continue to rewrite node with feedback
                    Ok(RewriteEmailNode { feedback }.into())
                }
            },
            Err(e) => Err(WorkflowError::node_execution_failed(e)),
        }
    }
}

#[async_trait]
impl Node<WorkflowState, WorkflowDeps, Email> for RewriteEmailNode {
    async fn run(
        &self,
        context: &RunContext<WorkflowState, WorkflowDeps>,
    ) -> Result<Next<WorkflowState, WorkflowDeps, Email>, WorkflowError> {
        let state = context.state.lock().await;
        let user = state.user.clone();
        let current_email = state.email.clone();
        drop(state);

        let prompt = format!(
            "You are an expert copywriter. Rewrite the welcome email based on the feedback provided. \
            User: {} ({})\nInterests: {}\n\
            Current email subject: {}\nCurrent email body: {}\n\
            Feedback: {}\n\
            Respond with a JSON object containing 'subject' and 'body'.",
            user.name,
            user.email,
            user.interests.join(", "),
            current_email.subject,
            current_email.body,
            self.feedback
        );

        let message = Message::new(MessageRole::User, vec![Part::Text { text: prompt }]);

        // Access writer agent from dependencies
        match context
            .deps
            .writer_agent
            .generate_typed::<Email>(vec![message])
            .await
        {
            Ok(response) => {
                let email = response.data;
                // Update state with rewritten email
                let mut state = context.state.lock().await;
                state.email = email;
                drop(state);

                // Go back to feedback node for another review
                Ok(FeedbackNode.into())
            }
            Err(e) => Err(WorkflowError::node_execution_failed(e)),
        }
    }
}

#[tokio::test]
async fn multi_agent_feedback_workflow() {
    let models = common::get_available_models().await;
    if models.is_empty() {
        println!("No models available for testing. Skipping multi-agent feedback test.");
        return;
    }

    let model = Arc::from(models.into_iter().next().unwrap());

    // Create agents
    let writer_agent = Agent::builder()
        .model(Arc::clone(&model))
        .system_instruction(
            "You are an expert copywriter. Write a welcome email to the user, incorporating their interests. \
            Respond with a JSON object containing 'subject' and 'body'."
        )
        .build();

    let feedback_agent = Agent::builder()
        .model(model)
        .system_instruction(
            "You are a senior editor. Review the email. If it properly incorporates the user's interests, \
            respond with status 'approved'. If not, respond with status 'needs_rewrite' and provide specific feedback. \
            Respond with a JSON object containing 'status' and optionally 'feedback'."
        )
        .build();

    // Create dependencies struct
    let deps = WorkflowDeps {
        writer_agent,
        feedback_agent,
    };

    // Create initial state
    let user = User {
        name: "Alex Chen".to_string(),
        email: "alex.chen@example.com".to_string(),
        interests: vec![
            "Rust programming".to_string(),
            "AI and machine learning".to_string(),
        ],
    };

    let initial_state = WorkflowState {
        user: user.clone(),
        email: Email {
            subject: String::new(),
            body: String::new(),
        },
    };

    // Create and run workflow using dependency injection
    let workflow = Workflow::new(WriteEmailNode, initial_state, deps);

    println!("🚀 Starting multi-agent feedback workflow with dependency injection...");

    match workflow.run().await {
        Ok(final_email) => {
            println!("✅ Multi-agent feedback workflow completed successfully!");
            println!("📧 Final email generated:");
            println!("Subject: {}", final_email.subject);
            println!("Body: {}", final_email.body);

            // Verify the email meets our requirements
            assert!(
                !final_email.subject.is_empty(),
                "Email subject should not be empty"
            );
            assert!(
                !final_email.body.is_empty(),
                "Email body should not be empty"
            );

            let email_content =
                format!("{} {}", final_email.subject, final_email.body).to_lowercase();
            let has_rust = email_content.contains("rust");
            let has_ai = email_content.contains("ai") || email_content.contains("machine learning");

            assert!(
                has_rust && has_ai,
                "Email should contain references to both Rust programming and AI/machine learning. \
                Subject: {}, Body: {}",
                final_email.subject,
                final_email.body
            );

            println!("✅ Email validation passed - contains user interests");
            println!("✅ Dependency injection workflow pattern validated successfully!");
        }
        Err(e) => {
            panic!("Multi-agent feedback workflow failed: {:?}", e);
        }
    }
}
