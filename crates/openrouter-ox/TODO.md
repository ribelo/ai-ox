Okay, let's re-introduce the `AgentExecutor` concept to centralize the execution logic (API calls, looping, streaming) while keeping the agent traits focused on defining the agent's configuration and capabilities. We will *not* re-introduce the `RunContext` for now.

The core idea is:

1.  `AgentExecutor` will hold the `OpenRouter` client.
2.  `AgentExecutor` methods will take an `agent` instance (implementing `BaseAgent`, `Agent`, or `TypedAgent`) as an argument.
3.  `AgentExecutor` will use the methods defined on the agent (`model()`, `tools()`, `instructions()`, `max_iterations()`, etc.) to perform the API calls and manage the execution flow.
4.  The default implementations containing the execution loops (`run`, `run_events`, `run_typed`) will be **removed** from the `Agent` and `TypedAgent` traits and **implemented** within the `AgentExecutor`.
5.  The `BaseAgent` trait's default `once` and `stream_once` implementations will also be removed, with the `AgentExecutor` handling these calls.

---

**Implementation Plan: Introduce AgentExecutor**

**1. Introduction & Objectives**

*   **1.1. Task Overview:** Introduce an `AgentExecutor` struct responsible for handling the execution flow of agents defined by the `BaseAgent`, `Agent`, and `TypedAgent` traits. This involves moving the API call logic, iteration loops, and streaming logic out of the default trait implementations and into the `AgentExecutor`.
*   **1.2. Goals & Success Criteria:**
    *   **Goal:** Create `src/agent/executor.rs` containing the `AgentExecutor` struct.
    *   **Success Criterion:** The `AgentExecutor` struct exists and holds an `OpenRouter` client instance.
    *   **Goal:** Move API call logic (single and streaming) from `BaseAgent` defaults to `AgentExecutor`.
    *   **Success Criterion:** `AgentExecutor` has `execute_once` and `stream_once` methods that take an `impl BaseAgent` and make API calls using the agent's configuration. Default implementations are removed from `BaseAgent`.
    *   **Goal:** Move iterative execution logic (`run`, `run_events`) from `Agent` defaults to `AgentExecutor`.
    *   **Success Criterion:** `AgentExecutor` has `execute_run` and `stream_run` methods that take an `impl Agent` and implement the iteration/tool-call loop. Default implementations are removed from `Agent`.
    *   **Goal:** Move typed execution logic (`once_typed`, `run_typed`) from `TypedAgent` defaults to `AgentExecutor`.
    *   **Success Criterion:** `AgentExecutor` has `execute_once_typed` and `execute_run_typed` methods that take an `impl TypedAgent`. Default implementations are removed from `TypedAgent`.
    *   **Goal:** Update existing tests to use the `AgentExecutor`.
    *   **Success Criterion:** Tests are refactored to instantiate an `AgentExecutor` and call its methods (e.g., `executor.execute_run(&agent, messages)`), and all tests pass.
    *   **Goal:** Ensure the project compiles and tests pass after the refactoring.
    *   **Success Criterion:** `cargo build` and `cargo test` execute without errors.
*   **1.3. Scope:**
    *   **Included:** Creating `AgentExecutor`, moving existing execution logic into it, removing default implementations from traits, updating tests to use the executor.
    *   **Not Included:** Adding new agent features, changing the core agent configuration logic within the traits, re-introducing `RunContext`, changing tool invocation logic within `ToolBox`.
*   **1.4. Technology Stack:** `Rust`, `Cargo`, `tokio`, `serde`, `async-trait`, `schemars`, `thiserror`, `bon`.

**2. Analysis & Requirements Elaboration**

*   **2.1. Functional Requirements:**
    *   The system must provide the same overall agent behaviors (`run`, `run_events`, `run_typed`) accessible via the `AgentExecutor`.
    *   The executor must correctly use the configuration provided by the specific agent instance (`model`, `tools`, `temperature`, `max_iterations`, etc.).
    *   Tool invocation must still function correctly, triggered by the executor's loop logic but using the `ToolBox` provided by the agent instance.
*   **2.2. Non-Functional Requirements:**
    *   **Maintainability:** Centralizing execution logic in the `AgentExecutor` should make it easier to modify or extend how agents are run (e.g., adding global pre/post-execution hooks later) without changing every agent implementation.
    *   **Clarity:** Separates the *definition* of an agent (traits) from the *execution* of an agent (executor).
*   **2.3. Constraints & Assumptions:**
    *   Constraint: Must integrate with the existing `agent` module structure.
    *   Assumption: The agent traits (`BaseAgent`, etc.) provide all necessary configuration details for the executor to function.
    *   Assumption: `ToolBox::invoke` remains the mechanism for executing tools, called by the executor.
*   **2.4. Open Questions / Areas for Clarification:** None currently anticipated.

**3. Proposed Design & Architecture**

*   **3.1. Solution Overview:** An `AgentExecutor` struct will be created, holding the `OpenRouter` client. Methods on the executor will orchestrate the agent lifecycle (API calls, loops, streaming), taking specific agent trait implementations as arguments to access configuration and capabilities (like `ToolBox`). Default method implementations containing execution logic will be removed from the `BaseAgent`, `Agent`, and `TypedAgent` traits.
*   **3.2. Component / Module Design:**
    *   `src/agent/executor.rs`: Defines `AgentExecutor` struct and its methods (`execute_once`, `stream_once`, `execute_run`, `stream_run`, `execute_once_typed`, `execute_run_typed`).
    *   `src/agent/traits.rs`: Trait definitions (`BaseAgent`, `Agent`, `TypedAgent`, `AnyAgent`) will have their default *execution* method implementations removed. Method signatures remain. `AgentInput` remains.
    *   `src/agent/simple_agent.rs` & `src/agent/typed_agent.rs`: No changes expected here, as they primarily implement the configuration methods of the traits.
    *   `src/agent/mod.rs`: Add `mod executor;` and `pub use executor::AgentExecutor;`. Update tests.
*   **3.3. Data Model Changes:** None.
*   **3.4. Key Design Considerations:**
    *   **Dependency:** `AgentExecutor` depends on `OpenRouter` and agent trait implementations. Agent implementations do *not* depend on `AgentExecutor`.
    *   **Trait Simplicity:** Traits become purer definitions of capability and configuration, leaving execution mechanics to the executor.

**4. Implementation Task Breakdown**

*   **4.1. Prerequisites:**
    *   Ensure the project is in the state achieved after the previous refactoring (files separated into `src/agent/`).
    *   Create a new feature branch.
*   **4.2. Implementation Steps:**
    ```markdown
    - [ ] **Define `AgentExecutor` Struct:**
        - [ ] Create `src/agent/executor.rs`.
        - [ ] Define the struct:
              ```rust
              // src/agent/executor.rs
              use crate::OpenRouter;

              #[derive(Debug, Clone)] // Clone might be useful
              pub struct AgentExecutor {
                  openrouter: OpenRouter,
              }

              impl AgentExecutor {
                  pub fn new(openrouter: OpenRouter) -> Self {
                      Self { openrouter }
                  }

                  // TODO: Add execution methods here
              }
              ```
        - [ ] In `src/agent/mod.rs`, add `mod executor;` and `pub use executor::AgentExecutor;`.

    - [ ] **Implement `execute_once` and `stream_once` in Executor:**
        - [ ] Copy the *logic* from the default `BaseAgent::once` implementation (in `src/agent/traits.rs`) into a new `async fn execute_once` method in `AgentExecutor`. This method should take `&self`, `agent: &A`, and `messages: impl Into<Messages> + Send` where `A: BaseAgent + ?Sized`. Use `agent.model()`, `agent.temperature()`, etc., and `self.openrouter`.
        - [ ] Copy the *logic* from the default `BaseAgent::stream_once` implementation into a new `fn stream_once` method in `AgentExecutor`. Adapt arguments similarly (`agent: &A`, `messages: impl Into<Messages> + Send`). Return the `Pin<Box<dyn Stream<...>>>`.
        - [ ] **Remove** the default `impl` blocks for `once` and `stream_once` from the `BaseAgent` trait definition in `src/agent/traits.rs`. Keep the method signatures in the trait definition.

    - [ ] **Implement `execute_run` in Executor:**
        - [ ] Copy the *logic* from the default `Agent::run` implementation (in `src/agent/traits.rs`) into a new `async fn execute_run` method in `AgentExecutor`. It should take `&self`, `agent: &A`, and `initial_messages: impl Into<Messages> + Send` where `A: Agent + ?Sized`.
        - [ ] Replace calls to `self.once(...)` within the copied logic with `self.execute_once(agent, ...)` .
        - [ ] Use `agent.tools()` and `agent.max_iterations()` where the original logic used `self.tools()` and `self.max_iterations()`.
        - [ ] **Remove** the default `impl` block for `run` from the `Agent` trait definition in `src/agent/traits.rs`. Keep the method signature.

    - [ ] **Implement `stream_run` in Executor:**
        - [ ] Copy the *logic* from the default `Agent::run_events` implementation into a new `fn stream_run` method in `AgentExecutor`. Adapt arguments (`agent: &A`, `initial_messages: impl Into<Messages> + Send`). Return the `Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>>...>>`.
        - [ ] Replace calls to `cloned_self.stream_once(...)` with `self.stream_once(agent, ...)` .
        - [ ] Use `agent.tools()`, `agent.max_iterations()`, etc., instead of `cloned_self.*()`. Adapt helper structs like `PartialToolCallsAggregator` as needed within the stream.
        - [ ] **Remove** the default `impl` block for `run_events` from the `Agent` trait definition. Keep the method signature (`run_events` should probably be renamed in the trait if it no longer *does* the run, perhaps to something indicating it *can* be run with events, or just removed if `impl Agent` is enough signal). *Decision:* Let's keep the signature `fn run_events(...)` in the trait for now, but its default impl is removed. The executor implements the actual streaming run.
        - [ ] *Self-Correction:* Rename the executor method to `stream_run` to avoid confusion with the (now default-impl-less) trait method `run_events`.

    - [ ] **Implement `execute_once_typed` in Executor:**
        - [ ] Copy the *logic* from the default `TypedAgent::once_typed` implementation into a new `async fn execute_once_typed` method in `AgentExecutor`. It takes `&self`, `agent: &A`, `messages: impl Into<Messages> + Send` where `A: TypedAgent + ?Sized`. It should return `Result<A::Output, AgentError>`.
        - [ ] Use `agent.model()`, `agent.tools()`, etc., and `self.openrouter`. The `response_format::<A::Output>()` call is crucial here.
        - [ ] **Remove** the default `impl` block for `once_typed` from the `TypedAgent` trait definition. Keep the method signature.

    - [ ] **Implement `execute_run_typed` in Executor:**
        - [ ] Copy the *logic* from the default `TypedAgent::run_typed` implementation into a new `async fn execute_run_typed` method in `AgentExecutor`. Adapt arguments (`agent: &A`, `initial_messages: impl Into<Messages> + Send`). Return `Result<A::Output, AgentError>`.
        - [ ] Replace calls to `self.once(...)` with `self.execute_once(agent, ...)` .
        - [ ] Replace the final call to `self.once_typed(...)` with `self.execute_once_typed(agent, ...)` .
        - [ ] Use `agent.tools()`, `agent.max_iterations()`, etc.
        - [ ] **Remove** the default `impl` block for `run_typed` from the `TypedAgent` trait definition. Keep the method signature.

    - [ ] **Update Tests:**
        - [ ] Go to `src/agent/mod.rs` tests.
        - [ ] In each test (`test_simple_agent`, `test_simple_typed_agent_once`, `test_simple_agent_run`, `test_simple_agent_run_events`, `test_agent_run_events`), instantiate the `AgentExecutor`: `let executor = AgentExecutor::new(openrouter.clone());` (or just `openrouter` if not cloning the agent).
        - [ ] Replace calls like `agent.run(message).await` with `executor.execute_run(&agent, message).await`.
        - [ ] Replace `agent.once_typed(message).await` with `executor.execute_once_typed(&agent, message).await`.
        - [ ] Replace `agent.run_events(message)` with `executor.stream_run(&agent, message)`.
        - [ ] Ensure the agent instance (`agent`) is passed by reference (`&agent`) to the executor methods.
        - [ ] Adjust test setup if necessary (e.g., how `openrouter` client is created and passed).

    - [ ] **Build, Test, Lint:**
        - [ ] Run `cargo check` frequently during the process.
        - [ ] Run `cargo build` to fix compilation errors (likely path issues or lifetime issues needing `+ ?Sized` bounds).
        - [ ] Run `cargo test --all-features` and `cargo test --all-features -- --ignored`. Fix any failures.
        - [ ] Run `cargo fmt` and `cargo clippy`.

    - [ ] **Final Review & Commit:**
        - [ ] Review the changes. Does `AgentExecutor` now contain the execution logic? Are the traits cleaner? Do tests use the executor?
        - [ ] Commit the changes.
    ```

**5. Testing Strategy**

*   **Testing Approach:** Primarily rely on adapting the existing tests to use the new `AgentExecutor`. The goal is to verify that the *same behavior* is achieved, just invoked differently.
*   **Unit Tests:** No new unit tests are strictly required by this refactoring, but the existing ones in `tool_call_aggregator.rs` must still pass.
*   **Integration Tests:** The existing tests in `agent/mod.rs` are critical. They must be updated to use `AgentExecutor` and must all pass, including ignored ones.
*   **Manual Testing:** Minimal manual testing (e.g., running an example that uses an agent via the executor) could be done as a sanity check.

**6. Integration & Deployment**

*   (Same as previous plan - standard code refactoring integration/deployment).

**7. Plan Format & Usage**

*   (Standard)

---

This plan provides a clear path to introducing the `AgentExecutor` and separating the execution logic from the agent trait definitions, aligning with your request while avoiding the complexity of the `RunContext`. Remember to carefully manage `use` statements and trait bounds (`+ ?Sized`) during the implementation.
