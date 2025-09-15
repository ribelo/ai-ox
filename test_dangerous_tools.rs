use ai_ox::tool::{ToolBox, ToolHooks};
use ai_ox::{toolbox, dangerous};
use thiserror::Error;

#[derive(Error, Debug)]
enum TestError {
    #[error("Test error: {message}")]
    Message { message: String },
}

struct TestService;

#[toolbox]
impl TestService {
    /// A safe function that just returns a greeting
    pub fn greet(&self, name: String) -> String {
        format!("Hello, {}!", name)
    }
    
    /// A dangerous function that could delete files
    #[dangerous]
    pub fn delete_file(&self, path: String) -> Result<String, TestError> {
        // This would normally delete a file
        Ok(format!("Would delete file: {}", path))
    }
    
    /// Another dangerous function
    #[dangerous]  
    pub async fn execute_command(&self, command: String) -> Result<String, TestError> {
        // This would normally execute a shell command
        Ok(format!("Would execute: {}", command))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let service = TestService;
    
    // Test that dangerous functions are correctly identified
    let dangerous_functions = service.dangerous_functions();
    println!("Dangerous functions: {:?}", dangerous_functions);
    
    // Should contain both "delete_file" and "execute_command"
    assert!(dangerous_functions.contains(&"delete_file"));
    assert!(dangerous_functions.contains(&"execute_command"));
    assert!(!dangerous_functions.contains(&"greet"));
    
    // Test with hooks
    let hooks = ToolHooks::new()
        .with_approval(|request| {
            Box::pin(async move {
                println!("Approval requested for: {}", request.operation);
                println!("Details: {}", request.details);
                println!("Risk level: {:?}", request.risk_level);
                // Auto-approve for this test
                true
            })
        });
        
    // Test invoking with hooks - should work
    let call = ai_ox::tool::ToolUse::new(
        "test_id",
        "delete_file", 
        serde_json::json!("/tmp/test.txt")
    );
    
    let result = service.invoke_with_hooks(call, hooks).await;
    println!("Result: {:?}", result);
    
    println!("✅ All tests passed!");
}