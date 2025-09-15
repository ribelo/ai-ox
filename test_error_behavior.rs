use conversion_ox::anthropic_gemini::*;
use gemini_ox::content::{Part, PartData};

fn main() {
    // Test that FileData now returns an error instead of being silently skipped
    let gemini_request = gemini_ox::generate_content::request::GenerateContentRequest::builder()
        .model("gemini-1.5-flash".to_string())
        .content_list(vec![
            gemini_ox::content::Content {
                role: gemini_ox::content::Role::User,
                parts: vec![
                    Part::new(PartData::FileData(gemini_ox::content::FileData::new(
                        "file://test.txt",
                        "text/plain"
                    )))
                ]
            }
        ])
        .build();
    
    match gemini_to_anthropic_request(gemini_request) {
        Ok(_) => println!("ERROR: Should have failed with FileData"),
        Err(e) => println!("SUCCESS: FileData properly returns error: {:?}", e),
    }
    
    // Test that ExecutableCode now returns an error
    let gemini_request2 = gemini_ox::generate_content::request::GenerateContentRequest::builder()
        .model("gemini-1.5-flash".to_string())
        .content_list(vec![
            gemini_ox::content::Content {
                role: gemini_ox::content::Role::User,
                parts: vec![
                    Part::new(PartData::ExecutableCode(gemini_ox::content::ExecutableCode::new(
                        gemini_ox::content::Language::Python,
                        "print('hello')"
                    )))
                ]
            }
        ])
        .build();
    
    match gemini_to_anthropic_request(gemini_request2) {
        Ok(_) => println!("ERROR: Should have failed with ExecutableCode"),
        Err(e) => println!("SUCCESS: ExecutableCode properly returns error: {:?}", e),
    }
    
    println!("All tests passed - no more silent skipping!");
}
