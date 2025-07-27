#[cfg(feature = "mistral")]
use ai_ox::model::mistral::MistralModel;
use ai_ox::model::Model;
use ai_ox::content::message::{Message, MessageRole};
use ai_ox::content::part::Part;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mistral")]
    {
        // Create Mistral model
        let model = MistralModel::new("mistral-small-latest").await?;
        
        // Create messages
        let messages = vec![
            Message::new(
                MessageRole::User,
                vec![Part::Text {
                    text: "Hello! What is the weather like in Paris today?".to_string(),
                }],
            )
        ];
        
        // Send request
        let response = model.request(messages.into()).await?;
        
        // Print response
        println!("Response: {:?}", response.to_string());
        println!("Model: {}", response.model_name);
        println!("Vendor: {}", response.vendor_name);
        
        if let Some(text) = response.to_string() {
            println!("Text: {}", text);
        }
    }
    
    #[cfg(not(feature = "mistral"))]
    {
        println!("This example requires the 'mistral' feature to be enabled");
    }
    
    Ok(())
}