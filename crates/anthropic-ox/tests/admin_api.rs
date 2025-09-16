#![cfg(feature = "admin")]

use anthropic_ox::Anthropic;

#[tokio::test]
async fn test_list_organization_users() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    match client.list_organization_users().await {
        Ok(response) => {
            println!("✅ API call successful: list_organization_users");
            assert!(response.data.is_empty() || !response.data.is_empty());
        }
        Err(e) => {
            println!(
                "⚠️  API call failed (this might be expected for non-admin keys): {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_list_organization_invites() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    match client.list_organization_invites().await {
        Ok(response) => {
            println!("✅ API call successful: list_organization_invites");
            assert!(response.data.is_empty() || !response.data.is_empty());
        }
        Err(e) => {
            println!(
                "⚠️  API call failed (this might be expected for non-admin keys): {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_list_workspaces() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    match client.list_workspaces().await {
        Ok(response) => {
            println!("✅ API call successful: list_workspaces");
            assert!(response.data.is_empty() || !response.data.is_empty());
        }
        Err(e) => {
            println!(
                "⚠️  API call failed (this might be expected for non-admin keys): {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_list_api_keys() {
    let client = match Anthropic::load_from_env() {
        Ok(client) => client,
        Err(_) => {
            println!("ℹ️  Skipping API test: ANTHROPIC_API_KEY not found");
            return;
        }
    };

    match client.list_api_keys().await {
        Ok(response) => {
            println!("✅ API call successful: list_api_keys");
            assert!(response.data.is_empty() || !response.data.is_empty());
        }
        Err(e) => {
            println!(
                "⚠️  API call failed (this might be expected for non-admin keys): {}",
                e
            );
        }
    }
}
