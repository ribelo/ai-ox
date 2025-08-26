use serde_json::json;
use conversion_ox::anthropic_gemini::draft07_to_openapi3;

#[test]
fn test_schema_conversion_bash_tool() {
    // This is the actual schema from Claude Code's Bash tool that might be causing issues
    let draft07_schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute"
            },
            "description": {
                "type": "string",
                "description": " Clear, concise description of what this command does in 5-10 words. Examples:\nInput: ls\nOutput: Lists files in current directory\n\nInput: git status\nOutput: Shows working tree status\n\nInput: npm install\nOutput: Installs package dependencies\n\nInput: mkdir foo\nOutput: Creates directory 'foo'"
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Set to true to run this command in the background. Use BashOutput to read the output later."
            },
            "timeout": {
                "type": "number",
                "description": "Optional timeout in milliseconds (max 600000)"
            }
        },
        "required": ["command"],
        "additionalProperties": false
    });

    let openapi_schema = draft07_to_openapi3(draft07_schema);
    
    println!("Converted schema: {}", serde_json::to_string_pretty(&openapi_schema).unwrap());
    
    // Check that problematic fields are removed
    assert!(!openapi_schema.as_object().unwrap().contains_key("$schema"));
    assert!(!openapi_schema.as_object().unwrap().contains_key("additionalProperties"));
    
    // Check that the structure is valid
    assert_eq!(openapi_schema["type"], "object");
    assert!(openapi_schema["properties"].is_object());
    assert!(openapi_schema["required"].is_array());
}

#[test]
fn test_schema_conversion_with_nullable_types() {
    // Test the nullable type conversion that might be problematic
    let draft07_schema = json!({
        "type": "object",
        "properties": {
            "optional_field": {
                "type": ["string", "null"],
                "description": "An optional string field"
            },
            "required_field": {
                "type": "string",
                "description": "A required string field"
            }
        },
        "required": ["required_field"],
        "additionalProperties": false
    });

    let openapi_schema = draft07_to_openapi3(draft07_schema);
    
    println!("Nullable conversion: {}", serde_json::to_string_pretty(&openapi_schema).unwrap());
    
    // Check nullable conversion
    let optional_field = &openapi_schema["properties"]["optional_field"];
    assert_eq!(optional_field["type"], "string");
    assert_eq!(optional_field["nullable"], true);
}

#[test]
fn test_schema_conversion_property_names_with_hyphens() {
    // Test property names with hyphens that might be problematic for Gemini
    let draft07_schema = json!({
        "type": "object",
        "properties": {
            "-A": {
                "type": ["number", "null"],
                "description": "Number of lines to show after each match"
            },
            "-B": {
                "type": ["number", "null"],
                "description": "Number of lines to show before each match"  
            },
            "-i": {
                "type": "boolean",
                "description": "Case insensitive search"
            }
        },
        "required": [],
        "additionalProperties": false
    });

    let openapi_schema = draft07_to_openapi3(draft07_schema);
    
    println!("Hyphen properties conversion: {}", serde_json::to_string_pretty(&openapi_schema).unwrap());
    
    // Check that hyphen properties are preserved
    let properties = openapi_schema["properties"].as_object().unwrap();
    assert!(properties.contains_key("-A"));
    assert!(properties.contains_key("-B"));
    assert!(properties.contains_key("-i"));
}

#[test]
fn test_schema_conversion_complex_tool() {
    // Test a more complex tool schema similar to what Claude Code sends
    let draft07_schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for in file contents"
            },
            "glob": {
                "type": ["string", "null"],
                "description": "Glob pattern to filter files"
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "Output mode for search results"
            },
            "head_limit": {
                "type": ["number", "null"],
                "description": "Limit output to first N entries"
            },
            "-A": {
                "type": ["number", "null"],
                "description": "Number of lines to show after each match"
            }
        },
        "required": ["pattern"],
        "additionalProperties": false,
        "title": "Grep Tool Schema",
        "default": {},
        "maxProperties": 10,
        "minProperties": 1
    });

    let openapi_schema = draft07_to_openapi3(draft07_schema);
    
    println!("Complex schema conversion: {}", serde_json::to_string_pretty(&openapi_schema).unwrap());
    
    // Verify all problematic Draft-07 fields are removed
    let obj = openapi_schema.as_object().unwrap();
    assert!(!obj.contains_key("$schema"));
    assert!(!obj.contains_key("additionalProperties"));
    assert!(!obj.contains_key("title"));
    assert!(!obj.contains_key("default"));
    assert!(!obj.contains_key("maxProperties"));
    assert!(!obj.contains_key("minProperties"));
    
    // Check nullable conversions
    let glob_field = &openapi_schema["properties"]["glob"];
    assert_eq!(glob_field["type"], "string");
    assert_eq!(glob_field["nullable"], true);
    
    let head_limit_field = &openapi_schema["properties"]["head_limit"];
    assert_eq!(head_limit_field["type"], "number");
    assert_eq!(head_limit_field["nullable"], true);
    
    // Check hyphen property transformation
    let properties = openapi_schema["properties"].as_object().unwrap();
    assert!(properties.contains_key("A")); // "-A" → "A"
    assert!(!properties.contains_key("-A")); // Original should be removed
    
    let a_field = &openapi_schema["properties"]["A"];
    assert_eq!(a_field["type"], "number");
    assert_eq!(a_field["nullable"], true);
}