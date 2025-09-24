#![deny(clippy::all)]
#![warn(clippy::pedantic)]
// Add necessary allows for proc macro code
#![allow(clippy::too_many_lines)] // Proc macros can be long
#![allow(clippy::needless_pass_by_value)] // Common in syn visitors/parsers

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, quote_spanned};
use syn::{
    Attribute, Expr, ExprLit, FnArg, GenericArgument, Ident, ImplItem, ImplItemFn, ItemImpl, Lit,
    Meta, PathArguments, ReturnType, Type, TypePath, TypeReference, Visibility, parse_macro_input,
    spanned::Spanned,
};

// Helper to represent extracted method info
struct ToolMethodInfo<'a> {
    name: &'a Ident,
    name_str: String,
    doc_comment: Option<String>,
    input_arg_ty: Option<&'a Type>, // The 'I' in I or Option<I>, or None
    is_input_optional: bool,        // True if original was Option<I>
    output_ty: Option<&'a Type>,    // The 'O' in Result<O, E> or just O, None for unit type
    error_ty: Option<&'a Type>,     // The 'E' in Result<O, E>, None for infallible tools
    is_async: bool,
    span: Span,
}

#[proc_macro_attribute]
pub fn toolbox(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let impl_block = parse_macro_input!(item as ItemImpl);

    // Ensure the impl block is for a struct or enum, not a trait impl
    if impl_block.trait_.is_some() {
        return syn::Error::new(
            impl_block.impl_token.span(),
            "`#[toolbox]` can only be applied to inherent impl blocks (e.g., `impl MyStruct { ... }`), not trait impl blocks.",
        )
        .to_compile_error()
        .into();
    }

    match impl_toolbox(&impl_block) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// Main function to generate the `impl ToolBox for ...` block
fn impl_toolbox(impl_block: &ItemImpl) -> syn::Result<proc_macro2::TokenStream> {
    // Always use ::ai_ox to ensure the macro works correctly in all contexts,
    // including tests, examples, and external crates
    let crate_prefix = quote! { ::ai_ox };

    let self_ty = &impl_block.self_ty;
    let generics = &impl_block.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut tool_methods = Vec::new();
    let mut errors: Option<syn::Error> = None;

    // Iterate through items in the impl block (methods, consts, etc.)
    for item in &impl_block.items {
        // We only care about functions/methods
        if let ImplItem::Fn(method) = item {
            // Only consider public methods as potential tools
            if matches!(method.vis, Visibility::Public(_)) {
                match process_method(method) {
                    Ok(Some(tool_info)) => {
                        // Successfully processed a valid tool method
                        tool_methods.push(tool_info);
                    }
                    Ok(None) => {
                        // Method is public but doesn't match the tool signature (e.g., wrong args, return type).
                        // Ignore silently as it might be a regular public method.
                    }
                    Err(err) => {
                        // An error occurred processing this method. Collect errors.
                        match errors.as_mut() {
                            Some(existing_error) => existing_error.combine(err),
                            None => errors = Some(err),
                        }
                    }
                }
            }
        }
    }

    // If any errors occurred during method processing, return the combined error.
    if let Some(collected_errors) = errors {
        return Err(collected_errors);
    }

    // Optional: Could emit a warning if tool_methods is empty, but let's allow empty toolboxes.

    // Create a binding for the unit type to avoid borrow checker issues
    let unit_type = &syn::parse_quote!(());

    // --- Generate `tools` method body ---
    let metadata_items = tool_methods.iter().map(|info| {
        let name = &info.name_str;
        // Provide an empty string if doc comment is missing
        let description = info.doc_comment.as_deref().unwrap_or("");
        // Use schema_for_type for the *effective* input parameter type.
        // If no input_arg_ty, use unit type `()`.
        // If input is Option<I>, use I for schema (OpenAPI generally doesn't wrap optionals explicitly in the schema type itself).
        // If input is I, use I for schema.
        let schema_input_ty = info.input_arg_ty.unwrap_or(unit_type); // Use the bound unit_type

        quote_spanned! {info.span=>
            // Generate FunctionMetadata for each tool
            #crate_prefix::tool::FunctionMetadata {
                name: #name.to_string(),
                description: Some(#description.to_string()),
                // Use the fully qualified path to schema_for_type.
                // Pass the inner type I even if the function takes Option<I>.
                // If no args, pass ().
                parameters: #crate_prefix::tool::schema_for_type::<#schema_input_ty>(),
            }
        }
    });

    // --- Generate `invoke` method match arms ---
    let invoke_match_arms = tool_methods.iter().map(|info| {
        let method_name = info.name;
        let method_name_str = &info.name_str;
        let input_arg_ty = info.input_arg_ty; // This is type 'I'
        let output_ty = info.output_ty;     // This is type 'O'
        let error_ty = info.error_ty;       // This is type 'E'
        let is_async = info.is_async;
        let await_token = if is_async { quote! {.await} } else { quote! {} };
        let span = info.span;

        // Code to deserialize args based on whether the method expects I or Option<I>
        let deserialize_code = if let Some(ty) = input_arg_ty {
            if info.is_input_optional {
                // Method expects Option<I>, deserialize directly to Option<I>
                quote_spanned! {span=>
                    // Deserialize into Option<#ty>. If input JSON is null or missing field, result is Ok(None).
                    let args: Option<#ty> = serde_json::from_value(call.args.clone())
                        .map_err(|e| #crate_prefix::tool::ToolError::input_deserialization(
                            #method_name_str, e
                        ))?;
                }
            } else {
                 // Method expects I, deserialize to I. Error if null/missing.
                 quote_spanned! {span=>
                     // Deserialize into #ty. This will fail if the JSON value is null or doesn't match.
                     let args: #ty = serde_json::from_value(call.args.clone())
                         .map_err(|e| #crate_prefix::tool::ToolError::input_deserialization(
                            #method_name_str, e
                         ))?;
                 }
            }
        } else {
            // No input args expected by the function.
            // We might want to check call.args is empty/null and error if not,
            // but for now, let's just ignore the input JSON value.
            quote! { let args = (); }
        };

        // Code to call the actual method (sync or async)
        // `args` variable holds the deserialized value (I, Option<I>, or ())
        let call_code = if input_arg_ty.is_some() {
            quote! { self.#method_name(args)#await_token }
        } else {
            quote! { self.#method_name()#await_token }
        };

        // Code to handle fallible, infallible, and side-effect tools
        let result_handling_code = match (output_ty, error_ty) {
            (Some(out_ty), Some(err_ty)) => {
                // Fallible tool - returns Result<O, E>
                // Check if the output type is String to avoid double JSON encoding
                let is_string_output = matches!(out_ty, Type::Path(type_path) if
                    type_path.qself.is_none() &&
                    type_path.path.segments.len() == 1 &&
                    type_path.path.segments[0].ident == "String"
                );

                if is_string_output {
                    // For String returns, use the string directly
                    quote_spanned! {span=>
                        // The actual method call happens here
                        let result: Result<#out_ty, #err_ty> = #call_code;

                        // Process the result from the tool method
                        match result {
                            Ok(output) => {
                                // Return the ToolResult with the string directly
                                Ok(#crate_prefix::content::part::Part::ToolResult {
                                    id: call.id.clone(),
                                    name: call.name.clone(),
                                    parts: vec![#crate_prefix::content::part::Part::Text { text: output, ext: std::collections::BTreeMap::new() }],
                                    ext: std::collections::BTreeMap::new(),
                                })
                            }
                            Err(user_err) => {
                                // The tool method returned an error (E)
                                // Map the user's error (E) into ToolError::Execution
                                Err(#crate_prefix::tool::ToolError::execution(
                                    #method_name_str, user_err
                                ))
                            },
                        }
                    }
                } else {
                    // For other types, serialize to JSON
                    quote_spanned! {span=>
                        // The actual method call happens here
                        let result: Result<#out_ty, #err_ty> = #call_code;

                        // Process the result from the tool method
                        match result {
                            Ok(output) => {
                                // Serialize the successful output (O) into a serde_json::Value
                                let response_value = serde_json::to_value(output).map_err(|e| {
                                    #crate_prefix::tool::ToolError::output_serialization(
                                         #method_name_str, e
                                    )
                                })?; // Early return on serialization error

                                // Return the ToolResult directly with the response_value
                                Ok(#crate_prefix::content::part::Part::ToolResult {
                                    id: call.id.clone(),
                                    name: call.name.clone(),
                                    parts: vec![#crate_prefix::content::part::Part::Text { text: serde_json::to_string(&response_value).unwrap(), ext: std::collections::BTreeMap::new() }],
                                    ext: std::collections::BTreeMap::new(),
                                })
                            }
                            Err(user_err) => {
                                // The tool method returned an error (E)
                                // Map the user's error (E) into ToolError::Execution
                                Err(#crate_prefix::tool::ToolError::execution(
                                    #method_name_str, user_err
                                ))
                            },
                        }
                    }
                }
            }
            (Some(out_ty), None) => {
                // Infallible tool - returns O directly
                // Check if the output type is String to avoid double JSON encoding
                let is_string_output = matches!(out_ty, Type::Path(type_path) if
                    type_path.qself.is_none() &&
                    type_path.path.segments.len() == 1 &&
                    type_path.path.segments[0].ident == "String"
                );

                if is_string_output {
                    // For String returns, use the string directly
                    quote_spanned! {span=>
                        // The actual method call happens here - this returns String directly
                        let output: #out_ty = #call_code;

                        // Create a message from the tool response
                        let message = #crate_prefix::content::Message::from_tool_response(
                            call.id.clone(),
                            call.name.clone(),
                            serde_json::Value::String(output)
                        );

                          // Return the ToolResult
                          Ok(#crate_prefix::content::part::Part::ToolResult {
                              id: call.id.clone(),
                              name: call.name.clone(),
                              parts: vec![#crate_prefix::content::part::Part::Text { text: output, ext: std::collections::BTreeMap::new() }],
                              ext: std::collections::BTreeMap::new(),
                          })
                    }
                } else {
                    // For other types, serialize to JSON
                    quote_spanned! {span=>
                        // The actual method call happens here - this returns O directly
                        let output: #out_ty = #call_code;

                        // Serialize the output (O) into a serde_json::Value
                        let response_value = serde_json::to_value(output).map_err(|e| {
                            #crate_prefix::tool::ToolError::output_serialization(
                                 #method_name_str, e
                            )
                        })?; // Early return on serialization error

                          // Return the ToolResult directly with the response_value
                          Ok(#crate_prefix::content::part::Part::ToolResult {
                              id: call.id.clone(),
                              name: call.name.clone(),
                              parts: vec![#crate_prefix::content::part::Part::Text { text: serde_json::to_string(&response_value).unwrap(), ext: std::collections::BTreeMap::new() }],
                              ext: std::collections::BTreeMap::new(),
                          })
                     }
                 }
            }
            (None, None) => {
                // Side-effect tool - returns () (unit type)
                quote_spanned! {span=>
                    // The actual method call happens here - this is for side effects only
                    #call_code;

                    // For side-effect tools, we return success content
                    Ok(#crate_prefix::content::part::Part::ToolResult {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        parts: vec![#crate_prefix::content::part::Part::Text { text: "Operation completed successfully".to_string(), ext: std::collections::BTreeMap::new() }],
                        ext: std::collections::BTreeMap::new(),
                    })
                }
            }
            (None, Some(_)) => {
                // This case shouldn't happen (no output but has error type)
                quote_spanned! {span=>
                    compile_error!("Invalid tool signature: error type without output type");
                }
            }
        };

        // Combine deserialization, call, and result handling into a match arm
        quote_spanned! {span=>
            #method_name_str => {
                #deserialize_code
                // The result handling code now evaluates to Result<Part, ToolError>
                #result_handling_code
            }
        }
    });

    // --- Generate the `impl ToolBox` block ---
    // Combine the original impl block provided by the user
    // with the generated `impl ToolBox for ...` block.
    let generated_impl = quote! {
        #impl_block // Keep the original impl block as provided

        // Manually implement the async trait method using BoxFuture
        impl #impl_generics #crate_prefix::tool::ToolBox for #self_ty #ty_generics #where_clause {

            /// Returns the metadata (function declarations) for all functions provided by this toolbox.
            ///
            /// This is used to inform the model about the available functions, their parameters,
            /// and descriptions.
            fn tools(&self) -> Vec<#crate_prefix::tool::Tool> {
                 // Collect FunctionMetadata, wrap in FunctionDeclarations, then wrap in Vec
                vec![
                    #crate_prefix::tool::Tool::FunctionDeclarations(vec![#(#metadata_items),*]) // Splice in the generated FunctionMetadata items
                ]
            }

            /// Invokes a function by name with the given arguments.
            /// This implementation matches the function name and dispatches
            /// to the appropriate method on the underlying service struct.
            /// It handles argument deserialization, potential async execution,
            /// result handling, and output serialization.
            // Manually implement async fn using BoxFuture
            // Use standard lifetime syntax 'a directly
            fn invoke(
                  &self,
                  call: #crate_prefix::tool::ToolUse,
            ) -> futures_util::future::BoxFuture<Result<#crate_prefix::content::part::Part, #crate_prefix::tool::ToolError>> {
                 Box::pin(async move { // Wrap the body in Box::pin(async move { ... })
                     let function_name = call.name.clone(); // Clone name for use in match
                     match function_name.as_str() {
                        #(#invoke_match_arms)* // Splice in the generated match arms
                        // If no match arm corresponds to the function name, return ToolNotFound error.
                        _ => Err(#crate_prefix::tool::ToolError::not_found( // Use crate_prefix
                         call.name // Pass the name that was not found
                        )),
                     }
                 }) // Close Box::pin(async move { ... })
            } // Close fn invoke

            // has_function uses the default implementation provided in the trait,
            // which relies on the tools() method generated above.
        }
    };

    Ok(generated_impl)
}

/// Processes a single method within the `impl` block to determine if it's a valid tool function
/// and extracts relevant information if it is.
fn process_method(method: &ImplItemFn) -> syn::Result<Option<ToolMethodInfo<'_>>> {
    let method_sig = &method.sig;
    let method_name = &method_sig.ident;
    let method_name_str = method_name.to_string();
    let method_span = method.span(); // Use the span of the whole method for context

    // --- Basic Signature Checks ---

    // Visibility check is handled in the caller (impl_toolbox)

    // Must have a receiver (`self`, `&self`, `&mut self`)
    let mut inputs = method_sig.inputs.iter();
    let receiver = inputs.next().ok_or_else(|| {
        syn::Error::new(
            method_sig.fn_token.span(),
            "Tool methods must have a receiver (e.g., `&self`) as the first argument.",
        )
    })?;
    match receiver {
        FnArg::Receiver(_) => {} // Standard receiver `&self`, `self`, etc. - Good
        FnArg::Typed(pt) if is_self_type(&pt.ty) => {} // Explicit type `Self`, `&Self` - Also Good
        FnArg::Typed(_) => {
            // First argument is typed but not 'Self' or is missing.
            // This is likely not intended as a tool method, so ignore silently.
            return Ok(None);
            // Alternatively, could error:
            // return Err(syn::Error::new(receiver.span(), "First argument must be a `self` receiver"));
        }
    }

    // Check remaining arguments: Must have 0 or 1 additional argument.
    let input_arg = inputs.next();
    let extra_arg = inputs.next(); // Should be None if valid

    let (input_arg_ty, is_input_optional) = match (input_arg, extra_arg) {
        // Case 1: One typed argument (`arg: T` or `arg: Option<T>`), and no more arguments.
        (Some(FnArg::Typed(pat_type)), None) => {
            // Check if the type is Option<T>
            if let Some(inner_ty) = get_option_inner_type(&pat_type.ty) {
                (Some(inner_ty), true) // Input is Option<T>, store inner T
            } else {
                (Some(&*pat_type.ty), false) // Input is T, store T
            }
        }
        // Case 2: No arguments besides the receiver, and no more arguments.
        (None, None) => (None, false), // No input argument
        // Case 3: Invalid signatures for a tool method.
        // - More than one argument after receiver.
        // - Second argument is another receiver (impossible in Rust syntax?).
        // - Argument is not typed (e.g., `_ : impl Trait` might be complex, ignore for now).
        // Treat these as non-tool methods and ignore silently.
        _ => return Ok(None),
    };

    // --- Return Type Check ---
    // Can return either `Result<O, E>` (fallible), just `O` (infallible), or `()` (side-effect)
    let (output_ty, error_ty) = match &method_sig.output {
        ReturnType::Type(_, ty) => {
            // First try to extract O and E from Result<O, E>
            if let Ok((ok_ty, err_ty)) = extract_result_types(ty) {
                // It's a Result<O, E> - fallible tool
                (Some(ok_ty), Some(err_ty))
            } else {
                // Not a Result, so it's just O - infallible tool
                (Some(ty.as_ref()), None)
            }
        }
        ReturnType::Default => {
            // Function returns `()` implicitly - treat as infallible side-effect tool
            (None, None)
        }
    };

    // --- Other Info ---
    let is_async = method_sig.asyncness.is_some();
    let doc_comment = extract_doc_comment(&method.attrs);
    // If all checks passed, return the extracted info
    Ok(Some(ToolMethodInfo {
        name: method_name,
        name_str: method_name_str,
        doc_comment,
        input_arg_ty, // Type I (inner type if Option<I>) or None
        is_input_optional,
        output_ty, // Type O
        error_ty,  // Type E
        is_async,
        span: method_span,
    }))
}

/// Helper: Checks if a type is `Self` or a reference to `Self`.
fn is_self_type(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { qself: None, path }) => path.is_ident("Self"),
        Type::Reference(TypeReference { elem, .. }) => is_self_type(elem),
        _ => false,
    }
}

/// Helper: Extracts `O` and `E` types from `Result<O, E>`.
/// Returns `Ok((&Type, &Type))` or `Err(String)` describing the failure.
fn extract_result_types(return_type: &Type) -> Result<(&Type, &Type), String> {
    // Check if it's a path type (like `std::result::Result` or just `Result`)
    if let Type::Path(type_path) = return_type {
        // Get the last segment of the path (e.g., `Result` in `std::result::Result`)
        if let Some(segment) = type_path.path.segments.last() {
            // Check if the identifier is "Result"
            if segment.ident == "Result" {
                // Check if it has angle-bracketed arguments like `<O, E>`
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    // Must have exactly two arguments
                    if args.args.len() == 2 {
                        let ok_arg = &args.args[0];
                        let err_arg = &args.args[1];
                        // Ensure both arguments are types (not lifetimes, consts, etc.)
                        if let (GenericArgument::Type(ok_ty), GenericArgument::Type(err_ty)) =
                            (ok_arg, err_arg)
                        {
                            // Success! Return references to the Ok type and Err type.
                            return Ok((ok_ty, err_ty));
                        }
                    }
                }
            }
        }
    }
    // If any check failed, return an error string.
    Err("Expected `Result<OkType, ErrorType>`".to_string())
}

/// Helper: Extracts the inner type `T` from `Option<T>`.
/// Returns `Some(&Type)` if it matches, `None` otherwise.
fn get_option_inner_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty {
        // Ensure it's a simple path (no `::` prefix like `<T as Trait>::Option`)
        if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
            let segment = type_path.path.segments.first().unwrap(); // Safe due to len check
            // Check if the identifier is "Option"
            if segment.ident == "Option" {
                // Check for angle-bracketed arguments `<T>`
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    // Must have exactly one argument
                    if args.args.len() == 1 {
                        // Ensure the argument is a type
                        if let GenericArgument::Type(inner_ty) = &args.args[0] {
                            // Success! Return a reference to the inner type.
                            return Some(inner_ty);
                        }
                    }
                }
            }
        }
    }
    // If any check failed, it's not Option<T> in the expected form.
    None
}

/// Helper: Extracts and concatenates `///` doc comments from attributes.
fn extract_doc_comment(attrs: &[Attribute]) -> Option<String> {
    let mut lines = Vec::new();
    for attr in attrs {
        // Check if the attribute path is `doc`
        if attr.path().is_ident("doc") {
            // Check if it's a Meta::NameValue attribute (like `doc = "..."`)
            if let Meta::NameValue(nv) = &attr.meta {
                // Check if the value expression is a literal string
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) = &nv.value
                {
                    // Get the string value, trim leading whitespace (handles `/// ` vs `///`),
                    // but keep trailing whitespace and internal newlines if present in the string literal.
                    lines.push(s.value().trim_start().to_string());
                }
            }
        }
    }
    // If no doc comment lines were found, return None.
    if lines.is_empty() {
        None
    } else {
        // Join the lines with newlines to reconstruct the comment block.
        Some(lines.join("\n"))
    }
}
