// use futures_util::future::BoxFuture;
// use gemini_ox::{
//     generate_content::request::GenerateContentRequest as GeminiRequest,
//     generate_content::response::GenerateContentResponse as GeminiResponse, Gemini,
// };

// use crate::{
//     content::{
//         delta::MessageStreamEvent, requests::GenerateContentRequest,
//         response::GenerateContentResponse,
//     },
//     errors::GenerateContentError,
//     generate_content::GenerateContent,
// };

// impl GenerateContent for Gemini {
//     fn request<'a>(
//         &'a self,
//         request: GenerateContentRequest,
//     ) -> BoxFuture<'a, Result<GenerateContentResponse, GenerateContentError>> {
//         let gemini_request: GeminiRequest = request.into();
//         self.generate_content()
//     }

//     fn request_stream<'a>(
//         &'a self,
//         request: &'a GenerateContentRequest,
//     ) -> futures_util::stream::BoxStream<'a, Result<MessageStreamEvent, GenerateContentError>> {
//         todo!()
//     }
// }
