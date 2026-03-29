mod traits;
pub use traits::{ChatMessage, ChatResponse, GenConfig, LlmProvider, Role};

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::OpenAiProvider;

mod prompts;
pub(crate) use prompts::Prompts;

pub mod mock;
pub use mock::MockLlmProvider;

pub mod schemas;
