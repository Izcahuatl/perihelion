use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListeningMode {
    AlwaysOn,
    ToggleOsc,
    ToggleButton,
}

impl Default for ListeningMode {
    fn default() -> Self {
        ListeningMode::ToggleButton
    }
}
impl ListeningMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ListeningMode::AlwaysOn => "Always On",
            ListeningMode::ToggleOsc => "Toggle-OSC",
            ListeningMode::ToggleButton => "Toggle-Button",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    Cpu,
    Dml,
    Cuda,
}

impl Default for Provider {
    fn default() -> Self {
        Provider::Cpu
    }
}
impl Provider {
    pub fn as_config_str(&self) -> &'static str {
        match self {
            Provider::Cpu => "cpu",
            Provider::Dml => "dml",
            Provider::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub listening_mode: ListeningMode,
    pub auto_copy: bool,
    pub append_mode: bool,
    pub num_threads: i32,
    pub high_accuracy: bool,
    pub search_depth: i32,
    pub hotwords: String,
    pub hotwords_boost: f32,
    pub provider: Provider,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            listening_mode: ListeningMode::default(),
            auto_copy: true,
            append_mode: true,
            num_threads: 4,
            high_accuracy: false,
            search_depth: 4,
            hotwords: String::new(),
            hotwords_boost: 1.5,
            provider: Provider::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub settings: Settings,
    pub selected_device: usize,
}

