use parking_lot::Mutex;
use sherpa_onnx::{OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, Wave};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::config::Settings;
use std::io::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    Small,  // 0.6b
    Large,  // 1.7b
}

impl ModelVariant {
    pub fn repo(&self) -> &'static str {
        match self {
            ModelVariant::Small => "ilmina/qwen3-asr-0.6b-sherpa-onnx",
            ModelVariant::Large => "ilmina/qwen3-asr-1.7b-sherpa-onnx",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ModelVariant::Small => "Qwen3-ASR 0.6B",
            ModelVariant::Large => "Qwen3-ASR 1.7B",
        }
    }

    pub fn dir_name(&self) -> &'static str {
        match self {
            ModelVariant::Small => "qwen3-asr-0.6b",
            ModelVariant::Large => "qwen3-asr-1.7b",
        }
    }

    pub fn model_dir(&self) -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("perihelion")
            .join("models")
            .join(self.dir_name())
    }
}

pub const MODEL_FILES: &[&str] = &[
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
];

pub const STRIP_TAGS: &[&str] = &[
    "<|en|>", "<|zh|>", "<|ja|>", "<|ko|>", "<|fr|>",
    "<|de|>", "<|it|>", "<|es|>", "<|ru|>", "<|asr|>", "<|text|>",
];

/// Returns total physical RAM in GB, or None if it cannot be determined.
#[cfg(target_os = "windows")]
pub fn get_total_ram_gb() -> Option<f64> {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct MEMORYSTATUSEX {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
    }
    let mut mem: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    mem.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut mem) };
    if ok != 0 {
        Some(mem.ullTotalPhys as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        None
    }
}

#[cfg(not(target_os = "windows"))]
pub fn get_total_ram_gb() -> Option<f64> {
    None
}

pub fn build_recognizer(model_dir: &Path, settings: &Settings) -> Result<OfflineRecognizer, Box<dyn std::error::Error>> {
    let hotwords_file = if !settings.hotwords.trim().is_empty() {
        let hw_path = model_dir.join("hotwords.txt");
        if let Ok(mut f) = std::fs::File::create(&hw_path) {
            let _ = f.write_all(settings.hotwords.as_bytes());
        }
        hw_path.to_string_lossy().into_owned()
    } else {
        String::new()
    };

    let mut config = OfflineRecognizerConfig::default();
    config.decoding_method = Some(if settings.high_accuracy {
        "modified_beam_search".to_string()
    } else {
        "greedy_search".to_string()
    });
    config.max_active_paths = settings.search_depth;
    config.hotwords_file = Some(hotwords_file);
    config.hotwords_score = settings.hotwords_boost;

    config.model_config = OfflineModelConfig {
        qwen3_asr: sherpa_onnx::OfflineQwen3ASRModelConfig {
            conv_frontend: Some(
                model_dir.join("conv_frontend.onnx").to_string_lossy().into_owned(),
            ),
            encoder: Some(
                model_dir.join("encoder.int8.onnx").to_string_lossy().into_owned(),
            ),
            decoder: Some(
                model_dir.join("decoder.int8.onnx").to_string_lossy().into_owned(),
            ),
            tokenizer: Some(
                model_dir.join("tokenizer").to_string_lossy().into_owned(),
            ),
            ..Default::default()
        },
        num_threads: settings.num_threads,
        provider: Some("cpu".to_string()),
        debug: false,
        ..Default::default()
    };

    OfflineRecognizer::create(&config)
        .ok_or("Failed to create OfflineRecognizer".into())
}

pub fn clean_transcription(text: &str) -> String {
    let needs_cleaning = STRIP_TAGS.iter().any(|tag| text.contains(tag));
    if !needs_cleaning {
        let trimmed = text.trim();
        if trimmed.len() == text.len() {
            return text.to_string();
        }
        return trimmed.to_string();
    }

    let mut result = text.to_string();
    for tag in STRIP_TAGS {
        result = result.replace(tag, "");
    }

    let trimmed = result.trim();
    if trimmed.len() == result.len() {
        result
    } else {
        trimmed.to_string()
    }
}

pub struct AsrEngine {
    recognizer: OfflineRecognizer,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
}

impl AsrEngine {
    pub fn new(model_dir: &Path, settings: &Settings) -> Result<Self, Box<dyn std::error::Error>> {
        let recognizer = build_recognizer(model_dir, settings)?;
        Ok(Self {
            recognizer,
            audio_buffer: Arc::new(Mutex::new(Vec::with_capacity(16000 * 30))),
        })
    }

    pub fn extend_samples<I: IntoIterator<Item = f32>>(&self, iter: I) {
        let mut buffer = self.audio_buffer.lock();
        buffer.extend(iter);
    }

    pub fn check_new_audio(&self, since: usize, min_total: usize) -> Option<(f32, usize)> {
        let buffer = self.audio_buffer.lock();
        if buffer.len() < min_total || buffer.len() <= since {
            return None;
        }
        let new_samples = &buffer[since..];
        let mut sum_sq: f32 = 0.0;
        let chunks = new_samples.chunks_exact(4);
        let remainder = chunks.remainder();
        for chunk in chunks {
            sum_sq += chunk[0] * chunk[0]
                    + chunk[1] * chunk[1]
                    + chunk[2] * chunk[2]
                    + chunk[3] * chunk[3];
        }
        for &s in remainder {
            sum_sq += s * s;
        }
        let rms = (sum_sq / new_samples.len() as f32).sqrt();
        Some((rms, buffer.len()))
    }

    pub fn clear_buffer(&self) -> Vec<f32> {
        let mut buffer = self.audio_buffer.lock();
        let taken = std::mem::take(&mut *buffer);
        buffer.reserve(16000 * 30);
        taken
    }

    pub fn transcribe_samples(&self, sample_rate: i32, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate, samples);
        self.recognizer.decode(&stream);
        let result = stream.get_result();
        Ok(result.map(|r| r.text).unwrap_or_default())
    }

    pub fn transcribe_file(&self, wav_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        let wave = Wave::read(wav_path.to_str().unwrap_or(""))
            .ok_or("Failed to read WAV file")?;
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(wave.sample_rate(), wave.samples());
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or("Failed to get recognition result")?;
        Ok(result.text)
    }
}
