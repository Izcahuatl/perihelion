use parking_lot::Mutex;
use sherpa_onnx::{OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, Wave};
use std::path::Path;
use std::sync::Arc;
use crate::config::Settings;
use std::io::Write;

pub const MODEL_REPO: &str = "pantinor/sherpa-onnx-qwen3-asr-0.6b-int8";

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

pub fn build_recognizer(model_dir: &Path, settings: &Settings) -> anyhow::Result<OfflineRecognizer> {
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
        provider: Some(settings.provider.as_config_str().to_string()),
        debug: false,
        ..Default::default()
    };

    OfflineRecognizer::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create OfflineRecognizer"))
}

pub fn clean_transcription(text: &str) -> String {
    // Fast path: if no tags present, just trim (avoids any allocation from replace)
    let needs_cleaning = STRIP_TAGS.iter().any(|tag| text.contains(tag));
    if !needs_cleaning {
        let trimmed = text.trim();
        if trimmed.len() == text.len() {
            return text.to_string();
        }
        return trimmed.to_string();
    }

    // Slow path: allocate once, strip in-place
    let mut result = text.to_string();
    for tag in STRIP_TAGS {
        // Skip the contains() check — replace() on a miss is already O(n) with no alloc
        // But we already know at least one tag is present
        result = result.replace(tag, "");
    }

    // Some Qwen ASR models hallucinate Chinese text or common short phrases on pure noise.
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
    pub fn new(model_dir: &Path, settings: &Settings) -> anyhow::Result<Self> {
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
        // Use chunks_exact to help LLVM auto-vectorize the sum-of-squares
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

    pub fn transcribe_samples(&self, sample_rate: i32, samples: &[f32]) -> anyhow::Result<String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate, samples);
        self.recognizer.decode(&stream);
        let result = stream.get_result();
        Ok(result.map(|r| r.text).unwrap_or_default())
    }

    pub fn transcribe_file(&self, wav_path: &Path) -> anyhow::Result<String> {
        let wave = Wave::read(wav_path.to_str().unwrap_or(""))
            .ok_or_else(|| anyhow::anyhow!("Failed to read WAV file"))?;
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(wave.sample_rate(), wave.samples());
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| anyhow::anyhow!("Failed to get recognition result"))?;
        Ok(result.text)
    }
}
