use eframe::egui;
use serde::{Deserialize, Serialize};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use parking_lot::Mutex;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    Wave,
};
use std::borrow::Cow;
use std::io::{BufWriter, Read, Write};
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

const MODEL_REPO: &str = "ilmina/eng";

const MODEL_FILES: &[&str] = &[
    "encoder_model.ort",
    "decoder_model_merged.ort",
    "tokens.txt",
];

const STRIP_TAGS: &[&str] = &[
    "<|en|>", "<|zh|>", "<|ja|>", "<|ko|>", "<|fr|>",
    "<|de|>", "<|it|>", "<|es|>", "<|ru|>", "<|asr|>", "<|text|>",
];

#[derive(Debug, Clone)]
enum DownloadEvent {
    Progress { file: Arc<str>, bytes: u64, total: u64 },
    FileDone(String),
    Error(String),
    AllDone,
}

enum InitEvent {
    Success(Arc<AsrEngine>),
    Error(String),
}

#[derive(Debug, Clone)]
enum TranscriptionEvent {
    FinalResult(String),
    Error(String),
}

fn build_recognizer(model_dir: &Path, settings: &Settings) -> anyhow::Result<OfflineRecognizer> {
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
        moonshine: sherpa_onnx::OfflineMoonshineModelConfig {
            encoder: Some(model_dir.join("encoder_model.ort").to_string_lossy().into_owned()),
            merged_decoder: Some(model_dir.join("decoder_model_merged.ort").to_string_lossy().into_owned()),
            ..Default::default()
        },
        tokens: Some(model_dir.join("tokens.txt").to_string_lossy().into_owned()),
        num_threads: settings.num_threads,
        provider: Some(settings.provider.as_config_str().to_string()),
        debug: false,
        ..Default::default()
    };

    OfflineRecognizer::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create OfflineRecognizer"))
}

fn clean_transcription(text: &str) -> String {
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
        // ⚡ Bolt: Check contains() before replace() since replace() ALWAYS allocates in Rust
        if result.contains(tag) {
            result = result.replace(tag, "");
        }
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

#[derive(Debug, Clone)]
pub enum ModelStatus {
    NotDownloaded,
    Downloading { current_file: Arc<str>, current_bytes: u64, total_bytes: u64 },
    Initializing,
    Ready,
    Error(String),
}

impl PartialEq for ModelStatus {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for ModelStatus {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Main,
    Settings,
    Debug,
    Help,
}

impl Default for View {
    fn default() -> Self {
        View::Main
    }
}

pub struct PerihelionApp {
    view: View,
    model_status: ModelStatus,
    download_rx: Option<Receiver<DownloadEvent>>,
    init_rx: Option<Receiver<InitEvent>>,
    osc_rx: Option<Receiver<bool>>,
    transcription_rx: Option<Receiver<TranscriptionEvent>>,
    recognized_text: String,
    is_listening: bool,
    status_message: Cow<'static, str>,
    settings: Settings,
    test_file_path: String,
    engine: Option<Arc<AsrEngine>>,
    audio_running: Arc<AtomicBool>,
    model_dir: PathBuf,
    osc_socket: Option<UdpSocket>,
    available_devices: Vec<String>,
    selected_device: usize,
}

impl PerihelionApp {
    fn model_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("perihelion")
            .join("models")
            .join("moonshine-english")
    }

    fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("perihelion")
            .join("config.json")
    }

    fn load_config() -> Option<AppConfig> {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(config) = serde_json::from_str(&contents) {
                    return Some(config);
                }
            }
        }
        None
    }

    fn save_config(&self) {
        let config = AppConfig {
            settings: self.settings.clone(),
            selected_device: self.selected_device,
        };
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&config) {
            let _ = std::fs::write(&path, json);
        }
    }

    fn check_model_exists(model_dir: &Path) -> bool {
        MODEL_FILES.iter().all(|f| model_dir.join(f).exists())
    }

    fn get_available_devices() -> Vec<String> {
        use cpal::traits::{DeviceTrait, HostTrait};
        let host = cpal::default_host();
        let mut devices = Vec::new();

        // Add default device name
        if let Some(device) = host.default_input_device() {
            if let Ok(desc) = device.description() {
                devices.push(format!("{} (default)", desc.name()));
            }
        }

        // Add all other devices
        if let Ok(device_iter) = host.input_devices() {
            for device in device_iter {
                if let Ok(desc) = device.description() {
                    let name = desc.name();
                    // Skip if it's the default device we already added
                    if !devices.iter().any(|d| d.contains(name)) {
                        devices.push(name.to_string());
                    }
                }
            }
        }

        // If no devices found, add a placeholder
        if devices.is_empty() {
            devices.push("No devices found".to_string());
        }

        devices
    }

    fn start_download(&mut self) {
        self.model_status = ModelStatus::Downloading {
            current_file: Arc::from(MODEL_FILES[0]),
            current_bytes: 0,
            total_bytes: 0,
        };
        let model_dir = self.model_dir.clone();
        let (tx, rx) = channel();
        self.download_rx = Some(rx);
        let files: Vec<String> = MODEL_FILES.iter().map(|f| f.to_string()).collect();

        thread::spawn(move || {
            std::fs::create_dir_all(&model_dir).ok();
            for file in &files {
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    MODEL_REPO, file
                );
                let dest = model_dir.join(file);
                if dest.exists() {
                    let _ = tx.send(DownloadEvent::FileDone(file.clone()));
                    continue;
                }
                let file_arc: Arc<str> = Arc::from(file.as_str());
                if let Err(e) = download_file_with_progress(&url, &dest, &tx, &file_arc) {
                    let _ = tx.send(DownloadEvent::Error(format!("{}: {}", file, e)));
                    return;
                }
                let _ = tx.send(DownloadEvent::FileDone(file.clone()));
            }
            let _ = tx.send(DownloadEvent::AllDone);
        });
    }

    fn start_init_engine(&mut self) {
        self.model_status = ModelStatus::Initializing;
        self.status_message = Cow::Borrowed("Initializing...");
        let (tx, rx) = channel();
        self.init_rx = Some(rx);
        let model_dir = self.model_dir.clone();
        let settings = self.settings.clone();

        thread::spawn(move || {
            if Self::check_model_exists(&model_dir) {
                match AsrEngine::new(&model_dir, &settings) {
                    Ok(engine) => {
                        let _ = tx.send(InitEvent::Success(Arc::new(engine)));
                    }
                    Err(e) => {
                        let _ = tx.send(InitEvent::Error(format!("Engine error: {}", e)));
                    }
                }
            } else {
                let _ = tx.send(InitEvent::Error("Model files not found".to_string()));
            }
        });
    }

    fn handle_init_events(&mut self) {
        let event = if let Some(rx) = &self.init_rx {
            rx.try_recv().ok()
        } else {
            None
        };
        if let Some(event) = event {
            match event {
                InitEvent::Success(engine) => {
                    self.engine = Some(engine);
                    self.model_status = ModelStatus::Ready;
                    self.status_message = Cow::Borrowed("Ready");
                }
                InitEvent::Error(e) => {
                    self.model_status = ModelStatus::Error(e);
                    self.status_message = Cow::Borrowed("Initialization failed");
                }
            }
            self.init_rx = None;
        }
    }

    fn handle_download_events(&mut self) {
        let Some(rx) = self.download_rx.take() else { return };
        while let Ok(event) = rx.try_recv() {
            match event {
                DownloadEvent::Progress {
                    file,
                    bytes,
                    total,
                } => {
                    self.model_status = ModelStatus::Downloading {
                        current_file: file,
                        current_bytes: bytes,
                        total_bytes: total,
                    };
                }
                DownloadEvent::FileDone(file) => {
                    self.status_message = Cow::Owned(format!("Downloaded: {}", file));
                }
                DownloadEvent::Error(e) => {
                    self.model_status = ModelStatus::Error(e);
                }
                DownloadEvent::AllDone => {
                    // Don't put rx back — download is complete
                    self.start_init_engine();
                    return;
                }
            }
        }
        self.download_rx = Some(rx);
    }

    fn ensure_osc_listener(&mut self, ctx: &egui::Context) {
        if self.settings.listening_mode == ListeningMode::ToggleOsc && self.osc_rx.is_none() {
            let (tx, rx) = channel();
            self.osc_rx = Some(rx);
            let ctx_clone = ctx.clone();
            thread::spawn(move || run_osc_listener(tx, ctx_clone));
        }
        if self.settings.listening_mode != ListeningMode::ToggleOsc {
            self.osc_rx = None;
        }
    }

    fn handle_osc_events(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.osc_rx.take() else { return };
        while let Ok(on) = rx.try_recv() {
            self.set_listening(on, ctx);
            self.status_message = Cow::Borrowed(if on { "OSC: On" } else { "OSC: Off" });
        }
        self.osc_rx = Some(rx);
    }

    fn handle_transcription_events(&mut self) {
        let Some(rx) = self.transcription_rx.take() else { return };
        while let Ok(event) = rx.try_recv() {
            match event {
                TranscriptionEvent::FinalResult(text) => {
                    self.append_text(&text);
                }
                TranscriptionEvent::Error(e) => {
                    self.status_message = Cow::Owned(format!("Transcription error: {}", e));
                }
            }
        }
        self.transcription_rx = Some(rx);
    }

    fn start_audio_capture(&mut self, ctx: &egui::Context) {
        if self.engine.is_none() {
            self.status_message = Cow::Borrowed("Engine not ready");
            return;
        }

        let engine = self.engine.clone().unwrap();
        let (tx, rx) = channel();
        self.transcription_rx = Some(rx);
        self.audio_running = Arc::new(AtomicBool::new(true));
        let running = self.audio_running.clone();
        let device_index = self.selected_device;
        let ctx_clone = ctx.clone();

        thread::spawn(move || {
            run_audio_capture(tx, engine, running, device_index, ctx_clone);
        });
    }

    fn stop_audio_capture(&mut self) {
        self.audio_running.store(false, Ordering::Relaxed);
        self.transcription_rx = None;
    }

    fn sync_listening_state(&mut self, ctx: &egui::Context) {
        match self.settings.listening_mode {
            ListeningMode::AlwaysOn => {
                if !self.is_listening {
                    self.set_listening(true, ctx);
                    self.status_message = Cow::Borrowed("Always On");
                }
            }
            ListeningMode::ToggleButton => {
                // state is managed by button clicks
            }
            ListeningMode::ToggleOsc => {
                // state is managed by OSC events
            }
        }
    }

    fn copy_to_clipboard(&self, text: &str) {
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    }

    fn append_text(&mut self, text: &str) {
        if self.settings.append_mode {
            if !self.recognized_text.is_empty() {
                self.recognized_text.push('\n');
            }
            self.recognized_text.push_str(text);
        } else {
            self.recognized_text = text.to_string();
        }
        if !text.is_empty() {
            self.send_osc_chatbox(text);
        }
        if self.settings.auto_copy {
            self.copy_to_clipboard(&self.recognized_text);
            self.status_message = Cow::Borrowed("Copied");
        }
    }

    fn clear_text(&mut self) {
        self.recognized_text.clear();
        self.status_message = Cow::Borrowed("Cleared");
    }

    fn set_listening(&mut self, listening: bool, ctx: &egui::Context) {
        if self.is_listening != listening {
            self.is_listening = listening;
            if listening {
                self.start_audio_capture(ctx);
            } else {
                self.stop_audio_capture();
            }
            self.send_osc_typing(listening);
        }
    }

    fn send_osc_message(&self, addr: &str, args: Vec<rosc::OscType>) {
        if let Some(socket) = &self.osc_socket {
            let msg = rosc::OscPacket::Message(rosc::OscMessage {
                addr: addr.to_string(),
                args,
            });
            if let Ok(bytes) = rosc::encoder::encode(&msg) {
                let _ = socket.send(&bytes);
            }
        }
    }

    fn send_osc_chatbox(&self, text: &str) {
        // Avoid allocation for short messages (the common case)
        let msg: Cow<'_, str> = if text.len() > 140 && text.chars().count() > 140 {
            Cow::Owned(text.chars().take(140).collect())
        } else {
            Cow::Borrowed(text)
        };
        self.send_osc_message("/chatbox/input", vec![
            rosc::OscType::String(msg.into_owned()),
            rosc::OscType::Bool(true),
        ]);
    }

    fn send_osc_typing(&self, typing: bool) {
        self.send_osc_message("/chatbox/typing", vec![rosc::OscType::Bool(typing)]);
    }

    fn start_listening(&mut self, ctx: &egui::Context) {
        self.set_listening(true, ctx);
        self.status_message = Cow::Borrowed("Listening...");
    }

    fn stop_listening(&mut self, ctx: &egui::Context) {
        self.set_listening(false, ctx);
        self.status_message = Cow::Borrowed("Ready");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppConfig {
    settings: Settings,
    selected_device: usize,
}

impl Default for PerihelionApp {
    fn default() -> Self {
        let model_dir = Self::model_dir();
        let model_exists = Self::check_model_exists(&model_dir);
        let osc_socket = UdpSocket::bind("0.0.0.0:0").ok().and_then(|s| {
            s.connect("127.0.0.1:9000").ok().map(|_| s)
        });
        let available_devices = Self::get_available_devices();

        let (settings, selected_device) = match Self::load_config() {
            Some(c) => (c.settings, c.selected_device),
            None => (Settings::default(), 0),
        };
        let selected_device = selected_device
            .min(available_devices.len().saturating_sub(1));

        let mut app = Self {
            view: View::default(),
            model_status: if model_exists {
                ModelStatus::Ready
            } else {
                ModelStatus::NotDownloaded
            },
            download_rx: None,
            init_rx: None,
            osc_rx: None,
            transcription_rx: None,
            recognized_text: String::new(),
            is_listening: false,
            status_message: Cow::Borrowed("Ready"),
            settings,
            test_file_path: String::new(),
            engine: None,
            audio_running: Arc::new(AtomicBool::new(false)),
            model_dir,
            osc_socket,
            available_devices,
            selected_device,
        };
        if model_exists {
            app.start_init_engine();
        }
        app
    }
}

impl eframe::App for PerihelionApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_config();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_download_events();
        self.handle_init_events();
        self.handle_transcription_events();
        self.ensure_osc_listener(ctx);
        self.handle_osc_events(ctx);
        self.sync_listening_state(ctx);

        if matches!(self.model_status, ModelStatus::Downloading { .. } | ModelStatus::Initializing) || self.is_listening {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        if self.model_status == ModelStatus::Initializing {
            egui::CentralPanel::default()
                .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(egui::Margin::same(20.0)))
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.available_height() / 2.0 - 40.0);
                        ui.add_sized([40.0, 40.0], egui::Spinner::new());
                        ui.add_space(20.0);
                        ui.heading(egui::RichText::new("Hang on, the model's starting!").size(24.0));
                    });
                });
            return;
        }

        // Sidebar
        egui::SidePanel::left("sidebar")
            .resizable(false)
            .exact_width(100.0)
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .inner_margin(egui::Margin::same(6.0)),
            )
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 6.0);
                    ui.add_space(4.0);

                    let mut sel =
                        |ui: &mut egui::Ui, label: &str, view: View| {
                            let active = self.view == view;
                            if ui
                                .add_sized(
                                    [88.0, 28.0],
                                    egui::SelectableLabel::new(active, label),
                                )
                                .clicked()
                            {
                                self.view = view;
                            }
                        };

                    sel(ui, "Main", View::Main);
                    sel(ui, "Settings", View::Settings);
                    sel(ui, "Debug", View::Debug);
                    sel(ui, "Help", View::Help);
                });
            });

        // Content
        let frame = egui::Frame::central_panel(&ctx.style())
            .inner_margin(egui::Margin::same(20.0));

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            let avail = ui.available_width();

            match self.view {
                View::Main => {
                    // Title + model status
                    ui.horizontal(|ui| {
                        ui.heading("Perihelion");
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                match &self.model_status {
                                    ModelStatus::Ready => {
                                        ui.colored_label(egui::Color32::GREEN, "ready");
                                    }
                                    ModelStatus::NotDownloaded => {
                                        ui.colored_label(egui::Color32::YELLOW, "missing");
                                    }
                                    ModelStatus::Downloading { .. } => {
                                        ui.add_sized(
                                            [12.0, 12.0],
                                            egui::Spinner::new(),
                                        );
                                        ui.label("downloading");
                                    }
                                    ModelStatus::Initializing => {
                                        ui.add_sized(
                                            [12.0, 12.0],
                                            egui::Spinner::new(),
                                        );
                                        ui.label("initializing");
                                    }
                                    ModelStatus::Error(_) => {
                                        ui.colored_label(egui::Color32::RED, "error");
                                    }
                                }
                                ui.label("Model:");
                            },
                        );
                    });

                    ui.separator();

                    // Download / progress
                    match &self.model_status {
                        ModelStatus::NotDownloaded => {
                            if ui
                                .add_sized(
                                    [avail, 28.0],
                                    egui::Button::new("Download ~1 GB"),
                                )
                                .clicked()
                            {
                                self.start_download();
                            }
                        }
                        ModelStatus::Downloading {
                            current_file,
                            current_bytes,
                            total_bytes,
                        } => {
                            if *total_bytes > 0 {
                                let progress = *current_bytes as f32 / *total_bytes as f32;
                                ui.add_sized(
                                    [avail, 16.0],
                                    egui::ProgressBar::new(progress).text(format!(
                                        "{}  {} / {} MB",
                                        current_file,
                                        current_bytes / 1_000_000,
                                        total_bytes / 1_000_000
                                    )),
                                );
                            } else {
                                ui.add_sized(
                                    [avail, 16.0],
                                    egui::ProgressBar::new(0.0).text("Starting..."),
                                );
                            }
                        }
                        ModelStatus::Initializing => {
                            ui.add_sized(
                                [avail, 16.0],
                                egui::ProgressBar::new(1.0).text("Initializing..."),
                            );
                        }
                        ModelStatus::Error(e) => {
                            let err_msg = e.clone();
                            ui.horizontal(|ui| {
                                ui.colored_label(egui::Color32::RED, &err_msg);
                                if ui.button("Retry").clicked() {
                                    self.start_download();
                                }
                            });
                        }
                        ModelStatus::Ready => {}
                    }

                    // Mode label
                    ui.horizontal(|ui| {
                        ui.label("Mode:");
                        ui.label(self.settings.listening_mode.as_str());
                    });

                    // Listen button
                    let text = if self.is_listening { "On" } else { "Off" };
                    let color = if self.is_listening {
                        egui::Color32::from_rgb(0, 150, 0)
                    } else {
                        egui::Color32::from_rgb(80, 80, 80)
                    };
                    let button =
                        egui::Button::new(egui::RichText::new(text).size(16.0).strong())
                            .fill(color);

                    let can_click = self.settings.listening_mode
                        == ListeningMode::ToggleButton;
                    let response = ui.add_sized([avail, 40.0], button);
                    if can_click && response.clicked() {
                        if self.is_listening {
                            self.stop_listening(ctx);
                        } else {
                            self.start_listening(ctx);
                        }
                    }

                    // Text output
                    ui.horizontal(|ui| {
                        ui.label("Output");
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.button("Copy").clicked() {
                                    self.copy_to_clipboard(&self.recognized_text);
                                    self.status_message = Cow::Borrowed("Copied");
                                }
                                if ui.button("Clear").clicked() {
                                    self.clear_text();
                                }
                            },
                        );
                    });

                    egui::ScrollArea::vertical().max_height(70.0).show(ui, |ui| {
                        ui.add_sized(
                            [avail, 52.0],
                            egui::TextEdit::multiline(&mut self.recognized_text.as_str())
                                .font(egui::TextStyle::Monospace)
                                .interactive(false),
                        );
                    });

                    // Status
                    ui.horizontal(|ui| {
                        ui.label(self.status_message.as_ref());
                        if self.is_listening {
                            ui.spinner();
                        }
                    });
                }

                View::Settings => {
                    ui.heading(egui::RichText::new("Settings").size(22.0).strong());
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.checkbox(
                                &mut self.settings.auto_copy,
                                "Auto-copy to clipboard",
                            );
                            ui.add_space(4.0);
                            ui.checkbox(
                                &mut self.settings.append_mode,
                                "Append transcriptions",
                            );
                        });

                        ui.add_space(16.0);
                        ui.label(egui::RichText::new("Activation Mode").size(16.0).strong());
                        ui.add_space(4.0);
                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.radio_value(
                                    &mut self.settings.listening_mode,
                                    ListeningMode::ToggleButton,
                                    "Toggle-Button (Manual)",
                                );
                                ui.radio_value(
                                    &mut self.settings.listening_mode,
                                    ListeningMode::AlwaysOn,
                                    "Always On",
                                );
                                ui.radio_value(
                                    &mut self.settings.listening_mode,
                                    ListeningMode::ToggleOsc,
                                    "OSC (VRChat)",
                                );
                            });
                        });

                        ui.add_space(16.0);
                        ui.label(egui::RichText::new("Microphone").size(16.0).strong());
                        ui.add_space(4.0);
                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let device_names: Vec<&str> = self.available_devices.iter().map(|s| s.as_str()).collect();
                                egui::ComboBox::from_id_salt("device_cb")
                                    .selected_text(
                                        self.available_devices.get(self.selected_device)
                                            .map(|s| s.as_str())
                                            .unwrap_or("Unknown")
                                    )
                                    .width(260.0)
                                    .show_ui(ui, |ui| {
                                        for (i, device_name) in device_names.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut self.selected_device,
                                                i,
                                                *device_name,
                                            );
                                        }
                                    });

                                if ui.button("↻ Refresh").clicked() {
                                    self.available_devices = Self::get_available_devices();
                                    self.selected_device = 0;
                                    self.status_message = Cow::Borrowed("Devices refreshed");
                                }
                            });
                        });

                        ui.add_space(20.0);
                        ui.heading(egui::RichText::new("AI Preferences").size(18.0).strong());
                        ui.add_space(8.0);

                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.checkbox(&mut self.settings.high_accuracy, "High Accuracy Mode (Modified Beam Search)");
                            if self.settings.high_accuracy {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.label("Search Depth:");
                                    ui.add(egui::DragValue::new(&mut self.settings.search_depth).range(1..=10));
                                });
                                ui.label(egui::RichText::new("Higher = More accurate but requires more CPU power").color(egui::Color32::GRAY));
                            }

                            ui.add_space(12.0);
                            ui.label("Hardware Processor:");
                            ui.horizontal_wrapped(|ui| {
                                ui.radio_value(&mut self.settings.provider, Provider::Cpu, "CPU");
                                ui.radio_value(&mut self.settings.provider, Provider::Dml, "GPU");
                                ui.radio_value(&mut self.settings.provider, Provider::Cuda, "GPU-CUDA)");
                            });

                            ui.add_space(12.0);
                            ui.horizontal(|ui| {
                                ui.label("CPU Threads:");
                                ui.add(egui::DragValue::new(&mut self.settings.num_threads).speed(1).range(1..=16));
                            });
                        });

                        ui.add_space(16.0);
                        ui.label(egui::RichText::new("Custom Dictionary").size(16.0).strong());
                        ui.add(egui::Label::new(egui::RichText::new("Add terms to help the AI recognize them").color(egui::Color32::GRAY)).wrap());
                        ui.add_space(4.0);
                        ui.add(
                            egui::TextEdit::multiline(&mut self.settings.hotwords)
                                .desired_rows(4)
                                .desired_width(ui.available_width())
                                .font(egui::TextStyle::Monospace)
                        );
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label("Hotword Focus:");
                            ui.add(egui::Slider::new(&mut self.settings.hotwords_boost, 0.0..=10.0).text("Boost multiplier"));
                        });

                        ui.add_space(16.0);
                        if ui.button(egui::RichText::new("Apply Settings & Reload").size(16.0)).clicked() {
                            self.save_config();
                            self.start_init_engine();
                        }

                        ui.add_space(20.0);
                        ui.heading(egui::RichText::new("Danger Zone").size(18.0).strong().color(egui::Color32::RED));
                        ui.add_space(8.0);
                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.label("If the model's fucked, you can delete it to reinstall it.");
                            ui.add_space(4.0);
                            if ui.button(egui::RichText::new("Annihilate Model").size(16.0).color(egui::Color32::RED)).clicked() {
                                self.set_listening(false, ctx);
                                let _ = std::fs::remove_dir_all(&self.model_dir);
                                self.engine = None;
                                self.model_status = ModelStatus::NotDownloaded;
                                self.status_message = Cow::Borrowed("Model files gone");
                            }
                            ui.add_space(8.0);
                            ui.label("Reset all settings to their factory defaults.");
                            ui.add_space(4.0);
                            if ui.button(egui::RichText::new("Reset Config").size(16.0).color(egui::Color32::RED)).clicked() {
                                self.settings = Settings::default();
                                self.selected_device = 0;
                                self.save_config();
                                self.status_message = Cow::Borrowed("Config reset");
                            }
                        });
                    });
                }

                View::Debug => {
                    ui.heading(egui::RichText::new("Debug").size(22.0).strong());
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [avail * 0.7, 24.0],
                            egui::TextEdit::singleline(&mut self.test_file_path).hint_text("Path to .wav file"),
                        );
                        if ui.button("Run").clicked() {
                            if let Some(engine) = &self.engine {
                                let path = Path::new(&self.test_file_path);
                                match engine.transcribe_file(path) {
                                    Ok(text) => self.append_text(&text),
                                    Err(e) => {
                                        self.status_message = Cow::Owned(format!("Error: {}", e));
                                    }
                                }
                            } else {
                                self.status_message = Cow::Borrowed("Model not ready");
                            }
                        }
                    });
                }

                View::Help => {
                    ui.heading(egui::RichText::new("Help & Tips").size(22.0).strong());
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.label(egui::RichText::new("Perihelion is an AI-powered speech-to-text app for VRC.").size(14.0));
                        ui.add_space(16.0);

                        ui.label(egui::RichText::new("Listening Modes").size(16.0).strong());
                        ui.add_space(4.0);
                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.label(egui::RichText::new("Toggle-Button:").strong());
                            ui.add(egui::Label::new("Click the big Start/Stop button on the Main tab to dictate manually.").wrap());
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("OSC (VRChat):").strong());
                            ui.add(egui::Label::new("Listens when the VRChat avatar parameter \"perihelion\" is ON in your avatar, whether via toggle or contact or whatever.").wrap());
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("Always On:").strong());
                            ui.add(egui::Label::new("The microphone remains active no matter what. Probably not a good idea.").wrap());
                        });
                    });
                }
            }
        });
    }
}

fn download_file_with_progress(
    url: &str,
    dest: &Path,
    tx: &std::sync::mpsc::Sender<DownloadEvent>,
    filename: &Arc<str>,
) -> anyhow::Result<()> {
    let response = ureq::get(url).call()
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;
    let total_size = response
        .headers()
        .get("Content-Length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    // Write to a .part file first — if we crash mid-download, the incomplete file
    // won't be mistaken for a valid model file on next launch
    let tmp_dest = dest.with_extension("part");
    let mut file = BufWriter::new(std::fs::File::create(&tmp_dest)?);
    let mut buffer = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_reported: u64 = 0;
    let mut reader = response.into_body().into_reader();
    while let Ok(n) = reader.read(&mut buffer) {
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;
        if downloaded - last_reported >= 262_144 || downloaded == total_size {
            last_reported = downloaded;
            // Arc::clone is O(1) — just an atomic refcount increment
            let _ = tx.send(DownloadEvent::Progress {
                file: Arc::clone(filename),
                bytes: downloaded,
                total: total_size,
            });
        }
    }
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp_dest, dest)?;
    Ok(())
}

fn extract_perihelion_value(packet: &rosc::OscPacket) -> Option<bool> {
    match packet {
        rosc::OscPacket::Message(msg) => {
            if msg.addr == "/avatar/parameters/perihelion" {
                return match &msg.args[..] {
                    [rosc::OscType::Float(v)] => Some(*v > 0.5),
                    [rosc::OscType::Bool(v)] => Some(*v),
                    [rosc::OscType::Int(v)] => Some(*v > 0),
                    _ => Some(false),
                };
            }
            None
        }
        rosc::OscPacket::Bundle(bundle) => {
            for packet in &bundle.content {
                if let Some(v) = extract_perihelion_value(packet) {
                    return Some(v);
                }
            }
            None
        }
    }
}

fn run_osc_listener(tx: Sender<bool>, ctx: egui::Context) {
    let socket = match UdpSocket::bind("0.0.0.0:9001") {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut buf = [0u8; rosc::decoder::MTU];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((size, _)) => {
                if let Ok((_, packet)) = rosc::decoder::decode_udp(&buf[..size]) {
                    if let Some(on) = extract_perihelion_value(&packet) {
                        if tx.send(on).is_err() {
                            break;
                        }
                        ctx.request_repaint();
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn resample_linear(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if in_rate == out_rate {
        return input.to_vec();
    }

    // For the common 48kHz→16kHz (ratio=3) or 44.1kHz→16kHz case,
    // fuse the anti-alias filter directly into the interpolation to avoid
    // allocating an intermediate buffer
    let ratio = in_rate as f32 / out_rate as f32;
    if in_rate > out_rate {
        let factor = (in_rate / out_rate) as usize;
        if factor > 1 && input.len() >= factor {
            let inv_factor = 1.0_f32 / factor as f32;
            let filtered_len = input.len() - factor + 1;
            let actual_out_len = (filtered_len as f32 / ratio).ceil() as usize;
            let mut out = Vec::with_capacity(actual_out_len);

            for i in 0..actual_out_len {
                let in_pos = i as f32 * ratio;
                let idx = in_pos as usize; // floor for positive values
                let frac = in_pos - idx as f32;

                // Inline the moving-average filter for each needed sample
                if idx + 1 < filtered_len {
                    let s0: f32 = input[idx..idx + factor].iter().sum::<f32>() * inv_factor;
                    let s1: f32 = input[idx + 1..idx + 1 + factor].iter().sum::<f32>() * inv_factor;
                    out.push(s0 * (1.0 - frac) + s1 * frac);
                } else if idx < filtered_len {
                    let s0: f32 = input[idx..idx + factor].iter().sum::<f32>() * inv_factor;
                    out.push(s0);
                }
            }
            return out;
        }
    }

    // General case for upsampling or non-integer ratios
    let out_len = (input.len() as f32 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let in_pos = i as f32 * ratio;
        let idx = in_pos as usize;
        let frac = in_pos - idx as f32;
        if idx + 1 < input.len() {
            out.push(input[idx] * (1.0 - frac) + input[idx + 1] * frac);
        } else if idx < input.len() {
            out.push(input[idx]);
        }
    }
    out
}

fn run_audio_capture(
    tx: Sender<TranscriptionEvent>,
    engine: Arc<AsrEngine>,
    running: Arc<AtomicBool>,
    device_index: usize,
    ctx: egui::Context,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();

    // Get the device at the specified index
    let device = {
        let mut device_iter = match host.input_devices() {
            Ok(iter) => iter,
            Err(e) => {
                let _ = tx.send(TranscriptionEvent::Error(format!("Failed to get devices: {}", e)));
                ctx.request_repaint();
                return;
            }
        };

        match device_iter.nth(device_index) {
            Some(d) => d,
            None => {
                let _ = tx.send(TranscriptionEvent::Error("Selected device not found".to_string()));
                ctx.request_repaint();
                return;
            }
        }
    };

    let config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(TranscriptionEvent::Error(format!("Failed to get config: {}", e)));
            ctx.request_repaint();
            return;
        }
    };

    let sample_rate = config.sample_rate() as i32;
    let channels = config.channels() as usize;
    // Process every 0.1 seconds of audio
    let chunk_size = ((sample_rate as usize) / 10).max(1024);

    // Clear buffer from previous runs before starting capture
    engine.clear_buffer();

    macro_rules! build_stream {
        ($device:expr, $config:expr, $running:expr, $engine:expr, $ch:expr, $ty:ty, $convert:expr) => {{
            let running = $running.clone();
            let engine = $engine.clone();
            $device.build_input_stream(
                &$config.config(),
                move |data: &[$ty], _: &cpal::InputCallbackInfo| {
                    if !running.load(Ordering::Relaxed) { return; }
                    if $ch == 1 {
                        engine.extend_samples(data.iter().map($convert));
                    } else {
                        engine.extend_samples(data.iter().step_by($ch).map($convert));
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )
        }};
    }

    let stream_result = match config.sample_format() {
        cpal::SampleFormat::F32 => build_stream!(device, config, running, engine, channels, f32, |&v: &f32| v),
        cpal::SampleFormat::I16 => build_stream!(device, config, running, engine, channels, i16, |&v: &i16| v as f32 / 32768.0),
        cpal::SampleFormat::U16 => build_stream!(device, config, running, engine, channels, u16, |&v: &u16| (v as f32 - 32768.0) / 32768.0),
        _ => {
            let _ = tx.send(TranscriptionEvent::Error("Unsupported audio sample format".to_string()));
            ctx.request_repaint();
            return;
        }
    };

    match stream_result {
        Ok(stream) => {
            if let Err(e) = stream.play() {
                let _ = tx.send(TranscriptionEvent::Error(format!("Failed to play stream: {}", e)));
                ctx.request_repaint();
                return;
            }

            let mut last_len = 0;
            let mut silence_chunks = 0;
            let mut is_speaking = false;

            // Keep the stream alive while running
            while running.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));

                if let Some((rms, total_len)) = engine.check_new_audio(last_len, chunk_size) {
                    last_len = total_len;

                    if rms < 0.005 {
                        silence_chunks += 1;
                    } else {
                        silence_chunks = 0;
                        is_speaking = true;
                    }

                    if !is_speaking {
                        // If we haven't started speaking yet, clear out old silence so it doesn't build up
                        if total_len > sample_rate as usize * 3 {
                            engine.clear_buffer();
                            last_len = 0;
                        }
                        continue;
                    }

                    // If user stopped speaking (0.6 seconds of silence = 6 chunks)
                    if silence_chunks >= 6 {
                        let samples_to_process = engine.clear_buffer();
                        last_len = 0;
                        silence_chunks = 0;
                        is_speaking = false;

                        if !samples_to_process.is_empty() {
                            let resampled = resample_linear(&samples_to_process, sample_rate as u32, 16000);
                            if let Ok(text) = engine.transcribe_samples(16000, &resampled) {
                                let clean_text = clean_transcription(&text);
                                if !clean_text.is_empty() {
                                    let _ = tx.send(TranscriptionEvent::FinalResult(clean_text));
                                    ctx.request_repaint();
                                }
                            }
                        }
                        continue;
                    }
                }
            }

            // Transcribe any remaining samples and clear out the buffer
            let samples = engine.clear_buffer();
            if !samples.is_empty() && is_speaking {
                let resampled = resample_linear(&samples, sample_rate as u32, 16000);
                if let Ok(text) = engine.transcribe_samples(16000, &resampled) {
                    let clean_text = clean_transcription(&text);
                    if !clean_text.is_empty() {
                        let _ = tx.send(TranscriptionEvent::FinalResult(clean_text));
                        ctx.request_repaint();
                    }
                }
            }
        }
        Err(e) => {
            let _ = tx.send(TranscriptionEvent::Error(format!("Failed to build stream: {}", e)));
            ctx.request_repaint();
        }
    }
}

fn load_custom_font(cc: &eframe::CreationContext<'_>) {
    let font_data_woff2 = include_bytes!("../assets/font.woff2");
    let title_font_data_woff2 = include_bytes!("../assets/title.woff2");

    let font_data_ttf = woff2_patched::convert_woff2_to_ttf(&mut std::io::Cursor::new(font_data_woff2))
        .expect("Failed to decode font.woff2 to TTF");
    let title_font_data_ttf = woff2_patched::convert_woff2_to_ttf(&mut std::io::Cursor::new(title_font_data_woff2))
        .expect("Failed to decode title.woff2 to TTF");

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "AraletN".to_owned(),
        std::sync::Arc::new(egui::FontData::from_owned(font_data_ttf)),
    );
    fonts.font_data.insert(
        "TitleFont".to_owned(),
        std::sync::Arc::new(egui::FontData::from_owned(title_font_data_ttf)),
    );
    fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .unwrap()
        .insert(0, "AraletN".to_owned());
    fonts
        .families
        .get_mut(&egui::FontFamily::Monospace)
        .unwrap()
        .insert(0, "AraletN".to_owned());
    fonts
        .families
        .insert(egui::FontFamily::Name("Title".into()), vec!["TitleFont".to_owned()]);

    cc.egui_ctx.set_fonts(fonts);
}

fn setup_custom_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.window_margin = egui::Margin::same(16.0);

    let rounding = egui::Rounding::same(8.0);
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = egui::Color32::from_rgb(24, 24, 28);
    visuals.window_fill = egui::Color32::from_rgb(30, 30, 34);

    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(40, 40, 46);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(50, 50, 58);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(70, 70, 80);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(90, 90, 105);

    visuals.widgets.noninteractive.rounding = rounding;
    visuals.widgets.inactive.rounding = rounding;
    visuals.widgets.hovered.rounding = rounding;
    visuals.widgets.active.rounding = rounding;
    visuals.widgets.open.rounding = rounding;
    visuals.window_rounding = egui::Rounding::same(12.0);

    style.visuals = visuals;

    if let Some(text_style) = style.text_styles.get_mut(&egui::TextStyle::Heading) {
        text_style.family = egui::FontFamily::Name("Title".into());
    }

    ctx.set_style(style);
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([540.0, 440.0])
            .with_min_inner_size([540.0, 440.0])
            .with_max_inner_size([540.0, 440.0])
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "Perihelion",
        options,
        Box::new(|cc| {
            load_custom_font(cc);
            setup_custom_style(&cc.egui_ctx);
            Ok(Box::new(PerihelionApp::default()))
        }),
    )
}
