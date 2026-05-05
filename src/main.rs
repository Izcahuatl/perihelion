use eframe::egui;
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    OfflineModelConfig, OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    Wave,
};
use std::io::{Read, Write};
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

const MODEL_REPO: &str = "pantinor/sherpa-onnx-qwen3-asr-0.6b-int8";

const MODEL_FILES: &[&str] = &[
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
];

#[derive(Debug, Clone)]
enum DownloadEvent {
    Progress { file: String, bytes: u64, total: u64 },
    FileDone(String),
    Error(String),
    AllDone,
}

enum InitEvent {
    Success(AsrEngine, Arc<StreamingEngine>),
    Error(String),
}

#[derive(Debug, Clone)]
enum TranscriptionEvent {
    PartialResult(String),
    FinalResult(String),
    Error(String),
}

pub struct StreamingEngine {
    recognizer: OfflineRecognizer,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
}

impl StreamingEngine {
    pub fn new(model_dir: &Path, settings: &Settings) -> anyhow::Result<Self> {
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
        config.decoding_method = Some(if settings.high_accuracy { "modified_beam_search".to_string() } else { "greedy_search".to_string() });
        config.max_active_paths = settings.search_depth;
        config.hotwords_file = Some(hotwords_file);
        config.hotwords_score = settings.hotwords_boost;

        config.model_config = OfflineModelConfig {
            qwen3_asr: OfflineQwen3ASRModelConfig {
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
            provider: Some(settings.provider.clone()),
            debug: false,
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow::anyhow!("Failed to create OfflineRecognizer"))?;

        Ok(Self {
            recognizer,
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn add_samples(&self, samples: &[f32]) {
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.extend_from_slice(samples);
    }

    pub fn extend_samples<I: IntoIterator<Item = f32>>(&self, iter: I) {
        let mut buffer = self.audio_buffer.lock().unwrap();
        buffer.extend(iter);
    }

    pub fn get_buffered_samples(&self, min_samples: usize) -> Option<Vec<f32>> {
        let buffer = self.audio_buffer.lock().unwrap();
        if buffer.len() >= min_samples {
            Some(buffer.clone())
        } else {
            None
        }
    }

    pub fn clear_buffer(&self) -> Vec<f32> {
        let mut buffer = self.audio_buffer.lock().unwrap();
        let len = buffer.len();
        buffer.drain(0..len).collect()
    }

    pub fn transcribe_samples(&self, sample_rate: i32, samples: &[f32]) -> anyhow::Result<String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate, samples);
        self.recognizer.decode(&stream);
        let result = stream.get_result();
        Ok(result.map(|r| r.text).unwrap_or_default())
    }
}

pub struct AsrEngine {
    recognizer: OfflineRecognizer,
}

impl AsrEngine {
    pub fn new(model_dir: &Path, settings: &Settings) -> anyhow::Result<Self> {
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
        config.decoding_method = Some(if settings.high_accuracy { "modified_beam_search".to_string() } else { "greedy_search".to_string() });
        config.max_active_paths = settings.search_depth;
        config.hotwords_file = Some(hotwords_file);
        config.hotwords_score = settings.hotwords_boost;

        config.model_config = OfflineModelConfig {
            qwen3_asr: OfflineQwen3ASRModelConfig {
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
            provider: Some(settings.provider.clone()),
            debug: false,
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow::anyhow!("Failed to create OfflineRecognizer"))?;
        Ok(Self { recognizer })
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
    pub provider: String,
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
            provider: "cpu".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    NotDownloaded,
    Downloading { current_file: String, current_bytes: u64, total_bytes: u64 },
    Initializing,
    Ready,
    Error(String),
}

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
    partial_text: String,
    is_listening: bool,
    status_message: String,
    settings: Settings,
    test_file_path: String,
    asr_engine: Option<AsrEngine>,
    streaming_engine: Option<Arc<StreamingEngine>>,
    audio_running: Arc<Mutex<bool>>,
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
            .join("qwen3-asr-0.6b-int8")
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
            current_file: MODEL_FILES[0].to_string(),
            current_bytes: 0,
            total_bytes: 0,
        };
        let model_dir = self.model_dir.clone();
        let (tx, rx) = channel();
        self.download_rx = Some(rx);
        let files: Vec<String> = MODEL_FILES.iter().map(|f| f.to_string()).collect();

        thread::spawn(move || {
            std::fs::create_dir_all(&model_dir).ok();
            std::fs::create_dir_all(model_dir.join("tokenizer")).ok();
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
                if let Err(e) = download_file_with_progress(&url, &dest, &tx, file) {
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
        self.status_message = "Initializing...".to_string();
        let (tx, rx) = channel();
        self.init_rx = Some(rx);
        let model_dir = self.model_dir.clone();
        let settings = self.settings.clone();

        thread::spawn(move || {
            if Self::check_model_exists(&model_dir) {
                match AsrEngine::new(&model_dir, &settings) {
                    Ok(asr) => {
                        match StreamingEngine::new(&model_dir, &settings) {
                            Ok(streaming) => {
                                let _ = tx.send(InitEvent::Success(asr, Arc::new(streaming)));
                            }
                            Err(e) => {
                                let _ = tx.send(InitEvent::Error(format!("Streaming Engine error: {}", e)));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(InitEvent::Error(format!("ASR Engine error: {}", e)));
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
                InitEvent::Success(asr, streaming) => {
                    self.asr_engine = Some(asr);
                    self.streaming_engine = Some(streaming);
                    self.model_status = ModelStatus::Ready;
                    self.status_message = "Ready".to_string();
                }
                InitEvent::Error(e) => {
                    self.model_status = ModelStatus::Error(e);
                    self.status_message = "Initialization failed".to_string();
                }
            }
            self.init_rx = None;
        }
    }

    fn handle_download_events(&mut self) {
        let events: Vec<DownloadEvent> = if let Some(rx) = &self.download_rx {
            rx.try_iter().collect()
        } else {
            Vec::new()
        };
        for event in events {
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
                    self.status_message = format!("Downloaded: {}", file);
                }
                DownloadEvent::Error(e) => {
                    self.model_status = ModelStatus::Error(e);
                }
                DownloadEvent::AllDone => {
                    self.start_init_engine();
                }
            }
        }
    }

    fn ensure_osc_listener(&mut self) {
        if self.settings.listening_mode == ListeningMode::ToggleOsc && self.osc_rx.is_none() {
            let (tx, rx) = channel();
            self.osc_rx = Some(rx);
            thread::spawn(move || run_osc_listener(tx));
        }
        if self.settings.listening_mode != ListeningMode::ToggleOsc {
            self.osc_rx = None;
        }
    }

    fn handle_osc_events(&mut self) {
        let events: Vec<bool> = if let Some(rx) = &self.osc_rx {
            rx.try_iter().collect()
        } else {
            Vec::new()
        };
        for on in events {
            self.set_listening(on);
            self.status_message = if on {
                "OSC: On".to_string()
            } else {
                "OSC: Off".to_string()
            };
        }
    }

    fn handle_transcription_events(&mut self) {
        let events: Vec<TranscriptionEvent> = if let Some(rx) = &self.transcription_rx {
            rx.try_iter().collect()
        } else {
            Vec::new()
        };
        for event in events {
            match event {
                TranscriptionEvent::PartialResult(text) => {
                    self.partial_text = text;
                }
                TranscriptionEvent::FinalResult(text) => {
                    self.append_text(&text);
                    self.partial_text.clear();
                }
                TranscriptionEvent::Error(e) => {
                    self.status_message = format!("Transcription error: {}", e);
                }
            }
        }
    }

    fn start_audio_capture(&mut self) {
        if self.streaming_engine.is_none() {
            self.status_message = "Streaming engine not ready".to_string();
            return;
        }

        let engine = self.streaming_engine.clone().unwrap();
        let (tx, rx) = channel();
        self.transcription_rx = Some(rx);
        self.audio_running = Arc::new(Mutex::new(true));
        let running = self.audio_running.clone();
        let device_index = self.selected_device;

        thread::spawn(move || {
            run_audio_capture(tx, engine, running, device_index);
        });
    }

    fn stop_audio_capture(&mut self) {
        *self.audio_running.lock().unwrap() = false;
        self.transcription_rx = None;
    }

    fn sync_listening_state(&mut self) {
        match self.settings.listening_mode {
            ListeningMode::AlwaysOn => {
                if !self.is_listening {
                    self.set_listening(true);
                    self.status_message = "Always On".to_string();
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
        if self.settings.append_mode && !self.recognized_text.is_empty() {
            self.recognized_text.push('\n');
        }
        self.recognized_text.push_str(text);
        if !text.is_empty() {
            self.send_osc_chatbox(text);
        }
        if self.settings.auto_copy {
            self.copy_to_clipboard(&self.recognized_text);
            self.status_message = "Copied".to_string();
        }
    }

    fn clear_text(&mut self) {
        self.recognized_text.clear();
        self.status_message = "Cleared".to_string();
    }

    fn set_listening(&mut self, listening: bool) {
        if self.is_listening != listening {
            self.is_listening = listening;
            if listening {
                self.start_audio_capture();
            } else {
                self.stop_audio_capture();
            }
            self.send_osc_typing(listening);
        }
    }

    fn send_osc_chatbox(&self, text: &str) {
        if let Some(socket) = &self.osc_socket {
            let truncated: String = text.chars().take(140).collect();
            let msg = rosc::OscPacket::Message(rosc::OscMessage {
                addr: "/chatbox/input".to_string(),
                args: vec![
                    rosc::OscType::String(truncated),
                    rosc::OscType::Bool(true),
                ],
            });
            if let Ok(bytes) = rosc::encoder::encode(&msg) {
                let _ = socket.send(&bytes);
            }
        }
    }

    fn send_osc_typing(&self, typing: bool) {
        if let Some(socket) = &self.osc_socket {
            let msg = rosc::OscPacket::Message(rosc::OscMessage {
                addr: "/chatbox/typing".to_string(),
                args: vec![rosc::OscType::Bool(typing)],
            });
            if let Ok(bytes) = rosc::encoder::encode(&msg) {
                let _ = socket.send(&bytes);
            }
        }
    }

    fn start_listening(&mut self) {
        self.set_listening(true);
        self.status_message = "Listening...".to_string();
    }

    fn stop_listening(&mut self) {
        self.set_listening(false);
        self.status_message = "Ready".to_string();
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

        let config = Self::load_config();
        let settings = config.as_ref().map(|c| c.settings.clone()).unwrap_or_default();
        let selected_device = config.map(|c| c.selected_device).unwrap_or(0)
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
            partial_text: String::new(),
            is_listening: false,
            status_message: "Ready".to_string(),
            settings,
            test_file_path: String::new(),
            asr_engine: None,
            streaming_engine: None,
            audio_running: Arc::new(Mutex::new(false)),
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
        self.ensure_osc_listener();
        self.handle_osc_events();
        self.sync_listening_state();

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
                            self.stop_listening();
                        } else {
                            self.start_listening();
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
                                    self.status_message = "Copied".to_string();
                                }
                                if ui.button("Clear").clicked() {
                                    self.clear_text();
                                }
                            },
                        );
                    });

                    egui::ScrollArea::vertical().max_height(70.0).show(ui, |ui| {
                        let mut text_display = if self.is_listening && !self.partial_text.is_empty() {
                            format!("{}\n{}", self.recognized_text, self.partial_text)
                        } else {
                            self.recognized_text.clone()
                        };

                        ui.add_sized(
                            [avail, 52.0],
                            egui::TextEdit::multiline(&mut text_display)
                                .font(egui::TextStyle::Monospace)
                                .interactive(false),
                        );
                    });

                    // Status
                    ui.horizontal(|ui| {
                        ui.label(self.status_message.clone());
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
                                    self.status_message = "Devices refreshed".to_string();
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
                                ui.radio_value(&mut self.settings.provider, "cpu".to_string(), "CPU");
                                ui.radio_value(&mut self.settings.provider, "dml".to_string(), "GPU");
                                ui.radio_value(&mut self.settings.provider, "cuda".to_string(), "GPU-CUDA)");
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
                                let _ = std::fs::remove_dir_all(&self.model_dir);
                                self.asr_engine = None;
                                self.streaming_engine = None;
                                self.model_status = ModelStatus::NotDownloaded;
                                self.status_message = "Model files gone".to_string();
                            }
                            ui.add_space(8.0);
                            ui.label("Reset all settings to their factory defaults.");
                            ui.add_space(4.0);
                            if ui.button(egui::RichText::new("Reset Config").size(16.0).color(egui::Color32::RED)).clicked() {
                                self.settings = Settings::default();
                                self.selected_device = 0;
                                self.save_config();
                                self.status_message = "Config reset".to_string();
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
                            if let Some(engine) = &self.asr_engine {
                                let path = Path::new(&self.test_file_path);
                                match engine.transcribe_file(path) {
                                    Ok(text) => self.append_text(&text),
                                    Err(e) => {
                                        self.status_message = format!("Error: {}", e);
                                    }
                                }
                            } else {
                                self.status_message = "Model not ready".to_string();
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
    filename: &str,
) -> anyhow::Result<()> {
    let mut response = reqwest::blocking::get(url)
        .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;
    let total_size = response.content_length().unwrap_or(0);
    let mut file = std::fs::File::create(dest)?;
    let mut buffer = [0u8; 8192];
    let mut downloaded: u64 = 0;
    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;
        let _ = tx.send(DownloadEvent::Progress {
            file: filename.to_string(),
            bytes: downloaded,
            total: total_size,
        });
    }
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

fn run_osc_listener(tx: Sender<bool>) {
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
    let ratio = in_rate as f32 / out_rate as f32;
    let out_len = (input.len() as f32 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let in_pos = i as f32 * ratio;
        let idx = in_pos.floor() as usize;
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
    streaming_engine: Arc<StreamingEngine>,
    running: Arc<Mutex<bool>>,
    device_index: usize,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();

    // Get the device at the specified index
    let device = {
        let mut device_iter = match host.input_devices() {
            Ok(iter) => iter,
            Err(e) => {
                let _ = tx.send(TranscriptionEvent::Error(format!("Failed to get devices: {}", e)));
                return;
            }
        };

        match device_iter.nth(device_index) {
            Some(d) => d,
            None => {
                let _ = tx.send(TranscriptionEvent::Error("Selected device not found".to_string()));
                return;
            }
        }
    };

    let config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(TranscriptionEvent::Error(format!("Failed to get config: {}", e)));
            return;
        }
    };

    let sample_rate = config.sample_rate() as i32;
    let channels = config.channels() as usize;
    // Process every 0.5 seconds of audio (or adapt based on sample rate)
    let chunk_size = ((sample_rate as usize) / 2).max(4096);

    // Create stream
    let engine_clone = streaming_engine.clone();
    let running_clone = running.clone();

    // Clear buffer from previous runs before starting capture
    engine_clone.clear_buffer();

    let stream_result = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &config.config(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !*running_clone.lock().unwrap() {
                        return;
                    }

                    if channels == 1 {
                        engine_clone.extend_samples(data.iter().copied());
                    } else {
                        engine_clone.extend_samples(data.iter().step_by(channels).copied());
                    }
                },
                |err| {
                    eprintln!("Stream error: {}", err);
                },
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &config.config(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !*running_clone.lock().unwrap() {
                        return;
                    }

                    let converter = |&v: &i16| v as f32 / 32768.0;
                    if channels == 1 {
                        engine_clone.extend_samples(data.iter().map(converter));
                    } else {
                        engine_clone.extend_samples(data.iter().step_by(channels).map(converter));
                    }
                },
                |err| {
                    eprintln!("Stream error: {}", err);
                },
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            device.build_input_stream(
                &config.config(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    if !*running_clone.lock().unwrap() {
                        return;
                    }

                    let converter = |&v: &u16| (v as f32 - 32768.0) / 32768.0;
                    if channels == 1 {
                        engine_clone.extend_samples(data.iter().map(converter));
                    } else {
                        engine_clone.extend_samples(data.iter().step_by(channels).map(converter));
                    }
                },
                |err| {
                    eprintln!("Stream error: {}", err);
                },
                None,
            )
        }
        _ => {
            let _ = tx.send(TranscriptionEvent::Error("Unsupported audio sample format".to_string()));
            return;
        }
    };

    match stream_result {
        Ok(stream) => {
            if let Err(e) = stream.play() {
                let _ = tx.send(TranscriptionEvent::Error(format!("Failed to play stream: {}", e)));
                return;
            }

            let mut last_len = 0;
            let mut silence_chunks = 0;
            let mut is_speaking = false;

            // Keep the stream alive while running
            while *running.lock().unwrap() {
                std::thread::sleep(std::time::Duration::from_millis(500));

                if let Some(samples) = streaming_engine.get_buffered_samples(chunk_size) {
                    if samples.len() > last_len {
                        let new_samples = &samples[last_len..];
                        last_len = samples.len();

                        let mut sum_sq = 0.0;
                        for &s in new_samples {
                            sum_sq += s * s;
                        }
                        let rms = if new_samples.is_empty() { 0.0 } else { (sum_sq / new_samples.len() as f32).sqrt() };

                        if rms < 0.005 {
                            silence_chunks += 1;
                        } else {
                            silence_chunks = 0;
                            is_speaking = true;
                        }

                        if !is_speaking {
                            // If we haven't started speaking yet, clear out old silence so it doesn't build up
                            if samples.len() > sample_rate as usize * 3 {
                                streaming_engine.clear_buffer();
                                last_len = 0;
                            }
                            continue;
                        }

                        // If user stopped speaking (1.5 seconds of silence = 3 chunks)
                        if silence_chunks >= 3 {
                            let samples_to_process = streaming_engine.clear_buffer();
                            last_len = 0;
                            silence_chunks = 0;
                            is_speaking = false;

                            if !samples_to_process.is_empty() {
                                let resampled = resample_linear(&samples_to_process, sample_rate as u32, 16000);
                                if let Ok(text) = streaming_engine.transcribe_samples(16000, &resampled) {
                                    let mut clean_text = text;
                                    for tag in &["<|en|>", "<|zh|>", "<|ja|>", "<|ko|>", "<|fr|>", "<|de|>", "<|it|>", "<|es|>", "<|ru|>", "<|asr|>", "<|text|>"] {
                                        clean_text = clean_text.replace(tag, "");
                                    }
                                    let clean_text = clean_text.trim().to_string();
                                    if !clean_text.is_empty() {
                                        let _ = tx.send(TranscriptionEvent::FinalResult(clean_text));
                                    } else {
                                        let _ = tx.send(TranscriptionEvent::PartialResult("".to_string()));
                                    }
                                }
                            }
                            continue;
                        }
                    }
                }
            }

            // Transcribe any remaining samples and clear out the buffer
            let samples = streaming_engine.clear_buffer();
            if !samples.is_empty() && is_speaking {
                let resampled = resample_linear(&samples, sample_rate as u32, 16000);
                if let Ok(text) = streaming_engine.transcribe_samples(16000, &resampled) {
                    let mut clean_text = text;
                    for tag in &["<|en|>", "<|zh|>", "<|ja|>", "<|ko|>", "<|fr|>", "<|de|>", "<|it|>", "<|es|>", "<|ru|>", "<|asr|>", "<|text|>"] {
                        clean_text = clean_text.replace(tag, "");
                    }
                    let clean_text = clean_text.trim().to_string();
                    if !clean_text.is_empty() {
                        let _ = tx.send(TranscriptionEvent::FinalResult(clean_text));
                    }
                }
            }
        }
        Err(e) => {
            let _ = tx.send(TranscriptionEvent::Error(format!("Failed to build stream: {}", e)));
        }
    }
}

fn load_custom_font(cc: &eframe::CreationContext<'_>) {
    let font_data = include_bytes!("../font.ttf");

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "AraletN".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(font_data)),
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
