use eframe::egui;
use std::borrow::Cow;
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

use crate::config::{AppConfig, ListeningMode, Settings};
use crate::events::{DownloadEvent, InitEvent, TranscriptionEvent};
use crate::engine::{AsrEngine, ModelVariant, MODEL_FILES, get_total_ram_gb};
use crate::audio::run_audio_capture;
use crate::download::download_file_with_progress;
use crate::osc::run_osc_listener;

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
    Manage,
    Settings,
    Debug,
}

impl Default for View {
    fn default() -> Self {
        View::Main
    }
}

pub struct PerihelionApp {
    view: View,
    model_status_small: ModelStatus,
    model_status_large: ModelStatus,
    active_model: Option<ModelVariant>,
    download_rx: Option<Receiver<DownloadEvent>>,
    downloading_variant: Option<ModelVariant>,
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
    osc_running: Option<Arc<AtomicBool>>,
    osc_socket: Option<UdpSocket>,
    available_devices: Vec<String>,
    selected_device: usize,
    total_ram_gb: Option<f64>,
}

impl PerihelionApp {
    fn model_status(&self, variant: ModelVariant) -> &ModelStatus {
        match variant {
            ModelVariant::Small => &self.model_status_small,
            ModelVariant::Large => &self.model_status_large,
        }
    }

    fn set_model_status(&mut self, variant: ModelVariant, status: ModelStatus) {
        match variant {
            ModelVariant::Small => self.model_status_small = status,
            ModelVariant::Large => self.model_status_large = status,
        }
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
                if let Ok(config) = nanoserde::DeJson::deserialize_json(&contents) {
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
        let json = nanoserde::SerJson::serialize_json(&config);
        let _ = std::fs::write(&path, json);
    }

    fn check_model_exists(model_dir: &Path) -> bool {
        MODEL_FILES.iter().all(|f| model_dir.join(f).exists())
    }

    fn get_available_devices() -> Vec<String> {
        use cpal::traits::{DeviceTrait, HostTrait};
        let host = cpal::default_host();
        let mut devices = Vec::new();

        if let Some(device) = host.default_input_device() {
            if let Ok(desc) = device.description() {
                devices.push(format!("{} (default)", desc.name()));
            }
        }

        if let Ok(device_iter) = host.input_devices() {
            for device in device_iter {
                if let Ok(desc) = device.description() {
                    let name = desc.name();
                    if !devices.iter().any(|d| d.contains(name)) {
                        devices.push(name.to_string());
                    }
                }
            }
        }

        if devices.is_empty() {
            devices.push("No devices found".to_string());
        }

        devices
    }

    fn start_download(&mut self, variant: ModelVariant) {
        self.set_model_status(variant, ModelStatus::Downloading {
            current_file: Arc::from(MODEL_FILES[0]),
            current_bytes: 0,
            total_bytes: 0,
        });
        self.downloading_variant = Some(variant);
        let model_dir = variant.model_dir();
        let repo = variant.repo();
        let (tx, rx) = channel();
        self.download_rx = Some(rx);
        let files = MODEL_FILES;

        thread::spawn(move || {
            std::fs::create_dir_all(&model_dir).ok();
            for &file in files {
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    repo, file
                );
                let dest = model_dir.join(file);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                if dest.exists() {
                    let _ = tx.send(DownloadEvent::FileDone(file.to_string()));
                    continue;
                }
                let file_arc: Arc<str> = Arc::from(file);
                if let Err(e) = download_file_with_progress(&url, &dest, &tx, &file_arc) {
                    let _ = tx.send(DownloadEvent::Error(format!("{}: {}", file, e)));
                    return;
                }
                let _ = tx.send(DownloadEvent::FileDone(file.to_string()));
            }
            let _ = tx.send(DownloadEvent::AllDone);
        });
    }

    fn start_init_engine(&mut self, variant: ModelVariant, ctx: Option<&egui::Context>) {
        if let Some(ctx) = ctx {
            if self.is_listening {
                self.set_listening(false, ctx);
            }
        }
        self.engine = None;
        if let Some(old) = self.active_model {
            if old != variant {
                self.set_model_status(old, if Self::check_model_exists(&old.model_dir()) {
                    ModelStatus::Ready
                } else {
                    ModelStatus::NotDownloaded
                });
            }
        }

        self.set_model_status(variant, ModelStatus::Initializing);
        self.active_model = Some(variant);
        self.status_message = Cow::Borrowed("Initializing...");
        let (tx, rx) = channel();
        self.init_rx = Some(rx);
        let model_dir = variant.model_dir();
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
            let variant = self.active_model.unwrap_or(ModelVariant::Small);
            match event {
                InitEvent::Success(engine) => {
                    self.engine = Some(engine);
                    self.set_model_status(variant, ModelStatus::Ready);
                    self.status_message = Cow::Borrowed("Ready");
                }
                InitEvent::Error(e) => {
                    self.set_model_status(variant, ModelStatus::Error(e));
                    self.status_message = Cow::Borrowed("Initialization failed");
                }
            }
            self.init_rx = None;
        }
    }

    fn handle_download_events(&mut self) {
        let Some(rx) = self.download_rx.take() else { return };
        let variant = self.downloading_variant.unwrap_or(ModelVariant::Small);
        while let Ok(event) = rx.try_recv() {
            match event {
                DownloadEvent::Progress {
                    file,
                    bytes,
                    total,
                } => {
                    self.set_model_status(variant, ModelStatus::Downloading {
                        current_file: file,
                        current_bytes: bytes,
                        total_bytes: total,
                    });
                }
                DownloadEvent::FileDone(file) => {
                    self.status_message = Cow::Owned(format!("Downloaded: {}", file));
                }
                DownloadEvent::Error(e) => {
                    self.set_model_status(variant, ModelStatus::Error(e));
                }
                DownloadEvent::AllDone => {
                    self.set_model_status(variant, ModelStatus::Ready);
                    self.downloading_variant = None;
                    self.status_message = Cow::Borrowed("Download complete");
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
            let running = Arc::new(AtomicBool::new(true));
            self.osc_running = Some(running.clone());
            thread::spawn(move || run_osc_listener(tx, ctx_clone, running));
        }
        if self.settings.listening_mode != ListeningMode::ToggleOsc {
            self.osc_rx = None;
            if let Some(running) = self.osc_running.take() {
                running.store(false, Ordering::Relaxed);
            }
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
        if self.settings.listening_mode == ListeningMode::AlwaysOn
            && !self.is_listening
            && self.engine.is_some()
        {
            self.set_listening(true, ctx);
            self.status_message = Cow::Borrowed("Always On");
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
        let msg = match text.char_indices().nth(140) {
            Some((idx, _)) => text[..idx].to_string(),
            None => text.to_string(),
        };
        self.send_osc_message("/chatbox/input", vec![
            rosc::OscType::String(msg),
            rosc::OscType::Bool(true),
        ]);
    }

    fn send_osc_typing(&self, typing: bool) {
        self.send_osc_message("/chatbox/typing", vec![rosc::OscType::Bool(typing)]);
    }
}

impl Default for PerihelionApp {
    fn default() -> Self {
        let small_exists = Self::check_model_exists(&ModelVariant::Small.model_dir());
        let large_exists = Self::check_model_exists(&ModelVariant::Large.model_dir());
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

        Self {
            view: View::default(),
            model_status_small: if small_exists { ModelStatus::Ready } else { ModelStatus::NotDownloaded },
            model_status_large: if large_exists { ModelStatus::Ready } else { ModelStatus::NotDownloaded },
            active_model: None,
            download_rx: None,
            downloading_variant: None,
            init_rx: None,
            osc_rx: None,
            transcription_rx: None,
            recognized_text: String::new(),
            is_listening: false,
            status_message: Cow::Borrowed("No model loaded"),
            settings,
            test_file_path: String::new(),
            engine: None,
            audio_running: Arc::new(AtomicBool::new(false)),
            osc_running: None,
            osc_socket,
            available_devices,
            selected_device,
            total_ram_gb: get_total_ram_gb(),
        }
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

        let any_busy = matches!(self.model_status_small, ModelStatus::Downloading { .. } | ModelStatus::Initializing)
            || matches!(self.model_status_large, ModelStatus::Downloading { .. } | ModelStatus::Initializing);
        if any_busy || self.is_listening {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        let active_initializing = self.active_model.map_or(false, |v| *self.model_status(v) == ModelStatus::Initializing);
        if active_initializing {
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

        egui::TopBottomPanel::top("navbar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(18, 18, 18))
                    .inner_margin(egui::Margin::symmetric(16.0, 10.0)),
            )
            .show(ctx, |ui| {
                let title_bar_rect = ui.max_rect();
                let title_bar_response = ui.interact(title_bar_rect, ui.id().with("title_bar"), egui::Sense::click_and_drag());
                if title_bar_response.is_pointer_button_down_on() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }

                ui.horizontal_centered(|ui| {
                    ui.label(egui::RichText::new("Perihelion").font(egui::FontId::proportional(18.0)).strong().color(egui::Color32::from_rgb(220, 220, 220)));
                    
                    ui.add_space(24.0);

                    let mut sel = |ui: &mut egui::Ui, label: &str, view: View| {
                        let active = self.view == view;
                        let bg_color = if active { egui::Color32::from_rgb(80, 80, 80) } else { egui::Color32::TRANSPARENT };
                        let text_color = if active { egui::Color32::WHITE } else { egui::Color32::GRAY };
                        
                        let btn = egui::Button::new(egui::RichText::new(label).size(14.0).color(text_color))
                            .fill(bg_color)
                            .rounding(4.0)
                            .frame(true);
                            
                        if ui.add(btn).clicked() {
                            self.view = view;
                        }
                    };

                    sel(ui, "Main", View::Main);
                    ui.add_space(8.0);
                    sel(ui, "Manage", View::Manage);
                    ui.add_space(8.0);
                    sel(ui, "Settings", View::Settings);
                    ui.add_space(8.0);
                    sel(ui, "Debug", View::Debug);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let close_btn = ui.add(egui::Button::new(egui::RichText::new("X").size(16.0)).frame(false));
                        if close_btn.is_pointer_button_down_on() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                });
            });

        let frame = egui::Frame::central_panel(&ctx.style())
            .inner_margin(egui::Margin::same(20.0));

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            let avail = ui.available_width();

            match self.view {
                View::Main => {
                    ui.horizontal(|ui| {
                        ui.label("Mode:");
                        ui.label(self.settings.listening_mode.as_str());
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if let Some(variant) = self.active_model {
                                    let status = self.model_status(variant);
                                    match status {
                                        ModelStatus::Ready => {
                                            ui.colored_label(egui::Color32::GREEN, "ready");
                                        }
                                        ModelStatus::Initializing => {
                                            ui.add_sized([12.0, 12.0], egui::Spinner::new());
                                            ui.label("initializing");
                                        }
                                        ModelStatus::Error(_) => {
                                            ui.colored_label(egui::Color32::RED, "error");
                                        }
                                        _ => {}
                                    }
                                    ui.label(variant.label());
                                    ui.label("·");
                                } else {
                                    ui.colored_label(egui::Color32::GRAY, "none");
                                    ui.label("Model:");
                                }
                            },
                        );
                    });

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
                            self.set_listening(false, ctx);
                            self.status_message = Cow::Borrowed("Ready");
                        } else {
                            self.set_listening(true, ctx);
                            self.status_message = Cow::Borrowed("Listening...");
                        }
                    }

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

                    ui.horizontal(|ui| {
                        ui.label(self.status_message.as_ref());
                        if self.is_listening {
                            ui.spinner();
                        }
                    });
                }

                View::Manage => {
                    ui.heading(egui::RichText::new("Manage Models").size(22.0).strong());
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for variant in &[ModelVariant::Small, ModelVariant::Large] {
                            let variant = *variant;
                            let is_active = self.active_model == Some(variant) && self.engine.is_some();
                            let status = self.model_status(variant).clone();

                            egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.strong(egui::RichText::new(variant.label()).size(16.0));
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if is_active {
                                            ui.colored_label(egui::Color32::GREEN, "● Running");
                                        } else {
                                            match &status {
                                                ModelStatus::Ready => {
                                                    ui.colored_label(egui::Color32::from_rgb(180, 180, 180), "Downloaded");
                                                }
                                                ModelStatus::NotDownloaded => {
                                                    ui.colored_label(egui::Color32::YELLOW, "Not downloaded");
                                                }
                                                ModelStatus::Downloading { .. } => {
                                                    ui.add_sized([12.0, 12.0], egui::Spinner::new());
                                                    ui.label("Downloading...");
                                                }
                                                ModelStatus::Initializing => {
                                                    ui.add_sized([12.0, 12.0], egui::Spinner::new());
                                                    ui.label("Starting...");
                                                }
                                                ModelStatus::Error(_) => {
                                                    ui.colored_label(egui::Color32::RED, "Error");
                                                }
                                            }
                                        }
                                    });
                                });

                                if variant == ModelVariant::Large {
                                    if let Some(ram) = self.total_ram_gb {
                                        if ram < 65.8 {
                                            ui.add_space(4.0);
                                            ui.horizontal(|ui| {
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(255, 180, 50),
                                                    format!("Your system has {:.1} GB RAM. The model may crash or run very slowly.", ram),
                                                );
                                            });
                                        }
                                    }
                                }

                                if let ModelStatus::Downloading { current_file, current_bytes, total_bytes } = &status {
                                    ui.add_space(4.0);
                                    if *total_bytes > 0 {
                                        let progress = *current_bytes as f32 / *total_bytes as f32;
                                        ui.add_sized(
                                            [ui.available_width(), 16.0],
                                            egui::ProgressBar::new(progress).text(format!(
                                                "{}  {} / {} MB",
                                                current_file,
                                                current_bytes / 1_000_000,
                                                total_bytes / 1_000_000
                                            )),
                                        );
                                    } else {
                                        ui.add_sized(
                                            [ui.available_width(), 16.0],
                                            egui::ProgressBar::new(0.0).text("Starting..."),
                                        );
                                    }
                                }

                                if let ModelStatus::Error(e) = &status {
                                    ui.add_space(4.0);
                                    ui.colored_label(egui::Color32::RED, e.as_str());
                                }

                                ui.add_space(8.0);
                                ui.horizontal(|ui| {
                                    match &status {
                                        ModelStatus::NotDownloaded | ModelStatus::Error(_) => {
                                            let already_downloading = self.download_rx.is_some();
                                            let btn = ui.add_enabled(
                                                !already_downloading,
                                                egui::Button::new("Download"),
                                            );
                                            if btn.clicked() {
                                                self.start_download(variant);
                                            }
                                        }
                                        ModelStatus::Ready => {
                                            if is_active {
                                                if ui.button("Stop").clicked() {
                                                    if self.is_listening {
                                                        self.set_listening(false, ctx);
                                                    }
                                                    self.engine = None;
                                                    self.active_model = None;
                                                    self.status_message = Cow::Borrowed("Model stopped");
                                                }
                                            } else {
                                                let initializing_any = self.init_rx.is_some();
                                                let btn = ui.add_enabled(
                                                    !initializing_any,
                                                    egui::Button::new("Start"),
                                                );
                                                if btn.clicked() {
                                                    self.start_init_engine(variant, Some(ctx));
                                                }
                                            }

                                            ui.add_space(8.0);
                                            if ui.add(egui::Button::new(
                                                egui::RichText::new("Delete").color(egui::Color32::from_rgb(200, 80, 80)),
                                            )).clicked() {
                                                if is_active {
                                                    if self.is_listening {
                                                        self.set_listening(false, ctx);
                                                    }
                                                    self.engine = None;
                                                    self.active_model = None;
                                                }
                                                let _ = std::fs::remove_dir_all(variant.model_dir());
                                                self.set_model_status(variant, ModelStatus::NotDownloaded);
                                                self.status_message = Cow::Borrowed("Model deleted");
                                            }
                                        }
                                        _ => {}
                                    }
                                });
                            });
                            ui.add_space(8.0);
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
                                egui::ComboBox::from_id_salt("device_cb")
                                    .selected_text(
                                        self.available_devices.get(self.selected_device)
                                            .map(|s| s.as_str())
                                            .unwrap_or("Unknown")
                                    )
                                    .width(260.0)
                                    .show_ui(ui, |ui| {
                                        for (i, name) in self.available_devices.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut self.selected_device,
                                                i,
                                                name.as_str(),
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
                            if let Some(variant) = self.active_model {
                                self.start_init_engine(variant, Some(ctx));
                            }
                        }

                        ui.add_space(20.0);
                        ui.heading(egui::RichText::new("Danger Zone").size(18.0).strong().color(egui::Color32::RED));
                        ui.add_space(8.0);
                        egui::Frame::group(&ctx.style()).inner_margin(egui::Margin::same(12.0)).show(ui, |ui| {
                            ui.label("Reset all settings to their factory defaults.");
                            ui.add_space(4.0);
                            if ui.button(egui::RichText::new("Reset Config").size(16.0).color(egui::Color32::RED)).clicked() {
                                self.settings = Settings::default();
                                self.selected_device = 0;
                                self.save_config();
                                self.status_message = Cow::Borrowed("Config reset");
                            }
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("Use the Manage tab to delete model files.").color(egui::Color32::GRAY));
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
            }
        });
    }
}
