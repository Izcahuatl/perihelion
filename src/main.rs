#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod config;
mod events;
mod engine;
mod download;
mod osc;
mod audio;
mod theme;
mod app;

use eframe::egui;
use app::PerihelionApp;
use theme::{load_custom_font, setup_custom_style};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([600.0, 500.0])
            .with_min_inner_size([600.0, 500.0])
            .with_max_inner_size([600.0, 500.0])
            .with_resizable(false)
            .with_decorations(false)
            .with_transparent(true),
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
