use eframe::egui;

pub fn load_custom_font(cc: &eframe::CreationContext<'_>) {
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

pub fn setup_custom_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(12.0, 12.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.window_margin = egui::Margin::same(20.0);

    let rounding = egui::Rounding::same(8.0);
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = egui::Color32::from_rgb(26, 26, 26);
    visuals.window_fill = egui::Color32::from_rgb(32, 32, 32);

    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(45, 45, 45);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(60, 60, 60);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(80, 80, 80);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(100, 100, 100);

    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 55, 55));
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 70, 70));
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(110, 110, 110));
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(150, 150, 150));

    visuals.widgets.noninteractive.rounding = rounding;
    visuals.widgets.inactive.rounding = rounding;
    visuals.widgets.hovered.rounding = rounding;
    visuals.widgets.active.rounding = rounding;
    visuals.widgets.open.rounding = rounding;
    visuals.window_rounding = egui::Rounding::same(12.0);

    visuals.selection.bg_fill = egui::Color32::from_rgb(100, 140, 255);

    style.visuals = visuals;

    if let Some(text_style) = style.text_styles.get_mut(&egui::TextStyle::Heading) {
        text_style.family = egui::FontFamily::Name("Title".into());
    }

    ctx.set_style(style);
}

