use nanoserde::{DeJson, SerJson};

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeJson, SerJson)]
pub enum Language {
    English,
    Japanese,
}

impl Default for Language {
    fn default() -> Self {
        Language::English
    }
}

impl Language {
    pub fn tr<'a>(&self, key: &'a str) -> &'a str {
        if *self == Language::English {
            return key;
        }
        match key {
            "Main" => "メイン",
            "Manage" => "管理",
            "Settings" => "設定",
            "Debug" => "デバッグ",
            "Mode:" => "モード:",
            "Model:" => "モデル:",
            "ready" => "準備完了",
            "initializing" => "初期化中",
            "error" => "エラー",
            "none" => "なし",
            "On" => "オン",
            "Off" => "オフ",
            "Output" => "出力",
            "Copy" => "コピー",
            "Clear" => "クリア",
            "Copied" => "コピーしました",
            "Cleared" => "クリアしました",
            "Listening..." => "リスニング中...",
            "Ready" => "準備完了",
            "No model loaded" => "モデルが読み込まれていません",
            "Engine not ready" => "エンジンが準備されていません",
            "Always On" => "常にオン",
            "Manage Models" => "モデルの管理",
            "Download" => "ダウンロード",
            "Start" => "開始",
            "Stop" => "停止",
            "Delete" => "削除",
            "Downloading..." => "ダウンロード中...",
            "Starting..." => "起動中...",
            "Not downloaded" => "未ダウンロード",
            "Downloaded" => "ダウンロード済み",
            "Error" => "エラー",
            "Model deleted" => "モデルを削除しました",
            "Model stopped" => "モデルを停止しました",
            "Download complete" => "ダウンロード完了",
            "Initialization failed" => "初期化に失敗しました",
            "OSC: On" => "OSC: オン",
            "OSC: Off" => "OSC: オフ",
            "Hang on, the model's starting!" => "少々お待ちください。モデルを起動中です！",
            "● Running" => "● 実行中",
            "Language:" => "言語:",
            "English" => "英語",
            "Japanese" => "日本語",
            "Auto-copy to clipboard" => "クリップボードに自動コピー",
            "Append transcriptions" => "文字起こしを追記する",
            "Activation Mode" => "起動モード",
            "Toggle-Button (Manual)" => "トグルボタン（手動）",
            "OSC (VRChat)" => "OSC (VRChat)",
            "Microphone" => "マイク",
            "Unknown" => "不明",
            "Refresh" => "更新",
            "Devices refreshed" => "デバイスを更新しました",
            "AI Preferences" => "AI設定",
            "High Accuracy Mode (Modified Beam Search)" => "高精度モード (Modified Beam Search)",
            "Search Depth:" => "検索の深さ:",
            "Higher = More accurate but requires more CPU power" => "高くすると精度が上がりますが、CPU負荷が高くなります",
            "CPU Threads:" => "CPUスレッド数:",
            "Custom Dictionary" => "カスタム辞書",
            "Add terms to help the AI recognize them" => "AIが認識しやすいように単語を追加します",
            "Hotword Focus:" => "ホットワードの強調:",
            "Boost multiplier" => "強調倍率",
            "Apply Settings & Reload" => "設定を適用して再読み込み",
            "Danger Zone" => "危険な操作",
            "Reset all settings to their factory defaults." => "すべての設定を初期状態にリセットします。",
            "Reset Config" => "設定をリセット",
            "Config reset" => "設定をリセットしました",
            "Use the Manage tab to delete model files." => "モデルファイルの削除は「管理」タブから行ってください。",
            "Path to .wav file" => ".wavファイルのパス",
            "Run" => "実行",
            "Model not ready" => "モデルの準備ができていません",
            "Your system has " => "システムのRAMは",
            " GB RAM. The model may crash or run very slowly." => "GBです。モデルがクラッシュするか、非常に遅くなる可能性があります。",
            _ => key,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, DeJson, SerJson)]
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

#[derive(Debug, Clone, DeJson, SerJson)]
pub struct Settings {
    pub language: Language,
    pub listening_mode: ListeningMode,
    pub auto_copy: bool,
    pub append_mode: bool,
    pub num_threads: i32,
    pub high_accuracy: bool,
    pub search_depth: i32,
    pub hotwords: String,
    pub hotwords_boost: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: Language::default(),
            listening_mode: ListeningMode::default(),
            auto_copy: true,
            append_mode: true,
            num_threads: 4,
            high_accuracy: false,
            search_depth: 4,
            hotwords: String::new(),
            hotwords_boost: 1.5,
        }
    }
}

#[derive(Debug, Clone, DeJson, SerJson)]
pub struct AppConfig {
    pub settings: Settings,
    pub selected_device: usize,
}

