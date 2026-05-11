use std::sync::Arc;
use crate::engine::AsrEngine;

#[derive(Debug, Clone)]
pub enum DownloadEvent {
    Progress { file: Arc<str>, bytes: u64, total: u64 },
    FileDone(String),
    Error(String),
    AllDone,
}

pub enum InitEvent {
    Success(Arc<AsrEngine>),
    Error(String),
}

#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    FinalResult(String),
    Error(String),
}
