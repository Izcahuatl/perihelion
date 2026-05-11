use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use eframe::egui;
use crate::engine::AsrEngine;
use crate::events::TranscriptionEvent;
use crate::engine::clean_transcription;

pub fn resample_linear(input: &[f32], in_rate: u32, out_rate: u32) -> Cow<'_, [f32]> {
    if in_rate == out_rate {
        return Cow::Borrowed(input);
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
            return Cow::Owned(out);
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
    Cow::Owned(out)
}

fn try_transcribe_and_send(
    engine: &AsrEngine,
    samples: &[f32],
    sample_rate: u32,
    tx: &Sender<TranscriptionEvent>,
    ctx: &egui::Context,
) {
    let resampled = resample_linear(samples, sample_rate, 16000);
    if let Ok(text) = engine.transcribe_samples(16000, &resampled) {
        let clean_text = clean_transcription(&text);
        if !clean_text.is_empty() {
            let _ = tx.send(TranscriptionEvent::FinalResult(clean_text));
            ctx.request_repaint();
        }
    }
}

pub fn run_audio_capture(
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
                            try_transcribe_and_send(&*engine, &samples_to_process, sample_rate as u32, &tx, &ctx);
                        }
                        continue;
                    }
                }
            }

            // Transcribe any remaining samples and clear out the buffer
            let samples = engine.clear_buffer();
            if !samples.is_empty() && is_speaking {
                try_transcribe_and_send(&*engine, &samples, sample_rate as u32, &tx, &ctx);
            }
        }
        Err(e) => {
            let _ = tx.send(TranscriptionEvent::Error(format!("Failed to build stream: {}", e)));
            ctx.request_repaint();
        }
    }
}

