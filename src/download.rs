use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use crate::events::DownloadEvent;

pub fn download_file_with_progress(
    url: &str,
    dest: &Path,
    tx: &std::sync::mpsc::Sender<DownloadEvent>,
    filename: &Arc<str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut response = minreq::get(url).send_lazy()
        .map_err(|e| format!("HTTP request failed: {}", e))?;
    let total_size = response
        .headers
        .get("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let tmp_dest = dest.with_extension("part");
    let mut file = BufWriter::new(std::fs::File::create(&tmp_dest)?);
    let mut buffer = [0u8; 262_144];
    let mut downloaded: u64 = 0;
    let mut last_reported: u64 = 0;
    
    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;
        if downloaded - last_reported >= 262_144 || downloaded == total_size {
            last_reported = downloaded;
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

