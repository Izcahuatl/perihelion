## 2024-05-07 - Avoid byte-level char casts for multi-lingual model output
**Learning:** `bytes[i] as char` corrupted UTF-8 output. The Qwen3 ASR model is multilingual and can output non-ASCII (e.g., Japanese).
**Action:** When iterating over a string, advance using `ch.len_utf8()` and extract proper chars using `text[i..].chars().next().unwrap()` to prevent text corruption.

## 2024-05-07 - Avoid allocating O(N) memory for un-sampled points when downsampling
**Learning:** Down-sampling with moving averages allocated a massive `filtered` array of the full signal size to calculate a moving average for every point, even though most points are skipped when interpolating.
**Action:** Calculate the moving average *lazily* only for the points needed during interpolation. This avoids memory allocation entirely and speeds up calculations significantly.
