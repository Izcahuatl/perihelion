## 2024-05-08 - Rust's `str::replace` Always Allocates
**Learning:** The application had an anti-pattern in `clean_transcription` where it skipped `.contains()` before calling `.replace()`, under the false assumption that `replace()` would not allocate on a miss. In Rust, `str::replace()` always allocates a new `String` regardless of whether the pattern is found or not.
**Action:** Always check `.contains()` before calling `.replace()` if the pattern is expected to miss frequently, to avoid unnecessary memory allocations.
