## 2026-05-07 - Rust Iterator Performance on Small vs Large Arrays
**Learning:** While `iter().map(|x| x * x).sum()` is typically very fast and well-optimized by LLVM in Rust, it can sometimes be slower than a manual chunked sum (`chunks_exact(4)`) for certain medium-sized arrays depending on the exact loop structure and compiler heuristics. Benchmarking is essential before replacing manual loop unrolling with iterator combinators.
**Action:** Always benchmark iterator abstractions against explicit loops for math-heavy DSP functions (like RMS calculations) before assuming they will be faster.

## 2026-05-07 - String Allocation in Hot Paths
**Learning:** Creating intermediate `String` instances (e.g., via `.collect()` after string manipulation) in a hot path, such as an OSC sender processing incoming audio strings, can introduce significant overhead (allocations are slow). String slicing via character indices provides a 7x performance boost for UTF-8 strings.
**Action:** Use `Cow::Borrowed` with string slices `&text[..idx]` determined by `char_indices().nth(N)` to truncate strings without allocation.
