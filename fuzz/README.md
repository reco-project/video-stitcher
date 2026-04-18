# reco-fuzz

`cargo-fuzz` harnesses for the adversarial-input surfaces flagged in
the 2026-04-18 deep review.

## Why

External inputs (calibration JSON, ONNX model metadata, user-supplied
paths) can be crafted to drive OOM allocations, integer overflow, or
panics inside reco-core / reco-detect. M1 closed the specific cases
the reviewers found (N-C1 ONNX OOM cap, N-C2 TRT dims overflow, B-10
sync_offset bounds). Fuzzing is the automated complement: instead of
waiting for another human to notice, we keep a continuous source of
new adversarial inputs running against the same parsers.

## Targets

| Target | Function under test | Covers |
|---|---|---|
| `calibration_json` | `serde_json::from_slice::<MatchCalibration>` + `.validate()` | B-10 sync_offset bounds, B-29 finite-float guards, serde depth bombs |
| `onnx_names` | `reco_detect::__fuzz_parse_names_dict_string` | N-C1 OOM cap, parse-state invariants |
| `input_path` | `reco_core::source::validate_input_path` | Path handling before FFmpeg gets to see it |

## Running locally

`cargo-fuzz` requires the nightly toolchain for the sanitizer flags
`libfuzzer-sys` relies on. Install once:

```bash
cargo install cargo-fuzz
rustup toolchain install nightly
```

Then, from the **fuzz/** directory (not the workspace root):

```bash
cd fuzz
cargo +nightly fuzz run calibration_json
cargo +nightly fuzz run onnx_names
cargo +nightly fuzz run input_path
```

Each target runs indefinitely; the fuzzer reports per-iteration
coverage and stops only on a finding or Ctrl-C. Corpus entries that
trigger new paths are written to `fuzz/corpus/<target>/`; findings
are written to `fuzz/artifacts/<target>/`.

## Running in CI

A nightly GH Actions job is planned for M8 follow-up work; not wired
up in this commit because the sanitizer dependencies take ~8 minutes
to compile fresh and we don't want it blocking per-push CI. When
landed, the job will run each target for a bounded time (e.g. 5
minutes each) and fail if new artifacts appear.

## Adding a target

1. Add a new `.rs` file in `fuzz_targets/`.
2. Add a matching `[[bin]]` entry in `Cargo.toml`.
3. If the function is not already `pub`, add a `__fuzz_<name>`
   re-export with `#[doc(hidden)]` in the parent crate.
4. Keep the input size bounded inside the target; the fuzzer is for
   finding logic bugs, not stressing the system allocator.

## Why the subcrate is outside the workspace

`libfuzzer-sys` relies on nightly compiler flags that would leak into
the rest of the workspace if the subcrate were a member. Keeping
`fuzz/` out of the workspace (via `exclude = ["fuzz"]` in the root
`Cargo.toml`) lets the stable-pinned toolchain continue to build every
other crate without interference.
