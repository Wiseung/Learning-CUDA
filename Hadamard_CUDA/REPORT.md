# Hadamard CUDA Report

## 1. Implementation Summary

The project treats the input as `[total_rows, head_dim]`, where:

- `total_rows = batch_size * seq_len * num_heads`
- each row is transformed independently
- the transform is normalized by `1 / sqrt(head_dim)`

Implemented GPU paths:

- `v1_butterfly`
  - shared-memory butterfly base path
  - FP16 uses `half2`
  - BF16 uses `bfloat162`
  - intra-warp Hadamard stages are now handled with register-local math plus `__shfl_xor_sync`
- `v2_wmma`
  - layered Hadamard decomposition
  - inner `H16` blocks use WMMA
  - outer group dimension uses butterfly reductions
  - BF16 WMMA is enabled only on `SM80+`
- `best_auto`
  - currently dispatches to `v1_butterfly`
  - kept as the stable default while `v2_wmma` remains available for comparison and future tuning

Quantization paths:

- INT4 per-row absmax
- FP8 E4M3 per-row absmax
- fused Hadamard + INT4
- fused Hadamard + FP8

Standalone and fused quantization use warp-level max reduction instead of a full shared-memory tree reduction.

## 2. Optimization History

Main iterations:

1. CPU reference, baseline shared-memory Hadamard, standalone quantization, fused quantization, tests, and benchmark harness.
2. Initial WMMA path using a more direct matrix-style formulation. Good only for `head_dim=64`.
3. Layered WMMA (`H_group x H16`) to constrain Tensor Core work to the natural 16x16 tile size.
4. FP16 `half2` vectorization for v1, standalone quantization, and fused quantization.
5. BF16 `bfloat162` vectorization and BF16 WMMA on `SM80+`.
6. Warp-level `absmax` reduction for standalone and fused quantization.
7. Warp-level Hadamard stages for vectorized v1 kernels.
8. Benchmark harness stabilization:
   - device-buffer reset between trials
   - median across multiple trials
   - interleaved Hadamard-path measurement to reduce order bias

One attempted optimization was rolled back:

- shared-memory padding to reduce bank conflicts
- NCU correctly showed bank-conflict pressure
- but the extra address arithmetic and staging overhead caused benchmark regressions
- that version was removed

## 3. Current Dispatch

At this point the safest default is:

- `best_auto -> v1_butterfly`

Reason:

- after the warp-shuffle butterfly change and the benchmark harness stabilization, the vectorized v1 path is consistently stronger than the current WMMA path across the measured matrix
- `v2_wmma` remains useful as an experimental path and as a profiling baseline, but it is not the default anymore

## 4. Benchmark Harness

The benchmark harness now tries to minimize measurement bias:

- each kernel family is measured over multiple trials
- the reported time is the median trial result
- work buffers are reset from a device-resident source buffer before each trial
- Hadamard kernels are measured in interleaved order so one path does not always benefit from being first or last

This is still not a full microbenchmarking framework, but it is materially more reliable than the earlier single-pass timing.

## 5. Benchmark Snapshot

Latest benchmark output is stored in `bench/results.csv`.

Representative Hadamard results with the stabilized harness:

| kernel | dtype | head_dim | total_rows | time_ms | throughput_GB_s |
| --- | --- | --- | --- | ---: | ---: |
| v1_butterfly | fp16 | 64 | 65536 | 0.093430 | 179.569 |
| v2_wmma | fp16 | 64 | 65536 | 0.115393 | 145.392 |
| best_auto | fp16 | 64 | 65536 | 0.093147 | 180.116 |
| v1_butterfly | fp16 | 128 | 65536 | 0.116732 | 287.448 |
| v2_wmma | fp16 | 128 | 65536 | 0.247268 | 135.701 |
| best_auto | fp16 | 128 | 65536 | 0.116559 | 287.874 |
| v1_butterfly | fp16 | 256 | 65536 | 0.295891 | 226.803 |
| v2_wmma | fp16 | 256 | 65536 | 0.622284 | 107.843 |
| best_auto | fp16 | 256 | 65536 | 0.293964 | 228.289 |
| v1_butterfly | bf16 | 64 | 65536 | 0.093322 | 179.778 |
| v2_wmma | bf16 | 64 | 65536 | 0.118344 | 141.766 |
| best_auto | bf16 | 64 | 65536 | 0.093539 | 179.361 |
| v1_butterfly | bf16 | 128 | 65536 | 0.116020 | 289.211 |
| v2_wmma | bf16 | 128 | 65536 | 0.251268 | 133.540 |
| best_auto | bf16 | 128 | 65536 | 0.114420 | 293.255 |
| v1_butterfly | bf16 | 256 | 65536 | 0.294817 | 227.629 |
| v2_wmma | bf16 | 256 | 65536 | 0.633479 | 105.937 |
| best_auto | bf16 | 256 | 65536 | 0.293751 | 228.455 |

Observations:

- after the warp-shuffle v1 update, the current WMMA implementation is no longer the leading path in the stabilized harness
- the gap is largest for `head_dim=128` and `head_dim=256`
- FP16 and BF16 show the same broad trend

Representative fused-quantization results:

| pair | dtype | head_dim | total_rows | step_ms | fused_ms | speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Hadamard + INT4 | fp16 | 64 | 65536 | 0.205353 | 0.198410 | 1.04x |
| Hadamard + FP8 | fp16 | 64 | 65536 | 0.221674 | 0.179343 | 1.24x |
| Hadamard + INT4 | fp16 | 128 | 65536 | 0.185135 | 0.252366 | 0.73x |
| Hadamard + FP8 | fp16 | 128 | 65536 | 0.213535 | 0.289597 | 0.74x |
| Hadamard + INT4 | fp16 | 256 | 65536 | 0.481185 | 0.549868 | 0.87x |
| Hadamard + FP8 | fp16 | 256 | 65536 | 0.538931 | 0.625060 | 0.86x |
| Hadamard + INT4 | bf16 | 64 | 65536 | 0.206234 | 0.199393 | 1.03x |
| Hadamard + FP8 | bf16 | 64 | 65536 | 0.221388 | 0.179354 | 1.23x |
| Hadamard + INT4 | bf16 | 128 | 65536 | 0.185170 | 0.253041 | 0.73x |
| Hadamard + FP8 | bf16 | 128 | 65536 | 0.214211 | 0.290668 | 0.74x |
| Hadamard + INT4 | bf16 | 256 | 65536 | 0.481260 | 0.551413 | 0.87x |
| Hadamard + FP8 | bf16 | 256 | 65536 | 0.539747 | 0.627842 | 0.86x |

Observations:

- fused quantization is still competitive for `head_dim=64`
- for `128/256`, the now-faster v1 path has reduced or erased the fused advantage
- this means the fused kernels should be revisited if they are expected to remain part of the final optimized path

## 6. Profiling Summary

NCU profiles were collected for:

- `fp16 + head_dim=64 + total_rows=65536`
- `fp16 + head_dim=128 + total_rows=65536`
- `bf16 + head_dim=64 + total_rows=65536`
- `bf16 + head_dim=128 + total_rows=65536`

Artifacts:

- `bench/ncu_fp16_hd64.csv`
- `bench/ncu_fp16_hd128.csv`
- `bench/ncu_bf16_hd64.csv`
- `bench/ncu_bf16_hd128.csv`

Important caveat:

- those NCU captures explain the earlier dispatch behavior
- they were collected before the final warp-shuffle v1 update and before the latest benchmark-harness stabilization
- they should not be treated as final proof that WMMA is still the best default path

What they still tell us:

- shared-memory conflict pressure is real in both v1 and v2
- the earlier 64-dim WMMA path achieved stronger scheduler efficiency and occupancy than the older v1
- the current v1 path should be re-profiled before any new dispatch rule is introduced

## 7. Correctness

`hadamard_test.exe` passes in full.

Maximum absolute error summary:

| path | dtype | max abs error |
| --- | --- | ---: |
| v1_butterfly | fp16 | 1.307e-03 |
| v1_butterfly | bf16 | 1.013e-02 |
| v2_wmma | fp16 | 1.307e-03 |
| v2_wmma | bf16 | 1.013e-02 |

Fused quantization checks:

- INT4 fused output is bit-exact against split execution
- FP8 fused output is bit-exact against split execution
- scale error is `0.000e+00`

## 8. Current Risks

- the benchmark harness is better, but small-kernel timing can still move noticeably across runs
- dispatch changes should still be conservative
- the fused kernels are now lagging behind the much faster v1 path on larger dimensions

## 9. Next Directions

- re-profile the new warp-shuffle v1 implementation with NCU before making any new WMMA decisions
- revisit fused quantization so it is competitive with the new v1 baseline, especially for `head_dim=128/256`
- push warp-level optimization further into the Hadamard butterfly, including cross-warp stage design rather than only intra-warp stages
- add CPU reference timings to `benchmark.cu`
- add multi-stream experiments for large `total_rows`
