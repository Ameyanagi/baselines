# Performance Records

This directory stores compact benchmark records that are small enough to keep in
git. Criterion's raw `target/criterion` directory is intentionally not checked
in.

## 2026-05-24 Baseline Workloads

Environment:

- Commit before optimization: `4b54c7a`
- OS: macOS 26.4 25E246
- CPU: Apple M4, 10 logical CPUs
- Memory: 32 GiB
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`, `aarch64-apple-darwin`
- Criterion config: `sample_size = 10`, warm-up `500 ms`, measurement `1 s`

Commands:

```sh
cargo bench --bench baseline_workloads -- --save-baseline perf_before_opt
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-beads.sample.txt
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --save-baseline beads_after_banded_threshold
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --baseline perf_before_opt
```

Full saved baseline means are in
[`baseline-workloads-2026-05-24.csv`](baseline-workloads-2026-05-24.csv).

Top slow paths before optimization:

| Benchmark | Mean |
| --- | ---: |
| `optimizers_misc_1d/beads_256` | 23.094 ms |
| `spline_1d/pspline_iasls_256` | 15.934 ms |
| `spline_1d/pspline_aspls_256` | 8.954 ms |
| `whittaker_2d/brpls_16x16` | 8.232 ms |
| `polynomial_1d/goldindec_256` | 6.662 ms |
| `whittaker_2d/arpls_16x16` | 6.648 ms |

BEADS profiling before optimization:

- Target: `optimizers_misc_1d/beads_256`
- `sample` captured 3835 samples from the Criterion profile run.
- Nearly all samples were inside `baselines::misc::beads::beads_filter_type_one`.
- The visible secondary costs were `_platform_memmove`, allocation/free, and the
  dense compatibility solve path. The tridiagonal solve appeared only in a small
  number of samples.

BEADS optimization result:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `optimizers_misc_1d/beads_256` | 23.094 ms | 0.609 ms | -97.36% |

The retained change narrows BEADS dense compatibility to fixture-sized small
inputs and sends the 256-point benchmark through the banded solver. Fixture
compatibility remained passing for the pinned pybaselines references.

P-spline solver optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `spline_1d/pspline_iasls_256` | 15.934 ms | 0.159 ms | -99.00% |
| `spline_1d/pspline_aspls_256` | 8.954 ms | 0.990 ms | -88.74% |
| `spline_1d/pspline_drpls_256` | 3.713 ms | 0.348 ms | -90.60% |

The retained changes update the data-domain first-difference penalty assembly
to use sparse differences between adjacent B-spline basis rows and use a
general banded solver for narrow non-symmetric P-spline systems, while keeping
the original dense solve path for smaller basis counts. The public API remains
unchanged.
