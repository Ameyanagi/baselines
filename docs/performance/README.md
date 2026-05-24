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
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --save-baseline beads_current_profile
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-beads-current.sample.txt
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --save-baseline beads_after_direct_band_assembly
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --baseline beads_current_profile
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-beads-direct-band.sample.txt
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --save-baseline beads_before_workspace_reuse
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-beads-workspace.sample.txt
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --baseline beads_before_workspace_reuse
cargo bench --bench baseline_workloads -- optimizers_misc_1d/beads_256 --save-baseline beads_after_workspace_reuse
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-goldindec.sample.txt
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --save-baseline goldindec_after_polynomial_workspace
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --baseline perf_before_opt
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-goldindec-current.sample.txt
git worktree add /tmp/baselines-before-unweighted-cache 496c8ac
cargo bench --manifest-path /tmp/baselines-before-unweighted-cache/Cargo.toml --bench baseline_workloads -- polynomial_1d --save-baseline before_unweighted_cache
cargo bench --bench baseline_workloads -- polynomial_1d --save-baseline polynomial1d_after_unweighted_cache
cargo bench --bench baseline_workloads -- polynomial_2d/quant_reg_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-quantreg2d.sample.txt
cargo bench --bench baseline_workloads -- polynomial_2d --baseline perf_before_opt
cargo bench --bench baseline_workloads -- polynomial_2d --save-baseline polynomial2d_after_basis_workspace
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --save-baseline goldindec_before_direct_rhs
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --profile-time 30
sample <baseline_workloads-pid> 5 -file /tmp/baselines-goldindec-current-2.sample.txt
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --baseline goldindec_before_direct_rhs
cargo bench --bench baseline_workloads -- polynomial_1d --baseline polynomial1d_after_unweighted_cache
cargo bench --bench baseline_workloads -- polynomial_1d --save-baseline polynomial1d_after_cost_dispatch
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --save-baseline goldindec_before_threshold_search
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-goldindec.sample.txt
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --baseline goldindec_before_threshold_search
cargo bench --bench baseline_workloads -- polynomial_1d_into/goldindec_into_256 --save-baseline goldindec_after_quadratic_specialization
cargo bench --bench baseline_workloads -- polynomial_1d/goldindec_256 --save-baseline goldindec_after_quadratic_specialization
cargo bench --bench baseline_workloads -- morphology_1d/amormol_256 --profile-time 30
sample <baseline_workloads-pid> 5 -file /tmp/baselines-amormol1d.sample.txt
cargo bench --bench baseline_workloads -- morphology_1d --save-baseline morphology1d_before_reflect_split
cargo bench --bench baseline_workloads -- morphology_1d --baseline morphology1d_before_reflect_split
cargo bench --bench baseline_workloads -- morphology_1d --save-baseline morphology1d_after_reflect_split
cargo bench --bench baseline_workloads -- morphology_2d/imor_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-imor2d.sample.txt
cargo bench --bench baseline_workloads -- morphology_2d --baseline perf_before_opt
cargo bench --bench baseline_workloads -- morphology_2d --save-baseline morphology2d_after_separable
cargo bench --bench baseline_workloads -- optimizers_2d/collab_pls_2x16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-collab2d.sample.txt
cargo bench --bench baseline_workloads -- spline_2d/pspline_iarpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-pspline-iarpls2d.sample.txt
cargo bench --bench baseline_workloads -- spline_2d --baseline perf_before_opt
cargo bench --bench baseline_workloads -- spline_2d --save-baseline pspline2d_after_workspace
cargo bench --bench baseline_workloads -- spline_2d/pspline_airpls_16x16 --save-baseline pspline_airpls2d_before_profile
cargo bench --bench baseline_workloads -- spline_2d/pspline_airpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-pspline-airpls2d.sample.txt
cargo bench --bench baseline_workloads -- spline_2d/pspline_airpls_16x16 --baseline pspline_airpls2d_before_profile
cargo bench --bench baseline_workloads -- spline_2d/pspline_iasls_16x16 --save-baseline pspline_iasls2d_before_profile
cargo bench --bench baseline_workloads -- spline_2d/pspline_iasls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-pspline-iasls2d.sample.txt
cargo bench --bench baseline_workloads -- spline_2d/pspline_iasls_16x16 --baseline pspline_iasls2d_before_profile
cargo bench --bench baseline_workloads -- spline_2d/pspline_iasls_16x16 --save-baseline pspline_iasls2d_after_workspace
cargo bench --bench baseline_workloads -- spline_2d/pspline_iasls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-pspline-iasls2d-after.sample.txt
cargo bench --bench baseline_workloads -- spline_2d_into/pspline_iasls_into_16x16 --save-baseline pspline_iasls2d_into_after_workspace
cargo bench --bench baseline_workloads -- spline_2d/pspline_iarpls_16x16 --save-baseline pspline_iarpls2d_current_profile
cargo bench --bench baseline_workloads -- spline_2d/pspline_iarpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-pspline-iarpls2d-current.sample.txt
cargo bench --bench baseline_workloads -- spline_2d/pspline_iarpls_16x16 --baseline pspline_iarpls2d_current_profile
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-arpls2d.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d --baseline perf_before_opt
cargo bench --bench baseline_workloads -- whittaker_2d --save-baseline whittaker2d_after_cg_fusion
cargo bench --bench baseline_workloads -- whittaker_2d/brpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-brpls2d-current.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d --baseline whittaker2d_after_cg_fusion
cargo bench --bench baseline_workloads -- whittaker_2d --save-baseline whittaker2d_after_clamped_weights
cargo bench --bench baseline_workloads -- whittaker_2d/brpls_16x16 --profile-time 30
sample <baseline_workloads-pid> 5 -file /tmp/baselines-brpls2d-operator-current.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d --save-baseline whittaker2d_before_operator_split
cargo bench --bench baseline_workloads -- whittaker_2d --baseline whittaker2d_before_operator_split
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_eigen_16x16 --save-baseline arpls_eigen_before_profile
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_eigen_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-arpls-eigen.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_eigen_16x16 --baseline arpls_eigen_before_profile
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_eigen_16x16 --save-baseline arpls_eigen_after_squared_diag
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_eigen_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-arpls-eigen-after.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d/brpls_16x16 --save-baseline brpls2d_current_2026_05_24
cargo bench --bench baseline_workloads -- whittaker_2d/brpls_16x16 --baseline brpls2d_current_2026_05_24
cargo bench --bench baseline_workloads -- 'whittaker_2d/' --baseline whittaker2d_after_clamped_weights
cargo bench --bench baseline_workloads -- 'whittaker_2d_into/' --baseline whittaker_into_coverage_2026_05_24
cargo bench --bench baseline_workloads -- 'whittaker_2d/' --save-baseline whittaker2d_after_direct_banded
cargo bench --bench baseline_workloads -- 'whittaker_2d_into/' --save-baseline whittaker2d_into_after_direct_banded
cargo bench --bench baseline_workloads -- whittaker_2d/brpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-brpls2d-direct.sample.txt
cargo bench --bench baseline_workloads -- 'whittaker_2d/' --baseline whittaker2d_after_direct_banded
cargo bench --bench baseline_workloads -- 'whittaker_2d_into/' --baseline whittaker2d_into_after_direct_banded
cargo bench --bench baseline_workloads -- 'whittaker_2d/' --save-baseline whittaker2d_after_flat_banded_storage
cargo bench --bench baseline_workloads -- 'whittaker_2d_into/' --save-baseline whittaker2d_into_after_flat_banded_storage
cargo bench --bench baseline_workloads -- spline_1d/pspline_aspls_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-pspline-aspls1d.sample.txt
cargo bench --bench baseline_workloads -- spline_1d --baseline perf_before_opt
cargo bench --bench baseline_workloads -- classification_1d/dietrich_256 --profile-time 30
sample <baseline_workloads-pid> 5 -file /tmp/baselines-dietrich.sample.txt
cargo bench --bench baseline_workloads -- classification_1d --save-baseline classification1d_before_dietrich_workspace
cargo bench --bench baseline_workloads -- classification_1d --baseline classification1d_before_dietrich_workspace
cargo bench --bench baseline_workloads -- classification_1d --save-baseline classification1d_after_dietrich_workspace
cargo bench --bench baseline_workloads -- classification_1d/dietrich_256 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-dietrich-after-workspace.sample.txt
cargo bench --bench baseline_workloads -- classification_1d --baseline classification1d_after_dietrich_workspace
cargo bench --bench baseline_workloads -- classification_1d --save-baseline classification1d_after_threshold_workspace
cargo bench --bench baseline_workloads -- classification_1d --save-baseline classification1d_after_mask_variant_coverage
cargo bench --bench baseline_workloads -- whittaker_1d --save-baseline coverage_after_history_custom
cargo bench --bench baseline_workloads -- optimizers_misc_1d --save-baseline coverage_after_history_custom
cargo bench --bench baseline_workloads whittaker_1d_into -- --save-baseline whittaker_into_coverage_2026_05_24
cargo bench --bench baseline_workloads whittaker_2d_into -- --save-baseline whittaker_into_coverage_2026_05_24
cargo bench --bench baseline_workloads whittaker_1d_into/aspls_into_256 -- --save-baseline whittaker_into_coverage_2026_05_24
cargo bench --bench baseline_workloads polynomial_1d_into -- --save-baseline nonwhittaker_1d_into_coverage_2026_05_24
cargo bench --bench baseline_workloads morphology_1d_into -- --save-baseline nonwhittaker_1d_into_coverage_2026_05_24
cargo bench --bench baseline_workloads smoothing_1d_into -- --save-baseline nonwhittaker_1d_into_coverage_2026_05_24
cargo bench --bench baseline_workloads polynomial_2d_into -- --save-baseline nonwhittaker_2d_into_coverage_2026_05_24
cargo bench --bench baseline_workloads morphology_2d_into -- --save-baseline nonwhittaker_2d_into_coverage_2026_05_24
cargo bench --bench baseline_workloads spline_2d_into -- --save-baseline nonwhittaker_2d_into_coverage_2026_05_24
cargo bench --bench baseline_workloads optimizers_2d_into -- --save-baseline nonwhittaker_2d_into_coverage_2026_05_24
cargo test --test benchmark_coverage
cargo doc --workspace --all-features --no-deps
cargo package --allow-dirty
cargo bench --bench baseline_workloads -- --save-baseline perf_current_2026_05_24
```

Full saved baseline means are in
[`baseline-workloads-2026-05-24.csv`](baseline-workloads-2026-05-24.csv).
Current post-optimization full-run means are in
[`current-workloads-2026-05-24.csv`](current-workloads-2026-05-24.csv).
Optimization comparison results are in
[`optimization-results-2026-05-24.csv`](optimization-results-2026-05-24.csv).
Rejected optimization experiments are in
[`rejected-experiments-2026-05-24.csv`](rejected-experiments-2026-05-24.csv).
Benchmark coverage additions made after the initial full baseline run are in
[`coverage-additions-2026-05-24.csv`](coverage-additions-2026-05-24.csv).
That file now includes Whittaker and non-Whittaker 1D/2D reusable-output
`*_into` benchmarks, which measure the allocation-reuse API surface separately
from the allocating fit APIs.

Benchmark coverage is also checked by `tests/benchmark_coverage.rs`, which
parses the public top-level algorithm functions in the 1D, 2D, misc, and batch
modules and verifies a matching Criterion workload exists in
`benches/baseline_workloads.rs`.

Top slow paths before optimization:

| Benchmark | Mean |
| --- | ---: |
| `optimizers_misc_1d/beads_256` | 23.094 ms |
| `spline_1d/pspline_iasls_256` | 15.934 ms |
| `spline_1d/pspline_aspls_256` | 8.954 ms |
| `whittaker_2d/brpls_16x16` | 8.232 ms |
| `polynomial_1d/goldindec_256` | 6.662 ms |
| `whittaker_2d/arpls_16x16` | 6.648 ms |

Top slow paths in the current full-run snapshot:

| Benchmark | Mean |
| --- | ---: |
| `whittaker_2d/brpls_16x16` | 3.563 ms |
| `whittaker_2d/arpls_16x16` | 2.535 ms |
| `whittaker_2d/aspls_16x16` | 1.734 ms |
| `optimizers_2d/collab_pls_2x16x16` | 1.064 ms |
| `spline_1d/pspline_aspls_256` | 991.249 us |
| `whittaker_2d/airpls_16x16` | 940.155 us |

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

BEADS follow-up profiling after the banded-solver threshold:

- Target: `optimizers_misc_1d/beads_256`
- Saved Criterion baseline before the change: `beads_current_profile`
- `sample` captured 4230 samples from the normal Criterion profile run.
- A temporary no-inline profile captured 3836 samples and attributed 1968
  samples to `add_a_penalty_a_bands`, 911 samples to `solve_spd_banded`, and
  313 samples to `add_btb_tridiagonal_bands`.
- The retained change writes the BEADS penalty bands and `B^T B` bands
  directly, and rewrites `A * penalty * A` assembly to use direct band indexing
  instead of generic symmetric-band helpers.

BEADS direct band assembly result:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `optimizers_misc_1d/beads_256` | 594.097 us | 453.495 us | -23.67% |

The follow-up profile captured 3800 samples after the change. The former
`add_a_penalty_a_bands` and `add_btb_tridiagonal_bands` leaf costs were no
longer visible as named hotspots in the normal optimized profile; remaining
visible secondary costs were the tridiagonal solve, logarithms in the BEADS
loss calculation, and small allocation/free stacks. Fixture compatibility
remained passing for the pinned pybaselines references.

BEADS workspace reuse profiling:

- Target: `optimizers_misc_1d/beads_256`
- Saved Criterion baseline before the change: `beads_before_workspace_reuse`
- `sample` captured 3830 samples from the Criterion profile run.
- The profile still pointed at the main BEADS filter loop and showed
  same-sized temporary vector allocation around the band system, diagonal
  penalty, absolute-value mask, and tridiagonal application paths.
- The retained change reuses those filter-loop work buffers while keeping the
  public API and fixture-compatible dense path unchanged.

BEADS workspace reuse result:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `optimizers_misc_1d/beads_256` | 466.970 us | 455.020 us | -10.41% |

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

P-spline asPLS profiling after the solver optimization:

- Target: `spline_1d/pspline_aspls_256`
- `sample` captured 3831 samples from the Criterion profile run.
- The profile was dominated by
  `linalg::pspline::PenalizedSpline::solve_coefficients_with_options`, with
  2291 samples in that path. Secondary costs included 506 samples in the dense
  fallback solve, 155 samples in coefficient evaluation, and 129 samples in
  exponential weight calculations.
- A direct dense row-scaled P-spline workspace experiment measured 7.489 ms
  against the current optimized 0.990 ms result, so it was rejected and not
  retained.

2D P-spline profiling before optimization:

- Target: `spline_2d/pspline_iarpls_16x16`
- `sample` captured 3816 samples from the Criterion profile run.
- The profile was dominated by `two_d::spline::fit_with_policy` and
  `solve_separable_pspline`, especially repeated
  `linalg::pspline::PenalizedSpline` solve allocation and small banded solves
  for each row and column pass.

2D P-spline optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `spline_2d/pspline_asls_16x16` | 85.921 us | 60.206 us | -29.93% |
| `spline_2d/pspline_iasls_16x16` | 359.947 us | 348.561 us | -3.16% |
| `spline_2d/pspline_airpls_16x16` | 1.095 ms | 0.729 ms | -33.41% |
| `spline_2d/pspline_arpls_16x16` | 250.138 us | 170.790 us | -31.72% |
| `spline_2d/pspline_iarpls_16x16` | 1.124 ms | 0.740 ms | -34.16% |
| `spline_2d/pspline_psalsa_16x16` | 218.839 us | 144.543 us | -33.95% |
| `spline_2d/pspline_brpls_16x16` | 522.743 us | 321.805 us | -38.44% |
| `spline_2d/pspline_lsrpls_16x16` | 88.214 us | 61.569 us | -30.20% |
| `spline_2d/irsqr_16x16` | 428.911 us | 288.705 us | -32.69% |
| `spline_2d/mixture_model_16x16` | 89.067 us | 58.332 us | -34.51% |

The retained change caches the row and column spline bases in
`Spline2DWorkspace` and reuses the hot banded P-spline solve buffers instead
of rebuilding the basis, normal bands, right-hand side, and output vectors for
every row and column pass. The first-difference variant still uses the
compatibility allocation path, so its improvement is intentionally smaller.
Fixture compatibility remained passing for the pinned pybaselines references.

The 2D P-spline airPLS path was profiled again because it remained one of the
slower non-Whittaker workloads. `sample` captured 3829 samples dominated by
`PenalizedSpline::solve_into` and `solve_spd_banded_into`. A direct-indexing
rewrite of the private symmetric-band Cholesky loop measured 714.400 us before
and 715.590 us after for `spline_2d/pspline_airpls_16x16`, so it was rejected
as no significant change.

The 2D P-spline IAsLS path was then profiled because it still used the
first-difference compatibility solve. `sample` captured 3824 samples before
the change; 2341 samples were in
`PenalizedSpline::solve_coefficients_dense_with_options`, with allocator frames
visible from rebuilding dense normal equations and right-hand sides for every
row/column pass. The retained change adds an internal workspace-backed dense
first-difference solve using the existing row-major dense solver and wires the
2D separable IAsLS path through it. The public API remains unchanged.

2D P-spline first-difference workspace results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `spline_2d/pspline_iasls_16x16` | 356.970 us | 131.100 us | -63.27% |
| `spline_2d_into/pspline_iasls_into_16x16` | 358.548 us | 122.230 us | -65.91% |

The follow-up profile captured 3834 samples; the old
`solve_coefficients_dense_with_options` path disappeared, and the remaining
top cost is the expected `solve_dense_in_place` work.

The current `spline_2d/pspline_iarpls_16x16` path was also profiled after the
workspace optimizations. `sample` captured 3846 samples, mostly in
`solve_coefficients_banded_into`, `solve_spd_banded_into`, and the separable
row/column pass. A cache-locality experiment that stored the intermediate
row-smoothed surface transposed measured 706.670 us before and 716.190 us
after, so it was rejected as no significant improvement.

Goldindec profiling before optimization:

- Target: `polynomial_1d/goldindec_256`
- The sampled Criterion profile was dominated by
  `baselines::polynomial::fit_penalized_polynomial_with_threshold` repeatedly
  calling `fit_weighted_polynomial`.
- The hottest leaf was `fit_weighted_polynomial_coefficients`; allocator,
  memset, and free stacks were also prominent, pointing to repeated tiny
  polynomial-fit allocations.

Goldindec optimization result:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `polynomial_1d/goldindec_256` | 6.662 ms | 3.371 ms | -49.20% |

The retained change reuses the penalized polynomial work buffers and replaces
the repeated `Vec<Vec<_>>` normal-equation solve with a contiguous internal
workspace. Fixture compatibility remained passing for the pinned pybaselines
references.

Goldindec profiling after the first workspace optimization:

- Target: `polynomial_1d/goldindec_256`
- `sample` captured 3817 samples from the Criterion profile run.
- 3415 samples were still inside
  `polynomial::fit_penalized_polynomial_with_threshold`.
- The hottest repeated leaf was
  `fit_weighted_polynomial_coefficients_with_workspace`, showing that the
  remaining work was rebuilding fixed unweighted polynomial normal equations
  during each threshold fit.

Unweighted polynomial-fit cache optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `polynomial_1d/poly_256` | 2.812 us | 2.597 us | -7.65% |
| `polynomial_1d/modpoly_256` | 39.209 us | 17.095 us | -56.40% |
| `polynomial_1d/penalized_poly_256` | 34.112 us | 14.798 us | -56.62% |
| `polynomial_1d/goldindec_256` | 3.361 ms | 1.354 ms | -59.72% |

The retained change adds a cached unweighted polynomial workspace for repeated
fits that share the same scaled grid and order. `modpoly`, `penalized_poly`,
and `goldindec` reuse the cached powers and normal matrix across iterations;
direct `poly` uses a one-shot unweighted solve to avoid cache setup overhead.
The full `polynomial_1d` Criterion group was saved as
`polynomial1d_after_unweighted_cache` so unchanged controls can be checked
later.

Goldindec profiling after the unweighted cache optimization:

- Target: `polynomial_1d/goldindec_256`
- `sample` captured 3823 samples from the Criterion profile run.
- The profile still spent 2895 samples in
  `fit_penalized_polynomial_with_threshold`. Within that path, 2190 samples
  were under the repeated unweighted polynomial fit and 1463 samples were in
  the surrounding adjusted-value and correction loop. The dense solve accounted
  for only 75 samples.
- A direct adjusted-RHS experiment measured 1.540 ms against the saved
  1.347 ms baseline, so it was rejected.

Penalized polynomial cost-dispatch optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `polynomial_1d/penalized_poly_256` | 14.798 us | 14.218 us | -3.92% |
| `polynomial_1d/goldindec_256` | 1.354 ms | 1.207 ms | -10.82% |

The retained change hoists non-quadratic cost dispatch out of the per-point
penalized polynomial loop and precomputes the Indec threshold cubic once per
inner fit. This keeps the adjusted-vector and unweighted-fit structure that
benchmarked well, while reducing the branch and `powi` work in Goldindec's
hot correction path. The full `polynomial_1d` Criterion group was saved as
`polynomial1d_after_cost_dispatch`.

Goldindec profiling after cost-dispatch optimization:

- Target: `polynomial_1d/goldindec_256`
- `sample` captured 3819 samples from the Criterion profile run.
- The profile remained dominated by the repeated quadratic unweighted
  polynomial fits inside `fit_penalized_polynomial_with_threshold`: 2048
  samples in the threshold-fit loop, 1574 samples in
  `fit_unweighted_polynomial_coefficients_with_workspace`, 87 samples in
  `memmove`, and 77 samples in the tiny dense solve.

Goldindec quadratic-path specialization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `polynomial_1d/goldindec_256` | 1.211 ms | 0.920 ms | -24.09% |
| `polynomial_1d_into/goldindec_into_256` | 1.235 ms | 0.924 ms | -25.16% |

The retained change specializes the order-2 unweighted polynomial RHS
accumulation and polynomial evaluation paths used by Goldindec's default
quadratic fits. The dense solve, threshold search, and public API are
unchanged. Fixture compatibility remained passing for the pinned pybaselines
references.

2D polynomial profiling before basis caching:

- Target: `polynomial_2d/quant_reg_16x16`
- `sample` captured 3816 samples from the Criterion profile run.
- 3640 samples were in `two_d::polynomial::quant_reg_into`.
- The hottest leaf was `fit_weighted_surface`; repeated polynomial basis
  construction and `powi` calls accounted for most of the profile, while the
  dense solve and allocation stacks were secondary.

2D polynomial basis-workspace optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `polynomial_2d/poly_16x16` | 20.723 us | 11.443 us | -44.78% |
| `polynomial_2d/modpoly_16x16` | 150.220 us | 65.565 us | -56.35% |
| `polynomial_2d/imodpoly_16x16` | 184.343 us | 59.038 us | -67.97% |
| `polynomial_2d/penalized_poly_16x16` | 203.859 us | 82.462 us | -59.55% |
| `polynomial_2d/quant_reg_16x16` | 609.275 us | 259.076 us | -57.48% |

The retained change caches 2D polynomial basis values per shape and order,
reuses contiguous dense-solve buffers, and evaluates fitted surfaces from the
cached basis. The full `polynomial_2d` Criterion group was saved as
`polynomial2d_after_basis_workspace`.

1D morphology profiling before reflect-window splitting:

- Target: `morphology_1d/amormol_256`
- `sample` captured 3831 samples from the Criterion profile run.
- The profile was dominated by reflected moving-window primitives: 1431
  samples in `moving_max_reflect`, 1417 in `moving_min_reflect`, and 858 in
  `convolve_reflect_same`.

1D morphology reflect-window split optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `morphology_1d/rolling_ball_256` | 5.397 us | 3.373 us | -37.49% |
| `morphology_1d/tophat_256` | 4.770 us | 2.839 us | -40.48% |
| `morphology_1d/mwmv_256` | 2.983 us | 2.008 us | -32.67% |
| `morphology_1d/mor_256` | 9.507 us | 5.601 us | -41.08% |
| `morphology_1d/mpls_256` | 10.573 us | 8.670 us | -18.00% |
| `morphology_1d/imor_256` | 195.076 us | 114.797 us | -41.15% |
| `morphology_1d/mormol_256` | 118.987 us | 76.772 us | -35.48% |
| `morphology_1d/amormol_256` | 385.625 us | 226.768 us | -41.19% |
| `morphology_1d/mpspline_256` | 38.849 us | 34.641 us | -10.83% |

The retained change splits the reflected moving-window and convolution loops
into fast interior paths and boundary-only reflected paths. This keeps exact
boundary behavior while avoiding `reflect_index` calls for the common
fully-in-bounds windows. The full `morphology_1d` Criterion group was saved as
`morphology1d_after_reflect_split`.

2D Whittaker profiling:

- Target: `whittaker_2d/brpls_16x16`
- `sample` captured 3831 samples from the Criterion profile run.
- The profile was dominated by `solve_weighted_system`, with 2373 samples in
  `apply_operator` and 1429 samples in the surrounding conjugate-gradient loop.
- Target: `whittaker_2d/arpls_16x16`
- `sample` captured 4216 samples from the Criterion profile run.
- The profile had the same matrix-free CG shape, with 2632 samples in
  `apply_operator` and 1557 samples in the surrounding `solve_weighted_system`
  dot/update loops.

2D Whittaker CG loop-fusion optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `whittaker_2d/asls_16x16` | 819.674 us | 666.876 us | -18.64% |
| `whittaker_2d/iasls_16x16` | 821.791 us | 663.294 us | -19.29% |
| `whittaker_2d/airpls_16x16` | 1.180 ms | 0.979 ms | -17.08% |
| `whittaker_2d/arpls_16x16` | 6.648 ms | 5.400 ms | -18.77% |
| `whittaker_2d/drpls_16x16` | 1.374 ms | 1.115 ms | -18.85% |
| `whittaker_2d/iarpls_16x16` | 1.637 ms | 1.313 ms | -19.80% |
| `whittaker_2d/aspls_16x16` | 2.236 ms | 1.795 ms | -19.72% |
| `whittaker_2d/psalsa_16x16` | 2.094 ms | 1.701 ms | -18.77% |
| `whittaker_2d/brpls_16x16` | 8.232 ms | 6.701 ms | -18.61% |
| `whittaker_2d/lsrpls_16x16` | 821.803 us | 671.233 us | -18.32% |

The retained change fuses CG dot products and residual-norm accumulation into
the existing operator and update loops. This reduces row-major passes over the
image-sized vectors without changing the matrix-free operator or public API.
`arpls_eigen` uses a separate eigenspace solver and showed no significant
change in the same benchmark group comparison.

After CG loop fusion, `whittaker_2d/brpls_16x16` was profiled again. `sample`
captured 4215 samples, all still in `solve_weighted_system`, confirming the
remaining hotspot was the same image-sized CG/operator loop. Pre-clamping
weights once per reweighting iteration, instead of applying `max(MIN_WEIGHT)`
inside every CG operator application, produced this incremental result against
`whittaker2d_after_cg_fusion`:

| Benchmark | Before mean | After mean | Incremental change |
| --- | ---: | ---: | ---: |
| `whittaker_2d/asls_16x16` | 666.876 us | 651.440 us | -2.31% |
| `whittaker_2d/iasls_16x16` | 663.294 us | 651.268 us | -1.81% |
| `whittaker_2d/airpls_16x16` | 978.738 us | 941.755 us | -3.78% |
| `whittaker_2d/arpls_16x16` | 5.400 ms | 5.243 ms | -2.90% |
| `whittaker_2d/drpls_16x16` | 1.115 ms | 1.086 ms | -2.61% |
| `whittaker_2d/iarpls_16x16` | 1.313 ms | 1.281 ms | -2.44% |
| `whittaker_2d/aspls_16x16` | 1.795 ms | 1.754 ms | -2.28% |
| `whittaker_2d/psalsa_16x16` | 1.701 ms | 1.651 ms | -2.91% |
| `whittaker_2d/brpls_16x16` | 6.701 ms | 6.533 ms | -2.50% |
| `whittaker_2d/lsrpls_16x16` | 671.233 us | 660.735 us | -1.56% |

After the clamped-weight optimization, `whittaker_2d/brpls_16x16` was profiled
again. `sample` captured 3836 samples; `brpls_into` spent 3773 samples in
`solve_weighted_system`, while `BrPlsWeights::update` remained small. An
interior/boundary split for the second-order operator was compared against the
saved Criterion baseline `whittaker2d_before_operator_split`. It regressed
`brpls_16x16` from 6.462 ms to 7.949 ms (+23.02%) and regressed the rest of the
matrix-free Whittaker group, so the code was reverted and the experiment is
recorded in `rejected-experiments-2026-05-24.csv`.

A direct symmetric-banded Cholesky path was then added for small 2D Whittaker
systems whose flattened row-major bandwidth remains modest. The direct path is
used only for methods where the full-group benchmark showed an improvement;
`airPLS` and `asPLS` stay on CG because direct solves were no better for those
reweighting policies. The band assembly is checked against the existing
matrix-free operator, and the pinned 2D pybaselines fixture remains passing.

2D Whittaker direct-banded optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `whittaker_2d/asls_16x16` | 651.440 us | 462.984 us | -28.93% |
| `whittaker_2d/iasls_16x16` | 651.268 us | 454.635 us | -30.19% |
| `whittaker_2d/arpls_16x16` | 5.243 ms | 4.018 ms | -23.37% |
| `whittaker_2d/drpls_16x16` | 1.086 ms | 890.614 us | -17.96% |
| `whittaker_2d/iarpls_16x16` | 1.281 ms | 1.234 ms | -3.62% |
| `whittaker_2d/psalsa_16x16` | 1.651 ms | 1.122 ms | -32.04% |
| `whittaker_2d/brpls_16x16` | 6.533 ms | 5.651 ms | -13.50% |
| `whittaker_2d/lsrpls_16x16` | 660.735 us | 559.511 us | -15.32% |

Reusable-output 2D Whittaker direct-banded results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `whittaker_2d_into/asls_into_16x16` | 658.134 us | 452.962 us | -31.17% |
| `whittaker_2d_into/iasls_into_16x16` | 654.446 us | 447.508 us | -31.62% |
| `whittaker_2d_into/arpls_into_16x16` | 5.335 ms | 4.070 ms | -23.71% |
| `whittaker_2d_into/drpls_into_16x16` | 1.092 ms | 893.953 us | -18.16% |
| `whittaker_2d_into/iarpls_into_16x16` | 1.296 ms | 1.233 ms | -4.87% |
| `whittaker_2d_into/psalsa_into_16x16` | 1.673 ms | 1.113 ms | -33.44% |
| `whittaker_2d_into/brpls_into_16x16` | 6.612 ms | 5.645 ms | -14.63% |
| `whittaker_2d_into/lsrpls_into_16x16` | 661.428 us | 565.610 us | -14.49% |

A follow-up profile of `whittaker_2d/brpls_16x16` after the direct-banded
change showed the remaining cost was almost entirely inside the shared banded
Cholesky solve. `sample` captured 3838 samples; 3736 were at the top of stack in
`SymmetricBandedWorkspace::solve_spd`, while reset/fill and `BrPlsWeights`
updates were small. The retained change stores symmetric bands in one
contiguous `Vec<f64>` and rewrites the factorization/triangular-solve loops to
use direct offset indexing, avoiding per-band `Vec` indirection and repeated
`abs_diff`/`min` accessors.

2D Whittaker flat-banded storage optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `whittaker_2d/asls_16x16` | 462.984 us | 279.473 us | -39.64% |
| `whittaker_2d/iasls_16x16` | 454.635 us | 281.775 us | -38.02% |
| `whittaker_2d/arpls_16x16` | 4.018 ms | 2.535 ms | -36.90% |
| `whittaker_2d/drpls_16x16` | 890.614 us | 561.607 us | -36.94% |
| `whittaker_2d/iarpls_16x16` | 1.234 ms | 767.475 us | -37.83% |
| `whittaker_2d/psalsa_16x16` | 1.122 ms | 706.018 us | -37.08% |
| `whittaker_2d/brpls_16x16` | 5.651 ms | 3.563 ms | -36.96% |
| `whittaker_2d/lsrpls_16x16` | 559.511 us | 351.027 us | -37.26% |

Reusable-output 2D Whittaker flat-banded results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `whittaker_2d_into/asls_into_16x16` | 452.962 us | 279.743 us | -38.24% |
| `whittaker_2d_into/iasls_into_16x16` | 447.508 us | 275.639 us | -38.41% |
| `whittaker_2d_into/arpls_into_16x16` | 4.070 ms | 2.535 ms | -37.71% |
| `whittaker_2d_into/drpls_into_16x16` | 893.953 us | 559.846 us | -37.37% |
| `whittaker_2d_into/iarpls_into_16x16` | 1.233 ms | 767.857 us | -37.71% |
| `whittaker_2d_into/psalsa_into_16x16` | 1.113 ms | 695.398 us | -37.54% |
| `whittaker_2d_into/brpls_into_16x16` | 5.645 ms | 3.567 ms | -36.80% |
| `whittaker_2d_into/lsrpls_into_16x16` | 565.610 us | 349.140 us | -38.27% |

The reduced eigenspace `whittaker_2d/arpls_eigen_16x16` path was then profiled
separately because it uses its own solver. `sample` captured 3821 samples
before the change; 940 samples were in `reduced_diagonal`, which computes the
weighted reduced-system diagonal used as the CG preconditioner. The retained
change precomputes squared row/column basis values and rewrites
`reduced_diagonal` as a two-stage squared-basis projection using the existing
scratch buffer. This avoids the previous
`row_eigen * col_eigen * rows * cols` loop shape without changing the public
API or eigenspace solve.

2D Whittaker eigenspace diagonal optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `whittaker_2d/arpls_eigen_16x16` | 787.690 us | 571.920 us | -27.07% |

The follow-up profile captured 3831 samples; `reduced_diagonal` dropped to 263
samples, and the remaining top costs were the expected reduced-basis
reconstruct/project loops.

1D classification profiling before optimization:

- Target: `classification_1d/dietrich_256`
- `sample` captured 3843 samples from the Criterion profile run.
- The profile was dominated by repeated all-ones weighted polynomial fits:
  2954 samples in `fit_weighted_polynomial_coefficients_with_workspace`, plus
  visible allocation/free stacks from rebuilding the polynomial workspace.
- The next visible cost was threshold classification: 449 samples in
  `iter_threshold` after the first workspace optimization, mostly from
  repeated temporary `Vec` allocation for masked power values.

1D classification optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `classification_1d/dietrich_256` | 88.800 us | 37.107 us | -58.21% |
| `classification_1d/dietrich_256` | 37.107 us | 33.597 us | -9.46% |
| `classification_1d/fabc_256` | 25.393 us | 23.177 us | -8.73% |

The retained changes reuse the cached unweighted polynomial workspace for the
Dietrich polynomial refinement and replace threshold-classification temporary
vectors with a two-pass masked mean/std calculation plus a reusable candidate
mask. The full `classification_1d` Criterion group was saved as
`classification1d_after_threshold_workspace`.

Classification benchmark coverage additions:

| Benchmark | Mean | Criterion baseline |
| --- | ---: | --- |
| `classification_1d/std_distribution_with_mask_256` | 11.615 us | `classification1d_after_mask_variant_coverage` |
| `classification_1d/fastchrom_with_mask_256` | 8.727 us | `classification1d_after_mask_variant_coverage` |
| `whittaker_1d/asls_with_history_256` | 38.059 us | `coverage_after_history_custom` |
| `whittaker_1d/aspls_with_history_256` | 252.321 us | `coverage_after_history_custom` |
| `optimizers_misc_1d/custom_bc_with_256` | 41.420 us | `coverage_after_history_custom` |

Rejected or no-op experiments:

| Benchmark | Experiment | Before mean | After mean | Change |
| --- | --- | ---: | ---: | ---: |
| `whittaker_2d/brpls_16x16` | Precomputed operator coefficients | 8.232 ms | 9.174 ms | +10.7% |
| `whittaker_2d/brpls_16x16` | Jacobi-preconditioned CG | 8.232 ms | 12.802 ms | +55.63% |
| `optimizers_2d/collab_pls_2x16x16` | Reuse shared Whittaker workspace and fill weights in place | 1.300 ms | 1.293 ms | no significant change |
| `spline_1d/pspline_aspls_256` | Direct dense row-scaled P-spline workspace | 0.990 ms | 7.489 ms | +656.45% |
| `polynomial_1d/goldindec_256` | Direct adjusted-RHS computation with cached-power evaluation | 1.347 ms | 1.540 ms | +14.30% |
| `spline_1d/pspline_aspls_256` | Reuse asPLS residual/weight/interpolation buffers | 0.944 ms | 0.960 ms | +1.75% |
| `whittaker_1d/arpls_256` | No-allocation arPLS weight update | 0.312 ms | 0.335 ms | +7.45% |
| `whittaker_2d/brpls_16x16` | Interior/boundary split for second-order operator | 6.462 ms | 7.949 ms | +23.02% |
| `spline_1d/pspline_arpls_256` | In-place arPLS spline weight update | 0.367 ms | 0.377 ms | +2.77% |
| `whittaker_2d/brpls_16x16` | Squared residual convergence check in CG loop | 6.800 ms | 6.566 ms | no significant change |

2D morphology profiling before optimization:

- Target: `morphology_2d/imor_16x16`
- `sample` captured 3705 samples from the Criterion profile run.
- The profile was dominated by the reflected moving-window primitives:
  1853 samples in `moving_min_reflect` and 1818 samples in
  `moving_max_reflect`.

2D morphology optimization results:

| Benchmark | Before mean | After mean | Change |
| --- | ---: | ---: | ---: |
| `morphology_2d/rolling_ball_16x16` | 13.432 us | 7.834 us | -41.67% |
| `morphology_2d/tophat_16x16` | 8.886 us | 3.501 us | -60.60% |
| `morphology_2d/mor_16x16` | 18.266 us | 6.781 us | -62.88% |
| `morphology_2d/imor_16x16` | 1.537 ms | 0.410 ms | -73.35% |
| `morphology_2d/noise_median_16x16` | 25.796 us | 24.759 us | -4.02% |

The retained change computes rectangular reflected min/max operations as
separable row and column passes and reuses IMor work buffers across iterations.
`noise_median` does not use the changed min/max primitive and is shown only
because it was part of the same benchmark group comparison. Fixture
compatibility remained passing for the pinned pybaselines references.
