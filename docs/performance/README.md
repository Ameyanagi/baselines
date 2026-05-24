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
cargo bench --bench baseline_workloads -- whittaker_2d/arpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-arpls2d.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d --baseline perf_before_opt
cargo bench --bench baseline_workloads -- whittaker_2d --save-baseline whittaker2d_after_cg_fusion
cargo bench --bench baseline_workloads -- whittaker_2d/brpls_16x16 --profile-time 20
sample <baseline_workloads-pid> 5 -file /tmp/baselines-brpls2d-current.sample.txt
cargo bench --bench baseline_workloads -- whittaker_2d --baseline whittaker2d_after_cg_fusion
cargo bench --bench baseline_workloads -- whittaker_2d --save-baseline whittaker2d_after_clamped_weights
```

Full saved baseline means are in
[`baseline-workloads-2026-05-24.csv`](baseline-workloads-2026-05-24.csv).
Optimization comparison results are in
[`optimization-results-2026-05-24.csv`](optimization-results-2026-05-24.csv).
Rejected optimization experiments are in
[`rejected-experiments-2026-05-24.csv`](rejected-experiments-2026-05-24.csv).

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

Rejected or no-op 2D experiments:

| Benchmark | Experiment | Before mean | After mean | Change |
| --- | --- | ---: | ---: | ---: |
| `whittaker_2d/brpls_16x16` | Precomputed operator coefficients | 8.232 ms | 9.174 ms | +10.7% |
| `whittaker_2d/brpls_16x16` | Jacobi-preconditioned CG | 8.232 ms | 12.802 ms | +55.63% |
| `optimizers_2d/collab_pls_2x16x16` | Reuse shared Whittaker workspace and fill weights in place | 1.300 ms | 1.293 ms | no significant change |

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
