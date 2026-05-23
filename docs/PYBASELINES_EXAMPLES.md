# pybaselines Gallery Examples

This page tracks Rust `ruviz` examples that mirror the upstream pybaselines
gallery at <https://pybaselines.readthedocs.io/en/latest/generated/examples/index.html>.

The examples write PNGs to `target/baselines-ruviz/`.

```console
cargo run --example ruviz_pybaselines_lam_effects
cargo run --example ruviz_pybaselines_gallery_basic
cargo run --release --example ruviz_pybaselines_gallery_whittaker_sweeps
cargo run --release --example ruviz_pybaselines_gallery_whittaker_solver_timings
cargo run --release --example ruviz_pybaselines_gallery_beads_preprocessing
cargo run --release --example ruviz_pybaselines_gallery_pspline_whittaker
cargo run --release --example ruviz_pybaselines_gallery_spline_lam_vs_num_knots
```

## Coverage

| pybaselines example | Rust example | Parameter status |
| --- | --- | --- |
| `general/plot_algorithm_convergence.py` | `ruviz_pybaselines_gallery_basic` | Uses `lam=5e6`, `tol=1e-3`, `max_iter=20` and `100`; Rust reports final tolerance, but does not yet expose full `tol_history`. |
| `general/plot_masked_data.py` | `ruviz_pybaselines_gallery_basic` | Uses the same synthetic data, mask region, `lam=1e5`, and `half_window=35`; Rust does not yet expose weighted classifier fits or arPLS output weights. |
| `general/plot_noisy_data.py` | `ruviz_pybaselines_gallery_basic` | Uses the same signal, baseline, noise scale, 11-point smoothing, and `poly_order=3`; Rust `imodpoly` does not yet expose pybaselines' `num_std=0.7` parameter. |
| `general/plot_padding.py` | `ruviz_pybaselines_gallery_basic` | Uses `half_window=80`, `num_points=1000`, `pad_len=161`, and the same padding mode names. |
| `general/plot_padding_extrapolate.py` | `ruviz_pybaselines_gallery_basic` | Uses `num_points=1000`, `pad_len=100`, and `extrapolate_window` values `1`, `100`, and `[100, 40]`. |
| `general/plot_reuse_Baseline.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data and matching method parameter values where Rust APIs exist; the timing comparison is not an exact analogue because Rust does not have a reusable `Baseline` object. |
| `general/plot_sorted_data.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data and `iarpls(lam=1e6)` on forward and reversed input. |
| `whittaker/plot_lam_effects.py` | `ruviz_pybaselines_lam_effects` | Uses matching signal, baseline, noise scale, arPLS, and lambda values `1`, `1e3`, `1e6`, and `1e10`. |
| `whittaker/plot_lam_vs_data_size.py` | `ruviz_pybaselines_gallery_whittaker_sweeps` | Uses the same `_make_data` baseline formulas, data sizes `[499, 1045, 2186, 4573, 9563, 20000]`, algorithms, coarse/fine lambda search, `tol=1e-2`, and `max_iter=50`. Candidate lambda values that fail a solve are skipped, matching the upstream example behavior. |
| `whittaker/plot_whittaker_solvers.py` | `ruviz_pybaselines_gallery_whittaker_solver_timings` | Uses the same `_make_data` default signal, lambda equation, data sizes `[499, 935, 1748, 3270, 6115, 11437, 21388, 40000]`, `repeats=25`, AsLS, and `max_iter=8`. The Rust example compares allocating `asls` against reusable-workspace `asls_into` because this crate has one native pentadiagonal solver rather than SciPy/pentapy solver backends; `tol=f64::MIN_POSITIVE` is used because the Rust API requires positive tolerance while pybaselines uses `tol=-1` to force fixed iterations. |
| `morphological/plot_half_window_effects.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data and `half_window` values `30`, `60`, and `120` through Rust full-window sizes `61`, `121`, and `241`. |
| `spline/plot_lam_vs_num_knots.py` | `ruviz_pybaselines_gallery_spline_lam_vs_num_knots` | Uses the same exponential `_make_data` baseline, `mixture_model`, `diff_order=2`, knot counts `[20, 53, 141, 376, 1000]`, data sizes `[500, 1045, 2186, 4573, 9563, 20000]`, coarse/fine lambda search, `tol=1e-2`, and `max_iter=50`. |
| `spline/plot_pspline_whittaker.py` | `ruviz_pybaselines_gallery_pspline_whittaker` | Uses the same exponential `_make_data` baseline, `arpls`, `pspline_arpls`, data sizes `[499, 1045, 2186, 4573, 9563, 20000]`, coarse/fine lambda search, `tol=1e-2`, and `max_iter=50`. |
| `classification/plot_classifier_masks.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data, `std_distribution`, `half_window` values `15` and `45`, and `smooth_half_window=10`; Rust does not yet return classifier masks. |
| `classification/plot_fastchrom_threshold.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data, `half_window=15`, fixed threshold `1.5`, and default percentile threshold; callable threshold is represented by the median fallback. |
| `misc/plot_beads_preprocessing.py` | `ruviz_pybaselines_gallery_beads_preprocessing` | Uses the same 1000-point grid, signal, three baseline formulas, noise scale, endpoint parabola preprocessing formula, and BEADS parameter sets. The Rust implementation now uses a banded BEADS solve for this workload. |
| `optimizers/plot_custom_bc_1_whittaker.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data, `lam_flexible=1e2`, `lam_stiff=5e5`, `crossover_index` near `x=160`, `sampling=15`, and smoothing `lam=1e1`. |
| `two_d/plot_along_axes_1d_baseline.py` | `ruviz_pybaselines_gallery_basic` | Uses matching data and `lam=1e4`; Rust `individual_axes` currently uses AsLS rather than pybaselines' `pspline_arpls`. |
| `two_d/plot_whittaker_2d_dof.py` | pending | Rust now supports row/column-specific lambda values for native 2D Whittaker fits, but does not yet expose pybaselines' eigensolver, `num_eigens`, or DOF output. The native 2D CG counterpart should be split into a dedicated heavy example. |

Generated examples should cite pybaselines as a behavioral and documentation
reference only. The Rust implementation does not copy pybaselines implementation
code.
