# Gallery Outputs

The gallery images are generated artifacts. They are written to
`docs/assets/ruviz/`, tracked for Markdown preview, and excluded from Cargo
packages. Each output set is paired with a working Rust example in `examples/`.
The reference gallery examples are inspired by pybaselines' public example
gallery; pybaselines is credited as a documentation and behavioral reference,
and the rendered outputs are produced by this crate's native Rust examples.

Generate every gallery image:

```console
cargo run --example ruviz_1d
cargo run --example ruviz_2d
cargo run --example ruviz_lam_effects
cargo run --example ruviz_gallery_basic
cargo run --release --example ruviz_gallery_whittaker_sweeps
cargo run --release --example ruviz_gallery_whittaker_solver_timings
cargo run --release --example ruviz_gallery_beads_preprocessing
cargo run --release --example ruviz_gallery_pspline_whittaker
cargo run --release --example ruviz_gallery_spline_lam_vs_num_knots
cargo run --release --example ruviz_gallery_whittaker_2d_dof
```

Open the generated output folder on macOS:

```console
open docs/assets/ruviz
```

List generated files:

```console
find docs/assets/ruviz -maxdepth 1 -name '*.png' | sort
```

## Runnable Examples

| Example | Working code | Command | Generated outputs |
| --- | --- | --- | --- |
| Basic 1D Whittaker | [`examples/ruviz_1d.rs`](../examples/ruviz_1d.rs) | `cargo run --example ruviz_1d` | `1d_*.png` |
| Basic 2D Whittaker | [`examples/ruviz_2d.rs`](../examples/ruviz_2d.rs) | `cargo run --example ruviz_2d` | `2d_*.png` |
| Whittaker lambda effects | [`examples/ruviz_lam_effects.rs`](../examples/ruviz_lam_effects.rs) | `cargo run --example ruviz_lam_effects` | `lam_effects*.png` |
| Basic gallery batch | [`examples/ruviz_gallery_basic.rs`](../examples/ruviz_gallery_basic.rs) | `cargo run --example ruviz_gallery_basic` | `gallery_algorithm_convergence*.png`, `gallery_noisy_data.png`, `gallery_masked_data.png`, `gallery_padding*.png`, `gallery_reuse_baseline.png`, `gallery_sorted_data.png`, `gallery_half_window_effects.png`, `gallery_classifier*.png`, `gallery_fastchrom*.png`, `gallery_custom_bc_whittaker.png`, `gallery_2d_individual_axes_*.png` |
| Whittaker lambda/data-size sweeps | [`examples/ruviz_gallery_whittaker_sweeps.rs`](../examples/ruviz_gallery_whittaker_sweeps.rs) | `cargo run --release --example ruviz_gallery_whittaker_sweeps` | `gallery_lam_vs_data_size_*.png` |
| Whittaker solver timings | [`examples/ruviz_gallery_whittaker_solver_timings.rs`](../examples/ruviz_gallery_whittaker_solver_timings.rs) | `cargo run --release --example ruviz_gallery_whittaker_solver_timings` | `gallery_whittaker_solver_*.png` |
| BEADS preprocessing | [`examples/ruviz_gallery_beads_preprocessing.rs`](../examples/ruviz_gallery_beads_preprocessing.rs) | `cargo run --release --example ruviz_gallery_beads_preprocessing` | `gallery_beads_*.png` |
| P-spline vs Whittaker | [`examples/ruviz_gallery_pspline_whittaker.rs`](../examples/ruviz_gallery_pspline_whittaker.rs) | `cargo run --release --example ruviz_gallery_pspline_whittaker` | `gallery_pspline_whittaker.png` |
| Spline lambda/knot sweeps | [`examples/ruviz_gallery_spline_lam_vs_num_knots.rs`](../examples/ruviz_gallery_spline_lam_vs_num_knots.rs) | `cargo run --release --example ruviz_gallery_spline_lam_vs_num_knots` | `gallery_spline_lam_vs_num_knots_*.png` |
| 2D Whittaker DOF | [`examples/ruviz_gallery_whittaker_2d_dof.rs`](../examples/ruviz_gallery_whittaker_2d_dof.rs) | `cargo run --release --example ruviz_gallery_whittaker_2d_dof` | `gallery_whittaker_2d_*.png` |

The reproducible source of truth for every gallery image is the linked Rust
example code.

## Rendered Outputs

### Basic 1D Whittaker

![1D Whittaker baselines](assets/ruviz/1d_baselines.png)
![1D corrected signals](assets/ruviz/1d_corrected.png)

### Basic 2D Whittaker

![2D observed surface](assets/ruviz/2d_observed.png)
![2D AsLS baseline](assets/ruviz/2d_asls_baseline.png)
![2D corrected surface](assets/ruviz/2d_corrected.png)
![2D true baseline](assets/ruviz/2d_true_baseline.png)

### Whittaker Lambda Effects

![Lambda effects summary](assets/ruviz/lam_effects.png)
![Lambda 1](assets/ruviz/lam_effects_1e0.png)
![Lambda 1e3](assets/ruviz/lam_effects_1e3.png)
![Lambda 1e6](assets/ruviz/lam_effects_1e6.png)
![Lambda 1e10](assets/ruviz/lam_effects_1e10.png)

### Basic Gallery Batch

![Algorithm convergence](assets/ruviz/gallery_algorithm_convergence.png)
![Algorithm convergence tolerance](assets/ruviz/gallery_algorithm_convergence_tolerance.png)
![Noisy data](assets/ruviz/gallery_noisy_data.png)
![Masked data](assets/ruviz/gallery_masked_data.png)
![Padding](assets/ruviz/gallery_padding.png)
![Padding extrapolate](assets/ruviz/gallery_padding_extrapolate.png)
![Reuse baseline](assets/ruviz/gallery_reuse_baseline.png)
![Sorted data](assets/ruviz/gallery_sorted_data.png)
![Half window effects](assets/ruviz/gallery_half_window_effects.png)
![Classifier masks](assets/ruviz/gallery_classifier_masks.png)
![Classifier mask overlay](assets/ruviz/gallery_classifier_mask_diagnostics.png)
![FastChrom threshold](assets/ruviz/gallery_fastchrom_threshold.png)
![FastChrom rolling std](assets/ruviz/gallery_fastchrom_rolling_std.png)
![FastChrom mask overlay](assets/ruviz/gallery_fastchrom_masks.png)
![Custom BC Whittaker](assets/ruviz/gallery_custom_bc_whittaker.png)
![2D individual axes observed](assets/ruviz/gallery_2d_individual_axes_observed.png)
![2D individual axes corrected](assets/ruviz/gallery_2d_individual_axes_corrected.png)

### Whittaker Lambda/Data-Size Sweeps

![AsLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_asls.png)
![IAsLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_iasls.png)
![airPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_airpls.png)
![arPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_arpls.png)
![iarPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_iarpls.png)
![drPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_drpls.png)
![asPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_aspls.png)
![psalsa lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_psalsa.png)
![derpsalsa lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_derpsalsa.png)
![brPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_brpls.png)
![lsrPLS lambda vs data size](assets/ruviz/gallery_lam_vs_data_size_lsrpls.png)
![Exponential baseline sweep summary](assets/ruviz/gallery_lam_vs_data_size_exponential_baseline.png)
![Gaussian baseline sweep summary](assets/ruviz/gallery_lam_vs_data_size_gaussian_baseline.png)
![Sine baseline sweep summary](assets/ruviz/gallery_lam_vs_data_size_sine_baseline.png)

### Whittaker Solver Timings

![Whittaker solver timings](assets/ruviz/gallery_whittaker_solver_timings.png)
![Whittaker solver relative reduction](assets/ruviz/gallery_whittaker_solver_relative_reduction.png)

### BEADS Preprocessing

![BEADS endpoint parabola baseline 1](assets/ruviz/gallery_beads_preprocessing_baseline_1.png)
![BEADS endpoint parabola baseline 2](assets/ruviz/gallery_beads_preprocessing_baseline_2.png)
![BEADS endpoint parabola baseline 3](assets/ruviz/gallery_beads_preprocessing_baseline_3.png)
![BEADS baseline 1](assets/ruviz/gallery_beads_baseline_1.png)
![BEADS baseline 2](assets/ruviz/gallery_beads_baseline_2.png)
![BEADS baseline 3](assets/ruviz/gallery_beads_baseline_3.png)

### P-spline And Spline Sweeps

![P-spline vs Whittaker](assets/ruviz/gallery_pspline_whittaker.png)
![Spline lambda vs knot baseline](assets/ruviz/gallery_spline_lam_vs_num_knots_baseline.png)
![Spline lambda vs data size](assets/ruviz/gallery_spline_lam_vs_num_knots_data_size.png)
![Spline lambda vs knot count](assets/ruviz/gallery_spline_lam_vs_num_knots_knots.png)

### 2D Whittaker DOF

![2D Whittaker actual polynomial](assets/ruviz/gallery_2d_whittaker_actual_polynomial.png)
![2D Whittaker actual sinusoidal](assets/ruviz/gallery_2d_whittaker_actual_sinusoidal.png)
![2D Whittaker analytical polynomial](assets/ruviz/gallery_2d_whittaker_analytical_polynomial.png)
![2D Whittaker analytical sinusoidal](assets/ruviz/gallery_2d_whittaker_analytical_sinusoidal.png)
![2D Whittaker eigen 40 polynomial](assets/ruviz/gallery_2d_whittaker_eigen_40_polynomial.png)
![2D Whittaker eigen 40 sinusoidal](assets/ruviz/gallery_2d_whittaker_eigen_40_sinusoidal.png)
![2D Whittaker reduced eigen polynomial](assets/ruviz/gallery_2d_whittaker_eigen_reduced_polynomial.png)
![2D Whittaker reduced eigen sinusoidal](assets/ruviz/gallery_2d_whittaker_eigen_reduced_sinusoidal.png)
![2D Whittaker underfit eigen polynomial](assets/ruviz/gallery_2d_whittaker_eigen_underfit_polynomial.png)
![2D Whittaker underfit eigen sinusoidal](assets/ruviz/gallery_2d_whittaker_eigen_underfit_sinusoidal.png)
![2D Whittaker DOF polynomial](assets/ruviz/gallery_2d_whittaker_dof_polynomial.png)
![2D Whittaker DOF sinusoidal](assets/ruviz/gallery_2d_whittaker_dof_sinusoidal.png)
