# Gallery Outputs

The gallery images are generated artifacts. They are written to
`docs/assets/ruviz/`, which is ignored by git and excluded from Cargo packages.
Each output set is paired with a working Rust example in `examples/`.

Generate every gallery image:

```console
cargo run --example ruviz_1d
cargo run --example ruviz_2d
cargo run --example ruviz_pybaselines_lam_effects
cargo run --example ruviz_pybaselines_gallery_basic
cargo run --release --example ruviz_pybaselines_gallery_whittaker_sweeps
cargo run --release --example ruviz_pybaselines_gallery_whittaker_solver_timings
cargo run --release --example ruviz_pybaselines_gallery_beads_preprocessing
cargo run --release --example ruviz_pybaselines_gallery_pspline_whittaker
cargo run --release --example ruviz_pybaselines_gallery_spline_lam_vs_num_knots
cargo run --release --example ruviz_pybaselines_gallery_whittaker_2d_dof
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
| pybaselines Whittaker lambda effects | [`examples/ruviz_pybaselines_lam_effects.rs`](../examples/ruviz_pybaselines_lam_effects.rs) | `cargo run --example ruviz_pybaselines_lam_effects` | `pybaselines_lam_effects*.png` |
| pybaselines basic gallery batch | [`examples/ruviz_pybaselines_gallery_basic.rs`](../examples/ruviz_pybaselines_gallery_basic.rs) | `cargo run --example ruviz_pybaselines_gallery_basic` | `pybaselines_gallery_algorithm_convergence*.png`, `pybaselines_gallery_noisy_data.png`, `pybaselines_gallery_masked_data.png`, `pybaselines_gallery_padding*.png`, `pybaselines_gallery_reuse_baseline.png`, `pybaselines_gallery_sorted_data.png`, `pybaselines_gallery_half_window_effects.png`, `pybaselines_gallery_classifier*.png`, `pybaselines_gallery_fastchrom*.png`, `pybaselines_gallery_custom_bc_whittaker.png`, `pybaselines_gallery_2d_individual_axes_*.png` |
| pybaselines Whittaker lambda/data-size sweeps | [`examples/ruviz_pybaselines_gallery_whittaker_sweeps.rs`](../examples/ruviz_pybaselines_gallery_whittaker_sweeps.rs) | `cargo run --release --example ruviz_pybaselines_gallery_whittaker_sweeps` | `pybaselines_gallery_lam_vs_data_size_*.png` |
| pybaselines Whittaker solver timings | [`examples/ruviz_pybaselines_gallery_whittaker_solver_timings.rs`](../examples/ruviz_pybaselines_gallery_whittaker_solver_timings.rs) | `cargo run --release --example ruviz_pybaselines_gallery_whittaker_solver_timings` | `pybaselines_gallery_whittaker_solver_*.png` |
| pybaselines BEADS preprocessing | [`examples/ruviz_pybaselines_gallery_beads_preprocessing.rs`](../examples/ruviz_pybaselines_gallery_beads_preprocessing.rs) | `cargo run --release --example ruviz_pybaselines_gallery_beads_preprocessing` | `pybaselines_gallery_beads_*.png` |
| pybaselines P-spline vs Whittaker | [`examples/ruviz_pybaselines_gallery_pspline_whittaker.rs`](../examples/ruviz_pybaselines_gallery_pspline_whittaker.rs) | `cargo run --release --example ruviz_pybaselines_gallery_pspline_whittaker` | `pybaselines_gallery_pspline_whittaker.png` |
| pybaselines spline lambda/knot sweeps | [`examples/ruviz_pybaselines_gallery_spline_lam_vs_num_knots.rs`](../examples/ruviz_pybaselines_gallery_spline_lam_vs_num_knots.rs) | `cargo run --release --example ruviz_pybaselines_gallery_spline_lam_vs_num_knots` | `pybaselines_gallery_spline_lam_vs_num_knots_*.png` |
| pybaselines 2D Whittaker DOF | [`examples/ruviz_pybaselines_gallery_whittaker_2d_dof.rs`](../examples/ruviz_pybaselines_gallery_whittaker_2d_dof.rs) | `cargo run --release --example ruviz_pybaselines_gallery_whittaker_2d_dof` | `pybaselines_gallery_whittaker_2d_*.png` |

The generated PNGs are intentionally not source artifacts. The reproducible
source of truth for every gallery image is the linked Rust example code.
