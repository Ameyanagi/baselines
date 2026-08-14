# baselines-rs

<p align="center">
  <img src="https://raw.githubusercontent.com/Ameyanagi/baselines/main/docs/assets/branding/baselines-icon-midnight-hex-v2.png" alt="baselines logo" width="220">
</p>

Python bindings for the Rust `baselines` crate. Python lists and NumPy arrays of
any numeric dtype are accepted and converted to `float64` arrays.

```python
import numpy as np
import baselines_rs

y = [1.0, 1.1, 4.2, 1.2, 1.0]
estimated = baselines_rs.baseline(
    y,
    method="asls",
    options={"lam": 1e5, "p": 0.05},
)
corrected = baselines_rs.correct(y, method="arpls")
```

Two-dimensional nested lists and NumPy arrays are accepted by `baseline_2d`
and `correct_2d`.
Call `methods()` or `methods_2d()` to list the available method names. Use
`fit` or `fit_2d` when convergence metadata and both outputs are useful:

```python
result = baselines_rs.fit(y, method="arpls", options={"tol": 1e-4})
print(result["baseline"], result["corrected"])
print(result["report"])
```

The package includes type information for IDEs and static type checkers.
