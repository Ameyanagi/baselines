# baselines-rs

<p align="center">
  <img src="https://raw.githubusercontent.com/Ameyanagi/baselines/main/docs/assets/branding/baselines-icon-midnight-hex-v2.png" alt="baselines logo" width="220">
</p>

Python bindings for the Rust `baselines` crate. The first release intentionally
exposes a small default-based API; use the Rust crate directly for the complete
parameter and workspace APIs.

```python
import numpy as np
import baselines_rs

y = np.array([1.0, 1.1, 4.2, 1.2, 1.0])
estimated = baselines_rs.baseline(y)
corrected = baselines_rs.correct(y, method="arpls")
```

Two-dimensional NumPy arrays are accepted by `baseline_2d` and `correct_2d`.
Call `methods()` or `methods_2d()` to list the available method names.
