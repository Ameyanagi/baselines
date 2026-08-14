import numpy as np

import baselines_rs


y = [1.0, 1.1, 4.2, 1.2, 1.0]
estimated = baselines_rs.baseline(
    y,
    method="asls",
    options={"lam": 1.0e4, "p": 0.05, "max_iter": 10},
)
result = baselines_rs.fit(y, method="arpls", options={"tol": 1.0e-4})
minimum_length = baselines_rs.baseline([3, 2, 3])

assert estimated.shape == (5,)
assert estimated.dtype == np.float64
assert minimum_length.shape == (3,)
assert np.isfinite(minimum_length).all()
assert result["corrected"].shape == (5,)
assert result["report"]["iterations"] >= 1
assert isinstance(result["report"]["converged"], bool)
assert baselines_rs.methods() == [
    "asls",
    "arpls",
    "airpls",
    "rolling_ball",
    "polynomial",
]

surface = [[1, 1, 1], [1, 4, 1]]
surface_result = baselines_rs.fit_2d(
    surface,
    method="rolling_ball",
    options={"window_rows": 3, "window_cols": 3},
)
assert surface_result["baseline"].shape == (2, 3)
assert surface_result["corrected"].shape == (2, 3)

try:
    baselines_rs.baseline(y, method="polynomial", options={"lambda": 1.0e4})
except ValueError as error:
    assert "not supported" in str(error)
else:
    raise AssertionError("unsupported options should fail")

print("Python binding smoke test: OK")
