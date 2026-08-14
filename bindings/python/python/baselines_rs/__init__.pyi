from typing import Literal, Optional, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

Method = Literal[
    "asls", "arpls", "airpls", "rolling_ball", "polynomial"
]
Method2D = Literal["asls", "arpls", "rolling_ball", "polynomial"]

BaselineOptions = TypedDict(
    "BaselineOptions",
    {
        "lambda": float,
        "lam": float,
        "p": float,
        "max_iter": int,
        "tol": float,
        "window_size": int,
        "order": int,
    },
    total=False,
)

BaselineOptions2D = TypedDict(
    "BaselineOptions2D",
    {
        "lambda": float,
        "lam": float,
        "lambda_rows": float,
        "lambda_cols": float,
        "p": float,
        "max_iter": int,
        "tol": float,
        "cg_max_iter": int,
        "cg_tol": float,
        "window_rows": int,
        "window_cols": int,
        "order": int,
    },
    total=False,
)

class FitReport(TypedDict):
    iterations: int
    converged: bool
    tolerance: float

class FitResult(TypedDict):
    baseline: NDArray[np.float64]
    corrected: NDArray[np.float64]
    report: FitReport

def baseline(
    y: ArrayLike,
    method: Method = "asls",
    options: Optional[BaselineOptions] = None,
) -> NDArray[np.float64]: ...
def correct(
    y: ArrayLike,
    method: Method = "asls",
    options: Optional[BaselineOptions] = None,
) -> NDArray[np.float64]: ...
def fit(
    y: ArrayLike,
    method: Method = "asls",
    options: Optional[BaselineOptions] = None,
) -> FitResult: ...
def baseline_2d(
    data: ArrayLike,
    method: Method2D = "asls",
    options: Optional[BaselineOptions2D] = None,
) -> NDArray[np.float64]: ...
def correct_2d(
    data: ArrayLike,
    method: Method2D = "asls",
    options: Optional[BaselineOptions2D] = None,
) -> NDArray[np.float64]: ...
def fit_2d(
    data: ArrayLike,
    method: Method2D = "asls",
    options: Optional[BaselineOptions2D] = None,
) -> FitResult: ...
def methods() -> list[Method]: ...
def methods_2d() -> list[Method2D]: ...

__version__: str
