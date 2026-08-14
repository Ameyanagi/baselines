"""Small Python API for Rust-powered baseline correction."""

from ._native import (
    baseline,
    baseline_2d,
    correct,
    correct_2d,
    fit,
    fit_2d,
    methods,
    methods_2d,
)

__all__ = [
    "baseline",
    "baseline_2d",
    "correct",
    "correct_2d",
    "fit",
    "fit_2d",
    "methods",
    "methods_2d",
]
__version__ = "0.2.0"
