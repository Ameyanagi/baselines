#!/usr/bin/env python3
"""Generate two-dimensional behavioral fixtures from pybaselines.

The generated files are reference data for compatibility tests only. They do
not contain pybaselines implementation code.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pybaselines
from pybaselines import Baseline2D


Shape = tuple[int, int]
SurfaceCall = Callable[[Baseline2D, np.ndarray], Any]


def reference_surface(shape: Shape = (24, 26)) -> np.ndarray:
    """Return a deterministic curved surface with peaks and a ridge."""
    rows, cols = shape
    values = np.empty(shape, dtype=float)
    for row in range(rows):
        y = row / (rows - 1)
        for col in range(cols):
            x = col / (cols - 1)
            baseline = (
                0.4
                + 0.2 * x
                + 0.15 * y
                + 0.08 * (x - 0.4) ** 2
                + 0.05 * math.sin(math.pi * x) * math.cos(math.pi * y)
            )
            peak = 0.7 * math.exp(-(((x - 0.35) ** 2) / 0.004 + ((y - 0.45) ** 2) / 0.01))
            ridge = 0.25 * math.exp(-((x - y * 0.7 - 0.15) ** 2) / 0.003)
            values[row, col] = baseline + peak + ridge
    return values


def tilted_plane_surface(shape: Shape = (18, 20)) -> np.ndarray:
    """Return a tilted plane with two compact positive peaks."""
    rows, cols = shape
    values = np.empty(shape, dtype=float)
    for row in range(rows):
        y = row / (rows - 1)
        for col in range(cols):
            x = col / (cols - 1)
            baseline = 0.25 + 0.4 * x + 0.15 * y
            peak_a = 0.3 * math.exp(-(((x - 0.25) ** 2) / 0.003 + ((y - 0.35) ** 2) / 0.006))
            peak_b = 0.45 * math.exp(-(((x - 0.75) ** 2) / 0.006 + ((y - 0.65) ** 2) / 0.004))
            values[row, col] = baseline + peak_a + peak_b
    return values


def ridge_valley_surface(shape: Shape = (20, 18)) -> np.ndarray:
    """Return a deterministic surface with a ridge and a shallow valley."""
    rows, cols = shape
    values = np.empty(shape, dtype=float)
    for row in range(rows):
        y = row / (rows - 1)
        for col in range(cols):
            x = col / (cols - 1)
            baseline = 0.5 + 0.15 * (x - 0.5) ** 2 + 0.1 * y
            ridge = 0.35 * math.exp(-((x - 0.55 * y - 0.2) ** 2) / 0.004)
            valley = -0.12 * math.exp(-(((x - 0.7) ** 2) / 0.01 + ((y - 0.3) ** 2) / 0.02))
            values[row, col] = baseline + ridge + valley
    return values


def noisy_surface(shape: Shape = (22, 22)) -> np.ndarray:
    """Return an image-like deterministic surface with pseudo-noise."""
    rows, cols = shape
    values = np.empty(shape, dtype=float)
    for row in range(rows):
        y = row / (rows - 1)
        for col in range(cols):
            x = col / (cols - 1)
            baseline = 0.35 + 0.12 * x + 0.18 * y + 0.04 * math.sin(2 * math.pi * x)
            peak = 0.5 * math.exp(-(((x - 0.45) ** 2) / 0.005 + ((y - 0.58) ** 2) / 0.008))
            noise = 0.012 * math.sin(37 * x + 11 * y) + 0.008 * math.cos(19 * x - 23 * y)
            values[row, col] = baseline + peak + noise
    return values


def collab_surface(values: np.ndarray) -> np.ndarray:
    """Return a second deterministic surface for collaborative fixtures."""
    rows, cols = values.shape
    modifier = np.empty_like(values)
    for row in range(rows):
        y = row / (rows - 1)
        for col in range(cols):
            x = col / (cols - 1)
            modifier[row, col] = 0.03 * x + 0.02 * y
    return values + modifier


def baseline_method_names() -> list[str]:
    """Return public two-dimensional Baseline2D method names."""
    return sorted(
        name
        for name, value in inspect.getmembers(Baseline2D)
        if not name.startswith("_") and callable(value)
    )


def flat(values: np.ndarray) -> list[float]:
    """Return row-major finite float values."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("pybaselines produced a non-finite baseline")
    return [float(value) for value in array.ravel(order="C")]


def as_baseline(result: Any) -> np.ndarray:
    """Extract the baseline array from a pybaselines result."""
    return np.asarray(result[0] if isinstance(result, tuple) else result, dtype=float)


def call_table() -> dict[str, SurfaceCall]:
    """Return fixture calls with conservative parameters."""
    eigens = (8, 8)
    return {
        "poly": lambda b, z: b.poly(z, poly_order=2),
        "modpoly": lambda b, z: b.modpoly(z, poly_order=2, max_iter=20),
        "imodpoly": lambda b, z: b.imodpoly(z, poly_order=2, max_iter=20),
        "penalized_poly": lambda b, z: b.penalized_poly(z, poly_order=2, max_iter=20),
        "quant_reg": lambda b, z: b.quant_reg(z, poly_order=2, quantile=0.05, max_iter=20),
        "asls": lambda b, z: b.asls(z, lam=1e4, p=0.01, num_eigens=eigens),
        "iasls": lambda b, z: b.iasls(z, lam=1e4, p=0.01, lam_1=1e-4),
        "airpls": lambda b, z: b.airpls(z, lam=1e4, num_eigens=eigens),
        "arpls": lambda b, z: b.arpls(z, lam=1e4, num_eigens=eigens),
        "drpls": lambda b, z: b.drpls(z, lam=1e4, eta=0.5),
        "iarpls": lambda b, z: b.iarpls(z, lam=1e4, num_eigens=eigens),
        "aspls": lambda b, z: b.aspls(z, lam=1e4, max_iter=20),
        "psalsa": lambda b, z: b.psalsa(z, lam=1e4, p=0.5, num_eigens=eigens),
        "brpls": lambda b, z: b.brpls(
            z,
            lam=1e3,
            max_iter=20,
            max_iter_2=10,
            num_eigens=eigens,
        ),
        "lsrpls": lambda b, z: b.lsrpls(z, lam=1e3, num_eigens=eigens),
        "rolling_ball": lambda b, z: b.rolling_ball(z, half_window=3),
        "tophat": lambda b, z: b.tophat(z, half_window=3),
        "mor": lambda b, z: b.mor(z, half_window=3),
        "imor": lambda b, z: b.imor(z, half_window=3, max_iter=20),
        "noise_median": lambda b, z: b.noise_median(z, half_window=3),
        "pspline_asls": lambda b, z: b.pspline_asls(z, lam=1e3, p=0.01, num_knots=eigens),
        "pspline_iasls": lambda b, z: b.pspline_iasls(
            z,
            lam=1e3,
            p=0.01,
            lam_1=1e-4,
            num_knots=eigens,
        ),
        "pspline_airpls": lambda b, z: b.pspline_airpls(z, lam=1e3, num_knots=eigens),
        "pspline_arpls": lambda b, z: b.pspline_arpls(z, lam=1e3, num_knots=eigens),
        "pspline_iarpls": lambda b, z: b.pspline_iarpls(z, lam=1e3, num_knots=eigens),
        "pspline_psalsa": lambda b, z: b.pspline_psalsa(
            z,
            lam=1e3,
            p=0.5,
            num_knots=eigens,
        ),
        "pspline_brpls": lambda b, z: b.pspline_brpls(
            z,
            lam=1e3,
            num_knots=eigens,
            max_iter=20,
            max_iter_2=10,
        ),
        "pspline_lsrpls": lambda b, z: b.pspline_lsrpls(z, lam=1e3, num_knots=eigens),
        "irsqr": lambda b, z: b.irsqr(
            z,
            lam=1e3,
            quantile=0.05,
            num_knots=eigens,
            max_iter=20,
        ),
        "mixture_model": lambda b, z: b.mixture_model(
            z,
            lam=1e3,
            p=0.01,
            num_knots=eigens,
            max_iter=20,
        ),
        "adaptive_minmax": lambda b, z: b.adaptive_minmax(
            z,
            poly_order=2,
            method="modpoly",
            method_kwargs={"max_iter": 20},
        ),
        "individual_axes": lambda b, z: b.individual_axes(
            z,
            axes=(0, 1),
            method="asls",
            method_kwargs={"lam": 1e4, "p": 0.01},
        ),
    }


EXTRA_CASE_METHODS = (
    "poly",
    "asls",
    "rolling_ball",
    "pspline_asls",
    "individual_axes",
)


def extra_surfaces() -> dict[str, np.ndarray]:
    """Return additional deterministic surfaces for targeted parity checks."""
    return {
        "tilted_plane": tilted_plane_surface(),
        "ridge_valley": ridge_valley_surface(),
        "noisy": noisy_surface(),
    }


def case_payload(description: str, values: np.ndarray) -> dict[str, Any]:
    """Return a JSON payload for one surface case."""
    rows, cols = values.shape
    return {
        "description": description,
        "shape": [rows, cols],
        "signal": flat(values),
        "baselines": {},
    }


def main() -> None:
    """Generate JSON fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/pybaselines_2d_reference.json"),
        help="Path to write the JSON fixture.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    z = reference_surface()
    calls = call_table()
    reference_case = case_payload("Full 33-method two-dimensional reference surface.", z)
    payload: dict[str, Any] = {
        "pybaselines_version": pybaselines.__version__,
        "pybaselines_methods": baseline_method_names(),
        "notice": (
            "Generated from pybaselines for behavioral comparison. "
            "pybaselines is BSD-3-Clause licensed and should be cited."
        ),
        "shape": reference_case["shape"],
        "signal": reference_case["signal"],
        "baselines": reference_case["baselines"],
        "cases": {"reference": reference_case},
    }

    for name, call in calls.items():
        reference_case["baselines"][name] = flat(as_baseline(call(Baseline2D(), z)))

    collab_baselines, _ = Baseline2D().collab_pls(
        [z, collab_surface(z)],
        method="asls",
        method_kwargs={"lam": 1e4, "p": 0.01, "num_eigens": (8, 8)},
    )
    payload["baselines"]["collab_pls_0"] = flat(np.asarray(collab_baselines[0]))
    payload["baselines"]["collab_pls_1"] = flat(np.asarray(collab_baselines[1]))

    for case_name, values in extra_surfaces().items():
        case = case_payload(f"Targeted 2D parity case for {case_name.replace('_', ' ')}.", values)
        for method_name in EXTRA_CASE_METHODS:
            case["baselines"][method_name] = flat(
                as_baseline(calls[method_name](Baseline2D(), values))
            )
        payload["cases"][case_name] = case

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
