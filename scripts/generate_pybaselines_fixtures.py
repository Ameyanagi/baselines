#!/usr/bin/env python3
"""Generate behavioral fixtures from pybaselines.

The generated files are reference data for compatibility tests only. They do
not contain pybaselines implementation code.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable

import pybaselines
from pybaselines import Baseline


def signal(n: int = 128) -> list[float]:
    """Return a deterministic synthetic baseline-correction signal."""
    values: list[float] = []
    for i in range(n):
        x = i / (n - 1)
        baseline = 0.8 + 0.2 * x + 0.05 * math.sin(2 * math.pi * x)
        peak_a = math.exp(-((x - 0.35) ** 2) / 0.0015)
        peak_b = 0.5 * math.exp(-((x - 0.72) ** 2) / 0.003)
        values.append(baseline + peak_a + peak_b)
    return values


def short_signal(n: int = 96) -> list[float]:
    """Return a shorter deterministic signal for parity spot checks."""
    values: list[float] = []
    for i in range(n):
        x = i / (n - 1)
        baseline = 0.4 + 0.25 * x
        peak = 0.7 * math.exp(-((x - 0.45) ** 2) / 0.002)
        values.append(baseline + peak)
    return values


def noisy_chromatogram_signal(n: int = 160) -> list[float]:
    """Return a deterministic chromatogram-like signal with pseudo-noise."""
    values: list[float] = []
    for i in range(n):
        x = i / (n - 1)
        baseline = 0.6 + 0.08 * x + 0.03 * math.sin(1.5 * math.pi * x)
        peak_a = 0.9 * math.exp(-((x - 0.3) ** 2) / 0.0012)
        peak_b = 0.45 * math.exp(-((x - 0.68) ** 2) / 0.006)
        noise = 0.01 * math.sin(37 * x) + 0.006 * math.cos(91 * x)
        values.append(baseline + peak_a + peak_b + noise)
    return values


def broad_baseline_signal(n: int = 192) -> list[float]:
    """Return a signal with a broad curved baseline and narrow peaks."""
    values: list[float] = []
    for i in range(n):
        x = i / (n - 1)
        baseline = 0.3 + 0.45 * (x - 0.2) ** 2 + 0.18 * x
        peak_a = 0.35 * math.exp(-((x - 0.22) ** 2) / 0.0008)
        peak_b = 0.8 * math.exp(-((x - 0.78) ** 2) / 0.002)
        values.append(baseline + peak_a + peak_b)
    return values


def mixed_peak_signal(n: int = 128) -> list[float]:
    """Return a signal with positive and negative features."""
    values: list[float] = []
    for i in range(n):
        x = i / (n - 1)
        baseline = 0.7 + 0.12 * x + 0.04 * math.sin(2 * math.pi * x)
        peak = 0.5 * math.exp(-((x - 0.35) ** 2) / 0.002)
        dip = -0.18 * math.exp(-((x - 0.62) ** 2) / 0.003)
        values.append(baseline + peak + dip)
    return values


def collab_signal(values: list[float]) -> list[float]:
    """Return a second deterministic signal for collaborative fixtures."""
    n = len(values)
    output: list[float] = []
    for i, value in enumerate(values):
        x = i / (n - 1)
        shoulder = 0.15 * math.exp(-((x - 0.55) ** 2) / 0.002)
        output.append(value + 0.03 * x + shoulder)
    return output


def as_list(result: Any) -> list[float]:
    """Extract the baseline array from a pybaselines result."""
    baseline = result[0] if isinstance(result, tuple) else result
    return [float(value) for value in baseline]


def baseline_method_names() -> list[str]:
    """Return public one-dimensional Baseline method names."""
    return sorted(
        name
        for name, value in inspect.getmembers(Baseline)
        if not name.startswith("_") and callable(value)
    )


def interp_points(y: list[float]) -> tuple[tuple[float, float], ...]:
    """Return interpolation anchors on pybaselines' default x-domain."""
    mid = len(y) // 2
    x = lambda i: -1.0 + 2.0 * i / (len(y) - 1)
    return (
        (x(0), y[0]),
        (x(mid), y[mid]),
        (x(len(y) - 1), y[-1]),
    )


def call_table() -> dict[str, Callable[[Baseline, list[float]], Any]]:
    """Return fixture calls with conservative parameters."""
    return {
        "poly": lambda b, y: b.poly(y, poly_order=2),
        "modpoly": lambda b, y: b.modpoly(y, poly_order=2),
        "imodpoly": lambda b, y: b.imodpoly(y, poly_order=2),
        "penalized_poly": lambda b, y: b.penalized_poly(y, poly_order=2),
        "loess": lambda b, y: b.loess(y, fraction=0.2, poly_order=0),
        "quant_reg": lambda b, y: b.quant_reg(y, poly_order=2, quantile=0.05),
        "goldindec": lambda b, y: b.goldindec(y, poly_order=2),
        "asls": lambda b, y: b.asls(y, lam=1e5, p=0.01),
        "iasls": lambda b, y: b.iasls(y, lam=1e5, p=0.01, lam_1=1e-4),
        "airpls": lambda b, y: b.airpls(y, lam=1e5),
        "arpls": lambda b, y: b.arpls(y, lam=1e5),
        "drpls": lambda b, y: b.drpls(y, lam=1e5, eta=0.5),
        "iarpls": lambda b, y: b.iarpls(y, lam=1e5),
        "aspls": lambda b, y: b.aspls(y, lam=1e5),
        "psalsa": lambda b, y: b.psalsa(y, lam=1e5, p=0.5),
        "derpsalsa": lambda b, y: b.derpsalsa(y, lam=1e5, p=0.01),
        "brpls": lambda b, y: b.brpls(y, lam=1e5),
        "lsrpls": lambda b, y: b.lsrpls(y, lam=1e5),
        "pspline_asls": lambda b, y: b.pspline_asls(y, lam=1e5, p=0.01),
        "pspline_iasls": lambda b, y: b.pspline_iasls(y, lam=1e5, p=0.01, lam_1=1e-4),
        "pspline_airpls": lambda b, y: b.pspline_airpls(y, lam=1e5),
        "pspline_arpls": lambda b, y: b.pspline_arpls(y, lam=1e5),
        "pspline_drpls": lambda b, y: b.pspline_drpls(y, lam=1e5, eta=0.5),
        "pspline_iarpls": lambda b, y: b.pspline_iarpls(y, lam=1e5),
        "pspline_aspls": lambda b, y: b.pspline_aspls(y, lam=1e5),
        "pspline_psalsa": lambda b, y: b.pspline_psalsa(y, lam=1e5, p=0.5),
        "pspline_derpsalsa": lambda b, y: b.pspline_derpsalsa(y, lam=1e5, p=0.01),
        "pspline_lsrpls": lambda b, y: b.pspline_lsrpls(y, lam=1e5),
        "pspline_brpls": lambda b, y: b.pspline_brpls(y, lam=1e5),
        "pspline_mpls": lambda b, y: b.pspline_mpls(y, half_window=8),
        "irsqr": lambda b, y: b.irsqr(y, lam=100, quantile=0.05),
        "mixture_model": lambda b, y: b.mixture_model(y, lam=1e5, p=0.01),
        "rolling_ball": lambda b, y: b.rolling_ball(y, half_window=8),
        "mwmv": lambda b, y: b.mwmv(y, half_window=8),
        "tophat": lambda b, y: b.tophat(y, half_window=8),
        "mor": lambda b, y: b.mor(y, half_window=8),
        "mpls": lambda b, y: b.mpls(y, half_window=8, lam=1e6, p=0.0),
        "imor": lambda b, y: b.imor(y, half_window=8),
        "mormol": lambda b, y: b.mormol(y, half_window=8),
        "amormol": lambda b, y: b.amormol(y, half_window=8),
        "jbcd": lambda b, y: b.jbcd(y, half_window=8),
        "mpspline": lambda b, y: b.mpspline(y, half_window=8),
        "snip": lambda b, y: b.snip(y, max_half_window=8),
        "noise_median": lambda b, y: b.noise_median(y, half_window=8),
        "swima": lambda b, y: b.swima(y, min_half_window=8, max_half_window=8),
        "ipsa": lambda b, y: b.ipsa(y, half_window=8, max_iter=20),
        "ria": lambda b, y: b.ria(y, half_window=8, max_iter=20),
        "peak_filling": lambda b, y: b.peak_filling(y, half_window=8, max_iter=20),
        "corner_cutting": lambda b, y: b.corner_cutting(y, max_iter=100),
        "dietrich": lambda b, y: b.dietrich(y, smooth_half_window=1),
        "golotvin": lambda b, y: b.golotvin(y, half_window=8, smooth_half_window=8),
        "std_distribution": lambda b, y: b.std_distribution(
            y, half_window=8, smooth_half_window=8
        ),
        "fastchrom": lambda b, y: b.fastchrom(y, half_window=8, smooth_half_window=8),
        "cwt_br": lambda b, y: b.cwt_br(
            y, poly_order=2, scales=[8], num_std=1.0, min_length=2
        ),
        "fabc": lambda b, y: b.fabc(y, lam=1e6, scale=8),
        "adaptive_minmax": lambda b, y: b.adaptive_minmax(
            y, poly_order=2, method="poly"
        ),
        "optimize_extended_range": lambda b, y: b.optimize_extended_range(
            y,
            method="asls",
            side="both",
            min_value=2,
            max_value=4,
            step=1,
            method_kwargs={"p": 0.01},
        ),
        "custom_bc": lambda b, y: b.custom_bc(
            y,
            method="asls",
            regions=((None, None),),
            sampling=4,
            method_kwargs={"lam": 1e5, "p": 0.01},
        ),
        "rubberband": lambda b, y: b.rubberband(y),
        "beads": lambda b, y: b.beads(y),
        "interp_pts": lambda b, y: b.interp_pts(
            y,
            baseline_points=interp_points(y),
            interp_method="linear",
        ),
    }


EXTRA_CASE_METHODS = (
    "asls",
    "arpls",
    "rolling_ball",
    "pspline_asls",
    "custom_bc",
    "rubberband",
    "beads",
)


def extra_signals() -> dict[str, list[float]]:
    """Return additional deterministic signals for targeted parity checks."""
    return {
        "short": short_signal(),
        "noisy_chromatogram": noisy_chromatogram_signal(),
        "broad_baseline": broad_baseline_signal(),
        "mixed_peaks": mixed_peak_signal(),
    }


def extra_case_methods(case_name: str) -> tuple[str, ...]:
    """Return targeted methods that are stable for a fixture case."""
    if case_name == "broad_baseline":
        return EXTRA_CASE_METHODS
    return (*EXTRA_CASE_METHODS, "cwt_br")


def main() -> None:
    """Generate JSON fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/pybaselines_1d_reference.json"),
    )
    args = parser.parse_args()

    baseline = Baseline()
    y = signal()
    calls = call_table()
    reference_case: dict[str, Any] = {
        "description": "Full 62-method reference signal.",
        "signal": y,
        "baselines": {},
    }
    payload: dict[str, Any] = {
        "pybaselines_version": pybaselines.__version__,
        "pybaselines_methods": baseline_method_names(),
        "notice": (
            "Generated from pybaselines for behavioral comparison. "
            "pybaselines is BSD-3-Clause licensed and should be cited."
        ),
        "signal": y,
        "baselines": reference_case["baselines"],
        "cases": {"reference": reference_case},
    }

    for name, call in calls.items():
        reference_case["baselines"][name] = as_list(call(baseline, y))

    collab_baselines, _ = baseline.collab_pls(
        [y, collab_signal(y)],
        method="asls",
        method_kwargs={"lam": 1e5, "p": 0.01},
    )
    payload["baselines"]["collab_pls_0"] = [float(value) for value in collab_baselines[0]]
    payload["baselines"]["collab_pls_1"] = [float(value) for value in collab_baselines[1]]

    for case_name, case_signal in extra_signals().items():
        case_baseline = Baseline()
        case = {
            "description": f"Targeted parity case for {case_name.replace('_', ' ')}.",
            "signal": case_signal,
            "baselines": {},
        }
        for method_name in extra_case_methods(case_name):
            case["baselines"][method_name] = as_list(
                calls[method_name](case_baseline, case_signal)
            )
        payload["cases"][case_name] = case

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
