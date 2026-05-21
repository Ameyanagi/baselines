#!/usr/bin/env python3
"""Generate behavioral fixtures from pybaselines.

The generated files are reference data for compatibility tests only. They do
not contain pybaselines implementation code.
"""

from __future__ import annotations

import argparse
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


def as_list(result: Any) -> list[float]:
    """Extract the baseline array from a pybaselines result."""
    baseline = result[0] if isinstance(result, tuple) else result
    return [float(value) for value in baseline]


def call_table() -> dict[str, Callable[[Baseline, list[float]], Any]]:
    """Return fixture calls with conservative parameters."""
    return {
        "poly": lambda b, y: b.poly(y, poly_order=2),
        "modpoly": lambda b, y: b.modpoly(y, poly_order=2),
        "imodpoly": lambda b, y: b.imodpoly(y, poly_order=2),
        "penalized_poly": lambda b, y: b.penalized_poly(y, poly_order=2),
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
        "pspline_iarpls": lambda b, y: b.pspline_iarpls(y, lam=1e5),
        "pspline_aspls": lambda b, y: b.pspline_aspls(y, lam=1e5),
        "pspline_psalsa": lambda b, y: b.pspline_psalsa(y, lam=1e5, p=0.5),
        "pspline_derpsalsa": lambda b, y: b.pspline_derpsalsa(y, lam=1e5, p=0.01),
        "pspline_lsrpls": lambda b, y: b.pspline_lsrpls(y, lam=1e5),
        "pspline_brpls": lambda b, y: b.pspline_brpls(y, lam=1e5),
        "pspline_mpls": lambda b, y: b.pspline_mpls(y, half_window=8),
        "rolling_ball": lambda b, y: b.rolling_ball(y, half_window=8),
        "mwmv": lambda b, y: b.mwmv(y, half_window=8),
        "tophat": lambda b, y: b.tophat(y, half_window=8),
        "mor": lambda b, y: b.mor(y, half_window=8),
        "mpls": lambda b, y: b.mpls(y, half_window=8, lam=1e6, p=0.0),
        "imor": lambda b, y: b.imor(y, half_window=8),
        "mormol": lambda b, y: b.mormol(y, half_window=8),
        "jbcd": lambda b, y: b.jbcd(y, half_window=8),
        "mpspline": lambda b, y: b.mpspline(y, half_window=8),
        "snip": lambda b, y: b.snip(y, max_half_window=8),
        "noise_median": lambda b, y: b.noise_median(y, half_window=8),
        "rubberband": lambda b, y: b.rubberband(y),
    }


def main() -> None:
    """Generate JSON fixtures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/pybaselines_1d_reference.json"),
    )
    args = parser.parse_args()

    y = signal()
    baseline = Baseline()
    payload: dict[str, Any] = {
        "pybaselines_version": pybaselines.__version__,
        "notice": (
            "Generated from pybaselines for behavioral comparison. "
            "pybaselines is BSD-3-Clause licensed and should be cited."
        ),
        "signal": y,
        "baselines": {},
    }

    for name, call in call_table().items():
        payload["baselines"][name] = as_list(call(baseline, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
