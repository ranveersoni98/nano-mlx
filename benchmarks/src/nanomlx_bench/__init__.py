"""Bits shared by the benchmark scripts: machine info plus JSON saving."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _optional(fn: Callable[[], Any]) -> Any | None:
    """Try something that might not be there. Missing tools just give None."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - probes shouldn't ever crash the bench
        return None


def machine_specs() -> dict:
    """What chip, cores, RAM, OS, and lib versions this run used. Goes in every file."""
    specs: dict = {
        "chip": platform.machine(),
        "os": platform.mac_ver()[0] or platform.platform(),
        "python": platform.python_version(),
    }

    def cpu_brand() -> str:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip()

    def ram_gb() -> float:
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)

    def cpu_count() -> int:
        import psutil

        count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
        if count is None:
            raise RuntimeError("could not determine CPU count")
        return int(count)

    def rust_version() -> str:
        out = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip()

    for key, fn in [
        ("cpu_brand", cpu_brand),
        ("ram_gb", ram_gb),
        ("cpu_count", cpu_count),
        ("rust_version", rust_version),
    ]:
        value = _optional(fn)
        if value:
            specs[key] = value

    for lib in ["mlx", "numpy", "tokenizers", "tiktoken", "regex"]:
        mod = _optional(lambda lib=lib: __import__(lib))
        if mod is not None:
            ver = getattr(mod, "__version__", None)
            if ver is None:
                ver = _optional(
                    lambda lib=lib: __import__("importlib.metadata").metadata.version(lib)
                )
            specs[f"{lib}_version"] = ver or "?"

    def power() -> dict:
        import re

        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        src = "AC" if "AC Power" in out else ("battery" if "Battery Power" in out else "?")
        m = re.search(r"(\d+)%;", out)
        return {"power_source": src, "battery_pct": int(m.group(1)) if m else None}

    p = _optional(power)
    if p:
        specs.update(p)
    return specs


def write_result(name: str, payload: dict) -> Path:
    """Save benchmarks/results/<name>.json with the machine info stuck on."""
    results = Path(__file__).resolve().parents[2] / "results"
    results.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine_specs(), **payload}
    path = results / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
