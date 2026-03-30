"""
Backend dispatcher for N-body acceleration and simulation.

Attempts to load the GPU backend first unless NBODY_FORCE_CPU=1 is set.
Falls back to the CPU backend when the GPU extensions are unavailable.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import warnings


_MODULE_DIR = Path(__file__).resolve().parent
_FORCE_CPU = os.environ.get("NBODY_FORCE_CPU", "0") == "1"


def _find_extension_path(stem: str) -> Path:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _MODULE_DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise ImportError(f"Could not find compiled extension for {stem!r} in {_MODULE_DIR}")


def _load_extension_with_init_name(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is not None:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
    return module


def _load_cpu_accel_backend() -> ModuleType:
    return _load_extension_with_init_name("accel", _find_extension_path("accel"))


def _load_cpu_backend() -> tuple[ModuleType, ModuleType, str]:
    accel_backend = _load_cpu_accel_backend()
    sim_backend = importlib.import_module("simulator")
    return accel_backend, sim_backend, "CPU (forced)" if _FORCE_CPU else "CPU"


def _load_gpu_backend() -> tuple[ModuleType, ModuleType, str]:
    accel_backend = importlib.import_module("accel_gpu")
    sim_backend = importlib.import_module("simulator_gpu")
    return accel_backend, sim_backend, "GPU"


if _FORCE_CPU:
    _accel_backend, _sim_backend, _BACKEND = _load_cpu_backend()
else:
    try:
        _accel_backend, _sim_backend, _BACKEND = _load_gpu_backend()
    except ImportError as exc:
        warnings.warn(
            (
                f"GPU backend unavailable ({exc}). Falling back to CPU. "
                "Set NBODY_FORCE_CPU=1 to force the CPU path explicitly."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        _accel_backend, _sim_backend, _BACKEND = _load_cpu_backend()


def get_accel(R, M, collision_radius):
    """Compute gravitational accelerations with the active backend."""
    return _accel_backend.get_accel(R, M, collision_radius)


def run_simulation(
    X0,
    V0,
    M,
    perturb_idx,
    perturb_pos,
    perturb_vel,
    sim_id,
    max_step,
    chunk_size,
    dt,
    yr,
    collision_radius,
):
    """Run the active simulation backend with the simulator.cpp-compatible signature."""
    runner = getattr(_sim_backend, "run_simulation", None)
    if runner is None:
        runner = getattr(_sim_backend, "run_simulation_gpu")
    return runner(
        X0,
        V0,
        M,
        perturb_idx,
        perturb_pos,
        perturb_vel,
        sim_id,
        max_step,
        chunk_size,
        dt,
        yr,
        collision_radius,
    )


def get_backend() -> str:
    """Return the active compute backend label."""
    return _BACKEND


def get_num_threads() -> int:
    """Return the active backend's parallelism diagnostic."""
    if hasattr(_accel_backend, "get_num_sm"):
        return int(_accel_backend.get_num_sm())
    return int(_accel_backend.get_num_threads())


def get_num_sm() -> int:
    """Return the GPU SM count when the GPU backend is active."""
    if hasattr(_accel_backend, "get_num_sm"):
        return int(_accel_backend.get_num_sm())
    raise AttributeError("SM count is only available when the GPU backend is active")


__all__ = [
    "get_accel",
    "get_backend",
    "get_num_sm",
    "get_num_threads",
    "run_simulation",
]
