import importlib
import importlib.machinery
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


CPU_G = 6.67259e-8
GPU_G = 6.67430e-8
COLLISION_RADIUS = 1.0e10
MODULE_DIR = Path(__file__).resolve().parent
BUILD_DIR = MODULE_DIR / "compiled"


def load_native_extension(module_name):
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for directory in (BUILD_DIR, MODULE_DIR):
            candidate = directory / f"{module_name}{suffix}"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(module_name, candidate)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create import spec for {candidate}")
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
    raise ImportError(f"Could not find compiled extension for {module_name!r}")


def reference_accel(positions, masses, collision_radius, gravity_constant):
    n_bodies = len(masses)
    accel = np.zeros_like(positions, dtype=np.float64)
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i == j:
                continue
            delta = positions[i] - positions[j]
            radius = float(np.sqrt(np.dot(delta, delta)))
            radius_soft = max(radius, collision_radius)
            accel[i] += -gravity_constant * masses[j] * delta / radius_soft**3
    return accel


def max_relative_error(lhs, rhs):
    denom = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0e-300)
    return float(np.max(np.abs(lhs - rhs) / denom))


def force_balance_ratio(masses, accelerations):
    weighted_forces = masses[:, None] * accelerations
    numerator = float(np.linalg.norm(np.sum(weighted_forces, axis=0)))
    denominator = float(np.sum(np.linalg.norm(weighted_forces, axis=1)))
    return numerator / max(denominator, 1.0e-300)


def build_two_body_orbit():
    year_seconds = 3.15576e7
    primary_mass = 1.989e33
    secondary_mass = 5.972e27
    orbital_radius = 1.496e13
    mu = primary_mass + secondary_mass
    relative_speed = math.sqrt(GPU_G * mu / orbital_radius)
    primary_speed = relative_speed * secondary_mass / mu
    secondary_speed = relative_speed * primary_mass / mu

    x0 = np.array(
        [
            [-secondary_mass / mu * orbital_radius, 0.0, 0.0],
            [primary_mass / mu * orbital_radius, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    v0 = np.array(
        [
            [0.0, -primary_speed, 0.0],
            [0.0, secondary_speed, 0.0],
        ],
        dtype=np.float64,
    )
    masses = np.array([primary_mass, secondary_mass], dtype=np.float64)
    perturb_idx = np.array([], dtype=np.int32)
    perturb_pos = np.zeros((0, 3), dtype=np.float64)
    perturb_vel = np.zeros((0, 3), dtype=np.float64)
    return x0, v0, masses, perturb_idx, perturb_pos, perturb_vel, 1.0e-3 * year_seconds, year_seconds


def assert_chunk_schema(chunks, expected_rows):
    assert isinstance(chunks, list), "simulation output must be a Python list"
    assert chunks, "simulation output must contain at least one chunk"
    for chunk in chunks:
        assert isinstance(chunk, np.ndarray), "each chunk must be a NumPy array"
        assert chunk.shape[1] == 11, "each chunk must preserve the 11-column schema"
        assert chunk.shape[0] == expected_rows, "unexpected row count for chunk output"
        assert np.isfinite(chunk).all(), "chunk output contains non-finite values"


def run_cpu_accel_checks(cpu_accel):
    print("Testing CPU accel extension")
    print("-" * 50)

    x_three_body = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0e13, 0.0, 0.0],
            [0.0, 2.0e13, 0.0],
        ],
        dtype=np.float64,
    )
    m_three_body = np.array([1.989e33, 5.972e27, 5.972e27], dtype=np.float64)

    a_three_body = cpu_accel.get_accel(x_three_body, m_three_body, COLLISION_RADIUS)
    a_reference = reference_accel(x_three_body, m_three_body, COLLISION_RADIUS, CPU_G)
    rel_err = max_relative_error(a_three_body, a_reference)
    print(f"CPU reference relative error: {rel_err:.3e}")
    assert rel_err < 1.0e-12, f"unexpected CPU force error: {rel_err}"

    x_single = np.array([[3.0e11, -5.0e11, 7.0e11]], dtype=np.float64)
    m_single = np.array([4.0e30], dtype=np.float64)
    a_single = cpu_accel.get_accel(x_single, m_single, COLLISION_RADIUS)
    assert np.allclose(a_single, 0.0), "self-force should be exactly zero"

    x_pair = np.array([[-2.5e12, 0.0, 0.0], [1.5e12, 0.0, 0.0]], dtype=np.float64)
    m_pair = np.array([2.0e30, 5.0e29], dtype=np.float64)
    a_pair = cpu_accel.get_accel(x_pair, m_pair, COLLISION_RADIUS)
    third_law_residual = force_balance_ratio(m_pair, a_pair)
    print(f"CPU mass-weighted Newton residual: {third_law_residual:.3e}")
    assert third_law_residual < 1.0e-12, "CPU Newton's third law residual is too large"

    num_threads = cpu_accel.get_num_threads()
    assert int(num_threads) > 0, "expected a positive OpenMP thread count"


def run_dispatcher_checks():
    print("\nTesting Python dispatcher")
    print("-" * 50)

    accel_dispatcher = importlib.import_module("accel_dispatcher")
    backend = accel_dispatcher.get_backend()
    compute_units = accel_dispatcher.get_num_threads()
    print(f"Active backend: {backend}")
    print(f"Reported compute units: {compute_units}")
    assert backend in {"GPU", "CPU", "CPU (forced)"}
    assert int(compute_units) > 0

    forced_env = os.environ.copy()
    forced_env["NBODY_FORCE_CPU"] = "1"
    forced = subprocess.run(
        [sys.executable, "-c", "from accel_dispatcher import get_backend; print(get_backend())"],
        cwd=str(MODULE_DIR),
        env=forced_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert forced.stdout.strip() == "CPU (forced)", "forced CPU dispatcher path failed"


def run_gpu_accel_checks():
    try:
        accel_gpu = load_native_extension("accel_gpu")
    except ImportError:
        print("\naccel_gpu not available; GPU-specific checks skipped")
        return None

    print("\nTesting GPU accel extension")
    print("-" * 50)

    num_sm = accel_gpu.get_num_sm()
    print(f"Streaming multiprocessors reported: {num_sm}")
    assert int(num_sm) > 0, "expected a positive SM count"

    x_three_body = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0e13, 0.0, 0.0],
            [0.0, 2.0e13, 0.0],
        ],
        dtype=np.float64,
    )
    m_three_body = np.array([1.989e33, 5.972e27, 5.972e27], dtype=np.float64)
    a_gpu = accel_gpu.get_accel(x_three_body, m_three_body, COLLISION_RADIUS)
    a_reference = reference_accel(x_three_body, m_three_body, COLLISION_RADIUS, GPU_G)
    rel_err = max_relative_error(a_gpu, a_reference)
    print(f"GPU reference relative error (N=3): {rel_err:.3e}")
    assert rel_err < 1.0e-10, f"GPU/reference mismatch for N=3: {rel_err}"

    rng = np.random.default_rng(142)
    n_partial_tile = 1025
    x_partial = rng.uniform(-2.0e13, 2.0e13, size=(n_partial_tile, 3)).astype(np.float64)
    m_partial = rng.uniform(1.0e25, 1.0e30, size=n_partial_tile).astype(np.float64)
    a_partial = accel_gpu.get_accel(x_partial, m_partial, COLLISION_RADIUS)
    force_balance = force_balance_ratio(m_partial, a_partial)
    print(f"GPU partial-tile net-force residual: {force_balance:.3e}")
    assert np.isfinite(a_partial).all(), "GPU partial-tile path returned non-finite values"
    assert force_balance < 1.0e-10, "GPU partial-tile force balance residual is too large"

    x_small = rng.uniform(-1.0e13, 1.0e13, size=(20, 3)).astype(np.float64)
    m_small = rng.uniform(1.0e25, 1.0e29, size=20).astype(np.float64)
    accel_gpu.get_accel(x_small, m_small, COLLISION_RADIUS)
    repetitions = 10
    start = time.perf_counter()
    for _ in range(repetitions):
        accel_gpu.get_accel(x_small, m_small, COLLISION_RADIUS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repetitions
    print(f"Average GPU get_accel wall time for N=20: {elapsed_ms:.3f} ms")
    return accel_gpu


def run_gpu_simulation_checks(accel_gpu):
    try:
        simulator_gpu = load_native_extension("simulator_gpu")
    except ImportError:
        print("simulator_gpu not available; GPU simulation checks skipped")
        return

    print("\nTesting GPU simulation backend")
    print("-" * 50)

    (
        x0,
        v0,
        masses,
        perturb_idx,
        perturb_pos,
        perturb_vel,
        dt,
        year_seconds,
    ) = build_two_body_orbit()
    max_step = 1000
    chunk_size = 50
    chunks = simulator_gpu.run_simulation(
        x0,
        v0,
        masses,
        perturb_idx,
        perturb_pos,
        perturb_vel,
        1,
        max_step,
        chunk_size,
        dt,
        year_seconds,
        COLLISION_RADIUS,
    )

    assert_chunk_schema(chunks, expected_rows=len(masses))
    expected_chunk_count = math.ceil(max_step / chunk_size)
    assert len(chunks) == expected_chunk_count, "unexpected number of GPU chunks"

    initial_radius = np.linalg.norm(x0[0] - x0[1])
    initial_ke = 0.5 * np.sum(masses * np.sum(v0 * v0, axis=1))
    initial_pe = -GPU_G * masses[0] * masses[1] / initial_radius
    initial_total_energy = initial_ke + initial_pe

    final_chunk = chunks[-1]
    final_total_energy = float(np.sum(final_chunk[:, 9] + final_chunk[:, 10]))
    energy_drift = abs(final_total_energy - initial_total_energy) / abs(initial_total_energy)
    print(f"GPU orbital energy drift after {max_step} steps: {energy_drift:.3e}")
    assert energy_drift < 1.0e-2, "GPU simulation exceeded 1% relative energy drift"

    final_positions = final_chunk[:, 3:6]
    final_accel = accel_gpu.get_accel(final_positions, masses, COLLISION_RADIUS)
    net_force = force_balance_ratio(masses, final_accel)
    print(f"GPU final-state net-force residual: {net_force:.3e}")
    assert net_force < 1.0e-10, "GPU final-state Newton residual is too large"


if __name__ == "__main__":
    cpu_accel = load_native_extension("accel")
    run_cpu_accel_checks(cpu_accel)
    run_dispatcher_checks()
    accel_gpu_module = run_gpu_accel_checks()
    if accel_gpu_module is not None:
        run_gpu_simulation_checks(accel_gpu_module)
    print("\nAll available Phase 1/2 tests passed")
