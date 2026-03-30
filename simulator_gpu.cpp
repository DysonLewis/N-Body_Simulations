#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "accel_gpu_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <vector>

namespace {

constexpr double ADAPTIVE_SAFETY_FACTOR = 5.0e-4;
constexpr double MIN_DT_FACTOR = 1.0e-4;
constexpr long double COLLISION_DISTANCE_EPS = 1.0e-10L;
constexpr double RELATIVE_VELOCITY_EPS = 1.0e-9;
constexpr long double RESTITUTION_VERY_CLOSE = 0.97L;
constexpr long double RESTITUTION_CLOSE = 0.98L;
constexpr long double RESTITUTION_APPROACHING = 0.999L;
constexpr int VIRIAL_CHECK_INTERVAL = 100;
constexpr int VIRIAL_WINDOW_SIZE = 400;
constexpr double VIRIAL_TOLERANCE = 0.03;
constexpr int VIRIAL_MIN_STEPS = 100000;
constexpr double VIRIAL_MIN_TIME_YR = 2000.0;
constexpr int OUTPUT_COLUMNS = 11;

double compute_adaptive_timestep(
    const std::vector<double>& X,
    const std::vector<double>& V,
    int N,
    double base_dt,
    double collision_radius) {
    double min_dt = base_dt;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const double dx = X[i * 3 + 0] - X[j * 3 + 0];
            const double dy = X[i * 3 + 1] - X[j * 3 + 1];
            const double dz = X[i * 3 + 2] - X[j * 3 + 2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 4.0 * collision_radius) {
                const double vx = V[i * 3 + 0] - V[j * 3 + 0];
                const double vy = V[i * 3 + 1] - V[j * 3 + 1];
                const double vz = V[i * 3 + 2] - V[j * 3 + 2];
                const double v_rel = std::sqrt(vx * vx + vy * vy + vz * vz);
                if (v_rel > RELATIVE_VELOCITY_EPS) {
                    const double required_dt =
                        ADAPTIVE_SAFETY_FACTOR * (collision_radius / v_rel);
                    min_dt = std::min(min_dt, required_dt);
                }
            }
        }
    }
    min_dt = std::max(min_dt, base_dt * MIN_DT_FACTOR);
    min_dt = std::min(min_dt, base_dt);
    return min_dt;
}

bool handle_collisions_gpu(
    std::vector<double>& X,
    std::vector<double>& V,
    const std::vector<double>& M,
    int N,
    double collision_radius) {
    bool modified = false;
    const long double coll_rad = static_cast<long double>(collision_radius);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const long double dx =
                static_cast<long double>(X[i * 3 + 0]) - static_cast<long double>(X[j * 3 + 0]);
            const long double dy =
                static_cast<long double>(X[i * 3 + 1]) - static_cast<long double>(X[j * 3 + 1]);
            const long double dz =
                static_cast<long double>(X[i * 3 + 2]) - static_cast<long double>(X[j * 3 + 2]);
            const long double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < coll_rad * 3.0L && r > COLLISION_DISTANCE_EPS) {
                const long double nx = dx / r;
                const long double ny = dy / r;
                const long double nz = dz / r;
                const long double v_rel_x =
                    static_cast<long double>(V[i * 3 + 0]) - static_cast<long double>(V[j * 3 + 0]);
                const long double v_rel_y =
                    static_cast<long double>(V[i * 3 + 1]) - static_cast<long double>(V[j * 3 + 1]);
                const long double v_rel_z =
                    static_cast<long double>(V[i * 3 + 2]) - static_cast<long double>(V[j * 3 + 2]);
                const long double v_rel_n = v_rel_x * nx + v_rel_y * ny + v_rel_z * nz;
                if (v_rel_n < 0.0L) {
                    const long double m_i = static_cast<long double>(M[i]);
                    const long double m_j = static_cast<long double>(M[j]);
                    const long double m_total = m_i + m_j;
                    long double restitution = RESTITUTION_APPROACHING;
                    if (r < coll_rad) {
                        restitution = RESTITUTION_VERY_CLOSE;
                    } else if (r < coll_rad * 1.2L) {
                        restitution = RESTITUTION_CLOSE;
                    }
                    const long double impulse =
                        2.0L * m_i * m_j * v_rel_n / m_total * restitution;
                    V[i * 3 + 0] -= static_cast<double>((impulse / m_i) * nx);
                    V[i * 3 + 1] -= static_cast<double>((impulse / m_i) * ny);
                    V[i * 3 + 2] -= static_cast<double>((impulse / m_i) * nz);
                    V[j * 3 + 0] += static_cast<double>((impulse / m_j) * nx);
                    V[j * 3 + 1] += static_cast<double>((impulse / m_j) * ny);
                    V[j * 3 + 2] += static_cast<double>((impulse / m_j) * nz);
                    modified = true;
                }
            }
        }
    }
    return modified;
}

bool check_virial_equilibrium(
    const std::deque<double>& virial_ratios,
    int window_size,
    double tolerance) {
    if (static_cast<int>(virial_ratios.size()) < window_size) {
        return false;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = static_cast<int>(virial_ratios.size()) - window_size;
         i < static_cast<int>(virial_ratios.size());
         ++i) {
        sum += virial_ratios[static_cast<std::size_t>(i)];
        sum_sq += virial_ratios[static_cast<std::size_t>(i)] *
                  virial_ratios[static_cast<std::size_t>(i)];
    }
    const double mean = sum / static_cast<double>(window_size);
    const double variance = (sum_sq / static_cast<double>(window_size)) - (mean * mean);
    const double std_dev = std::sqrt(std::max(variance, 0.0));
    return (std::abs(mean - 1.0) < tolerance) && (std_dev < tolerance);
}

void update_progress_bar(PyObject* pbar, int n) {
    if (pbar == nullptr || pbar == Py_None) {
        return;
    }
    PyObject* update_method = PyObject_GetAttrString(pbar, "update");
    if (update_method == nullptr) {
        PyErr_Clear();
        return;
    }
    PyObject* arg = PyLong_FromLong(n);
    PyObject* result = PyObject_CallFunctionObjArgs(update_method, arg, nullptr);
    Py_XDECREF(result);
    Py_DECREF(arg);
    Py_DECREF(update_method);
}

void set_progress_postfix(PyObject* pbar, double time_yr) {
    if (pbar == nullptr || pbar == Py_None) {
        return;
    }
    PyObject* set_postfix = PyObject_GetAttrString(pbar, "set_postfix_str");
    if (set_postfix == nullptr) {
        PyErr_Clear();
        return;
    }
    char time_buffer[64];
    std::snprintf(time_buffer, sizeof(time_buffer), "%.1f yr", time_yr);
    PyObject* time_str = PyUnicode_FromString(time_buffer);
    PyObject* result = PyObject_CallFunctionObjArgs(set_postfix, time_str, nullptr);
    Py_XDECREF(result);
    Py_DECREF(time_str);
    Py_DECREF(set_postfix);
}

PyObject* create_progress_bar(int sim_id, int total_steps) {
    PyObject* tqdm_module = PyImport_ImportModule("tqdm");
    if (tqdm_module == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    PyObject* tqdm_class = PyObject_GetAttrString(tqdm_module, "tqdm");
    Py_DECREF(tqdm_module);
    if (tqdm_class == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    char desc_buffer[64];
    std::snprintf(desc_buffer, sizeof(desc_buffer), "Simulation %d", sim_id);
    PyObject* kwargs = Py_BuildValue(
        "{s:i,s:s,s:s,s:O,s:s}",
        "total", total_steps,
        "desc", desc_buffer,
        "unit", "step",
        "leave", Py_True,
        "bar_format", "{desc}: {n}/{total} steps | {postfix}");
    PyObject* args = PyTuple_New(0);
    if (kwargs == nullptr || args == nullptr) {
        Py_XDECREF(kwargs);
        Py_XDECREF(args);
        Py_DECREF(tqdm_class);
        PyErr_Clear();
        return nullptr;
    }
    PyObject* pbar = PyObject_Call(tqdm_class, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(tqdm_class);
    if (pbar == nullptr) {
        PyErr_Clear();
    }
    return pbar;
}

void close_progress_bar(PyObject* pbar) {
    if (pbar == nullptr || pbar == Py_None) {
        return;
    }
    PyObject* close_method = PyObject_GetAttrString(pbar, "close");
    if (close_method != nullptr) {
        PyObject* result = PyObject_CallObject(close_method, nullptr);
        Py_XDECREF(result);
        Py_DECREF(close_method);
    } else {
        PyErr_Clear();
    }
    Py_DECREF(pbar);
}

PyArrayObject* build_chunk_array(
    int sim_id,
    double time_yr,
    const std::vector<double>& X,
    const std::vector<double>& V,
    const std::vector<double>& KE,
    const std::vector<double>& PE,
    int N) {
    npy_intp dims[2] = {static_cast<npy_intp>(N), OUTPUT_COLUMNS};
    PyArrayObject* chunk_arr =
        reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (chunk_arr == nullptr) {
        return nullptr;
    }
    double* chunk_data = static_cast<double*>(PyArray_DATA(chunk_arr));
    for (int i = 0; i < N; ++i) {
        const int row = i * OUTPUT_COLUMNS;
        chunk_data[row + 0] = static_cast<double>(sim_id);
        chunk_data[row + 1] = time_yr;
        chunk_data[row + 2] = static_cast<double>(i);
        chunk_data[row + 3] = X[i * 3 + 0];
        chunk_data[row + 4] = X[i * 3 + 1];
        chunk_data[row + 5] = X[i * 3 + 2];
        chunk_data[row + 6] = V[i * 3 + 0];
        chunk_data[row + 7] = V[i * 3 + 1];
        chunk_data[row + 8] = V[i * 3 + 2];
        chunk_data[row + 9] = KE[static_cast<std::size_t>(i)];
        chunk_data[row + 10] = PE[static_cast<std::size_t>(i)];
    }
    return chunk_arr;
}

/*
Runs the chunked GPU simulation backend with the same Python-call signature as simulator.cpp.
The inner leapfrog loop stays on the GPU, while chunk-boundary work returns to the CPU.
*/
PyObject* run_simulation_impl([[maybe_unused]] PyObject* self, PyObject* args) {
    PyObject *X0_obj = nullptr, *V0_obj = nullptr, *M_obj = nullptr;
    PyObject *perturb_idx_obj = nullptr, *perturb_pos_obj = nullptr, *perturb_vel_obj = nullptr;
    PyArrayObject *X0_arr = nullptr, *V0_arr = nullptr, *M_arr = nullptr;
    PyArrayObject *perturb_idx_arr = nullptr, *perturb_pos_arr = nullptr, *perturb_vel_arr = nullptr;
    int sim_id = 0, max_step = 0, chunk_size = 0;
    int N = 0;
    double dt = 0.0, yr = 0.0, collision_radius = 0.0;
    DeviceState* ds = nullptr;
    PyObject* all_chunks = nullptr;
    PyObject* pbar = nullptr;
    int completed_steps = 0;
    double t = 0.0;
    bool equilibrium_reached = false;
    std::vector<double> X_cpu;
    std::vector<double> V_cpu;
    std::vector<double> M_cpu;
    std::vector<double> KE_cpu;
    std::vector<double> PE_cpu;
    std::deque<double> virial_ratios;
    const double* X0_data = nullptr;
    const double* V0_data = nullptr;
    const double* M_data = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOiiiddd",
            &X0_obj,
            &V0_obj,
            &M_obj,
            &perturb_idx_obj,
            &perturb_pos_obj,
            &perturb_vel_obj,
            &sim_id,
            &max_step,
            &chunk_size,
            &dt,
            &yr,
            &collision_radius)) {
        return nullptr;
    }

    X0_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(X0_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    V0_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(V0_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    M_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(M_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    perturb_idx_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(perturb_idx_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    perturb_pos_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(perturb_pos_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    perturb_vel_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(perturb_vel_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (X0_arr == nullptr || V0_arr == nullptr || M_arr == nullptr ||
        perturb_idx_arr == nullptr || perturb_pos_arr == nullptr || perturb_vel_arr == nullptr) {
        goto fail;
    }

    if (PyArray_NDIM(X0_arr) != 2 || PyArray_DIM(X0_arr, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "X0 must be an N x 3 float64 array");
        goto fail;
    }
    if (PyArray_NDIM(V0_arr) != 2 || PyArray_DIM(V0_arr, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "V0 must be an N x 3 float64 array");
        goto fail;
    }
    if (PyArray_NDIM(M_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "M must be a 1D float64 array");
        goto fail;
    }

    N = static_cast<int>(PyArray_DIM(X0_arr, 0));
    if (N <= 0) {
        PyErr_SetString(PyExc_ValueError, "run_simulation_gpu requires at least one particle");
        goto fail;
    }
    if (PyArray_DIM(V0_arr, 0) != PyArray_DIM(X0_arr, 0) ||
        PyArray_DIM(M_arr, 0) != PyArray_DIM(X0_arr, 0)) {
        PyErr_SetString(PyExc_ValueError, "X0, V0, and M must describe the same particle count");
        goto fail;
    }
    if (max_step < 0 || chunk_size <= 0 || dt <= 0.0 || yr <= 0.0 || collision_radius <= 0.0) {
        PyErr_SetString(
            PyExc_ValueError,
            "max_step must be non-negative and chunk_size, dt, yr, and collision_radius must be positive");
        goto fail;
    }

    X_cpu.resize(static_cast<std::size_t>(N) * 3);
    V_cpu.resize(static_cast<std::size_t>(N) * 3);
    M_cpu.resize(static_cast<std::size_t>(N));
    KE_cpu.assign(static_cast<std::size_t>(N), 0.0);
    PE_cpu.assign(static_cast<std::size_t>(N), 0.0);

    X0_data = static_cast<const double*>(PyArray_DATA(X0_arr));
    V0_data = static_cast<const double*>(PyArray_DATA(V0_arr));
    M_data = static_cast<const double*>(PyArray_DATA(M_arr));
    std::copy(X0_data, X0_data + static_cast<std::size_t>(N) * 3, X_cpu.begin());
    std::copy(V0_data, V0_data + static_cast<std::size_t>(N) * 3, V_cpu.begin());
    std::copy(M_data, M_data + static_cast<std::size_t>(N), M_cpu.begin());

    ds = gpu_alloc(N);
    if (ds == nullptr || !gpu_upload(ds, X_cpu.data(), V_cpu.data(), M_cpu.data(), N) ||
        !gpu_launch_force(ds, collision_radius)) {
        goto simulation_fail;
    }

    all_chunks = PyList_New(0);
    if (all_chunks == nullptr) {
        goto simulation_fail;
    }
    pbar = create_progress_bar(sim_id, max_step);

    while (completed_steps < max_step) {
        const double current_dt = compute_adaptive_timestep(X_cpu, V_cpu, N, dt, collision_radius);
        const int steps_this_chunk = std::min(chunk_size, max_step - completed_steps);
        for (int s = 0; s < steps_this_chunk; ++s) {
            if (!gpu_launch_halfkick(ds, current_dt) ||
                !gpu_launch_drift(ds, current_dt) ||
                !gpu_launch_force(ds, collision_radius) ||
                !gpu_launch_fullkick(ds, current_dt) ||
                !gpu_launch_ke(ds) ||
                !gpu_launch_pe(ds, collision_radius)) {
                goto simulation_fail;
            }
        }
        if (!gpu_sync() ||
            !gpu_download(ds, X_cpu.data(), V_cpu.data(), KE_cpu.data(), PE_cpu.data(), N)) {
            goto simulation_fail;
        }

        if (handle_collisions_gpu(X_cpu, V_cpu, M_cpu, N, collision_radius)) {
            if (!gpu_upload_positions_velocities(ds, X_cpu.data(), V_cpu.data(), N) ||
                !gpu_launch_force(ds, collision_radius) ||
                !gpu_launch_ke(ds) ||
                !gpu_launch_pe(ds, collision_radius) ||
                !gpu_sync() ||
                !gpu_download(ds, X_cpu.data(), V_cpu.data(), KE_cpu.data(), PE_cpu.data(), N)) {
                goto simulation_fail;
            }
        }

        completed_steps += steps_this_chunk;
        t += current_dt * static_cast<double>(steps_this_chunk);

        PyArrayObject* chunk_arr =
            build_chunk_array(sim_id, t / yr, X_cpu, V_cpu, KE_cpu, PE_cpu, N);
        if (chunk_arr == nullptr) {
            goto simulation_fail;
        }
        if (PyList_Append(all_chunks, reinterpret_cast<PyObject*>(chunk_arr)) != 0) {
            Py_DECREF(chunk_arr);
            goto simulation_fail;
        }
        Py_DECREF(chunk_arr);

        double total_ke = 0.0;
        double total_pe = 0.0;
        for (int i = 0; i < N; ++i) {
            total_ke += KE_cpu[static_cast<std::size_t>(i)];
            total_pe += PE_cpu[static_cast<std::size_t>(i)];
        }
        if (completed_steps % VIRIAL_CHECK_INTERVAL == 0 &&
            (completed_steps >= VIRIAL_MIN_STEPS || t >= VIRIAL_MIN_TIME_YR * yr) &&
            std::abs(total_pe) > 1.0e-30) {
            const double virial_ratio = std::abs(2.0 * total_ke / total_pe);
            virial_ratios.push_back(virial_ratio);
            if (virial_ratios.size() > static_cast<std::size_t>(VIRIAL_WINDOW_SIZE * 2)) {
                virial_ratios.pop_front();
            }
            if (check_virial_equilibrium(virial_ratios, VIRIAL_WINDOW_SIZE, VIRIAL_TOLERANCE)) {
                equilibrium_reached = true;
                break;
            }
        }

        update_progress_bar(pbar, steps_this_chunk);
        set_progress_postfix(pbar, t / yr);
    }

    if (equilibrium_reached && completed_steps < max_step) {
        update_progress_bar(pbar, max_step - completed_steps);
        std::printf("Virial equilibrium reached at step %d (%.2f years)\n", completed_steps, t / yr);
    } else if (!equilibrium_reached) {
        std::printf("Maximum steps reached without achieving virial equilibrium\n");
    }

    close_progress_bar(pbar);
    if (ds != nullptr) {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        ds = nullptr;
    }
    Py_DECREF(X0_arr);
    Py_DECREF(V0_arr);
    Py_DECREF(M_arr);
    Py_DECREF(perturb_idx_arr);
    Py_DECREF(perturb_pos_arr);
    Py_DECREF(perturb_vel_arr);
    return all_chunks;

simulation_fail:
    close_progress_bar(pbar);
    Py_XDECREF(all_chunks);
    if (ds != nullptr) {
        const bool ignored = gpu_free(ds);
        (void)ignored;
    }

fail:
    Py_XDECREF(X0_arr);
    Py_XDECREF(V0_arr);
    Py_XDECREF(M_arr);
    Py_XDECREF(perturb_idx_arr);
    Py_XDECREF(perturb_pos_arr);
    Py_XDECREF(perturb_vel_arr);
    return nullptr;
}

}  // namespace

static PyMethodDef simulator_gpu_methods[] = {
    {"run_simulation_gpu", run_simulation_impl, METH_VARARGS,
     "Run the chunked GPU N-body simulation backend."},
    {"run_simulation", run_simulation_impl, METH_VARARGS,
     "Run the chunked GPU N-body simulation backend."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef simulator_gpu_module = {
    PyModuleDef_HEAD_INIT,
    "simulator_gpu",
    "CUDA-accelerated chunked N-body simulation module.",
    -1,
    simulator_gpu_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit_simulator_gpu(void) {
    import_array();
    return PyModule_Create(&simulator_gpu_module);
}
