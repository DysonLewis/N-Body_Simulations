/*
N-body simulation integration module in C++
Handles the main simulation loop with leapfrog integration
Calls Python accel module for gravitational acceleration calculations
Streaming, returns data in chunks to avoid memory overflow

Adaptive timestep when particles get close
Better collision detection with approach threshold
Distance-dependent damping for tight oscillations
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>
#include <omp.h>
#include <deque>

#define G 6.6743e-8L  // gravitational constant in cm^3 g^-1 s^-2

struct SimulationRecord {
    int simulation;
    double time_yr;
    int body_idx;
    double x, y, z;
    double vx, vy, vz;
    double KE, PE, E_tot;
};

/*
Call the Python accel module's get_accel function to compute accelerations.

Args:
    X_arr: N x 3 numpy array of positions in cm
    M_arr: N length numpy array of masses in grams
    collision_radius: minimum distance for gravitational softening
    
Returns:
    N x 3 numpy array of accelerations in cm/s^2
*/
static PyArrayObject* call_get_accel(PyArrayObject* X_arr, PyArrayObject* M_arr, double collision_radius) {
    PyObject* accel_module = PyImport_ImportModule("accel");
    if (!accel_module) {
        PyErr_SetString(PyExc_ImportError, "Failed to import accel module");
        return nullptr;
    }
    
    PyObject* get_accel_func = PyObject_GetAttrString(accel_module, "get_accel");
    Py_DECREF(accel_module);
    
    if (!get_accel_func) {
        PyErr_SetString(PyExc_AttributeError, "Failed to find get_accel function");
        return nullptr;
    }
    
    PyObject* args = PyTuple_Pack(3, X_arr, M_arr, PyFloat_FromDouble(collision_radius));
    PyObject* result = PyObject_CallObject(get_accel_func, args);
    
    Py_DECREF(args);
    Py_DECREF(get_accel_func);
    
    if (!result) {
        return nullptr;
    }
    
    return (PyArrayObject*)result;
}

/*
Compute adaptive timestep based on particle proximity and relative velocities.
Reduces timestep when particles are close to resolve rapid oscillations.

Args:
    X: positions vector (flattened)
    V: velocities vector (flattened)
    N: number of particles
    base_dt: base timestep in seconds
    collision_radius: collision detection radius
    
Returns:
    adaptive timestep in seconds
*/
static double compute_adaptive_timestep(const std::vector<double>& X, 
                                       const std::vector<double>& V,
                                       int N, double base_dt, 
                                       double collision_radius) {
    double min_dt = base_dt;
    double safety_factor = 5e-4;  // Want timestep << collision timescale
    
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            double dx = X[i*3+0] - X[j*3+0];
            double dy = X[i*3+1] - X[j*3+1];
            double dz = X[i*3+2] - X[j*3+2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // If particles are close, estimate collision timescale
            if (r < 4.0 * collision_radius) {
                double vx = V[i*3+0] - V[j*3+0];
                double vy = V[i*3+1] - V[j*3+1];
                double vz = V[i*3+2] - V[j*3+2];
                double v_rel = std::sqrt(vx*vx + vy*vy + vz*vz);
                
                if (v_rel > 1e-9) {
                    // Timescale to cross collision radius
                    double collision_timescale = collision_radius / v_rel;
                    double required_dt = safety_factor * collision_timescale;
                    min_dt = std::min(min_dt, required_dt);
                }
            }
        }
    }
    
    // Don't let timestep get too small (would take forever)
    // Also don't let it get too large (instability)
    min_dt = std::max(min_dt, base_dt * 1e-5);  // Allow down to x of base_dt
    min_dt = std::min(min_dt, base_dt);
    
    return min_dt;
}

/*
Handle elastic collisions between particles with improved detection.
Detects approaching particles even before collision and applies distance-dependent damping.

Args:
    X: positions vector (flattened)
    V: velocities vector (flattened)
    M: masses vector
    N: number of particles
    collision_radius: minimum distance before collision handling
*/
static void handle_collisions(std::vector<double>& X, std::vector<double>& V, 
                             const std::vector<double>& M, int N, double collision_radius) {
    // Convert collision_radius to long double for precision
    long double coll_rad = (long double)collision_radius;
    
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            long double dx = (long double)X[i*3+0] - (long double)X[j*3+0];
            long double dy = (long double)X[i*3+1] - (long double)X[j*3+1];
            long double dz = (long double)X[i*3+2] - (long double)X[j*3+2];
            long double r = sqrtl(dx*dx + dy*dy + dz*dz);
            
            // Check for approaching particles even if not yet colliding
            // This helps catch fast-moving particles before they tunnel through
            long double approach_threshold = coll_rad * 3.0L;
            
            if (r < approach_threshold && r > 1e-10L) {
                long double nx = dx / r;
                long double ny = dy / r;
                long double nz = dz / r;
                
                long double v_rel_x = (long double)V[i*3+0] - (long double)V[j*3+0];
                long double v_rel_y = (long double)V[i*3+1] - (long double)V[j*3+1];
                long double v_rel_z = (long double)V[i*3+2] - (long double)V[j*3+2];
                
                long double v_rel_n = v_rel_x*nx + v_rel_y*ny + v_rel_z*nz;
                
                // Apply collision if approaching
                if (v_rel_n < 0.0L) {
                    long double m_i = (long double)M[i];
                    long double m_j = (long double)M[j];
                    long double m_total = m_i + m_j;
                    
                    // Distance-dependent damping: more damping when very close
                    // This helps kill tight oscillations without affecting distant encounters
                    long double restitution;
                    if (r < coll_rad) {
                        // Very close: damping to prevent jitter
                        restitution = 0.97L;
                    } else if (r < coll_rad * 1.2L) {
                        // Close: moderate damping
                        restitution = 0.98L;
                    } else {
                        // Approaching but not too close: minimal damping
                        restitution = 0.999L;
                    }
                    
                    long double impulse = 2.0L * m_i * m_j * v_rel_n / m_total * restitution;
                    
                    // Apply impulse to velocities
                    V[i*3+0] -= (double)((impulse / m_i) * nx);
                    V[i*3+1] -= (double)((impulse / m_i) * ny);
                    V[i*3+2] -= (double)((impulse / m_i) * nz);
                    
                    V[j*3+0] += (double)((impulse / m_j) * nx);
                    V[j*3+1] += (double)((impulse / m_j) * ny);
                    V[j*3+2] += (double)((impulse / m_j) * nz);
                }
            }
        }
    }
}

/*
Check if system has reached virial equilibrium using moving average.
Virial theorem: 2*KE + PE ≈ 0 at equilibrium
Returns true if virial ratio is stable near 1.0

Args:
    virial_ratios: deque of recent virial ratio measurements
    window_size: number of measurements to average
    tolerance: acceptable deviation from ideal ratio
*/
static bool check_virial_equilibrium(const std::deque<double>& virial_ratios, 
                                    int window_size, double tolerance) {
    if ((int)virial_ratios.size() < window_size) {
        return false;
    }
    
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = (int)virial_ratios.size() - window_size; i < (int)virial_ratios.size(); i++) {
        sum += virial_ratios[i];
        sum_sq += virial_ratios[i] * virial_ratios[i];
    }
    
    double mean = sum / window_size;
    double variance = (sum_sq / window_size) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    return (std::abs(mean - 1.0) < tolerance) && (std_dev < tolerance);
}

/*
Update progress bar by calling Python's tqdm.
*/
static void update_progress_bar(PyObject* pbar, int n) {
    if (!pbar || pbar == Py_None) return;
    
    PyObject* update_method = PyObject_GetAttrString(pbar, "update");
    if (update_method) {
        PyObject* arg = PyLong_FromLong(n);
        PyObject* result = PyObject_CallFunctionObjArgs(update_method, arg, nullptr);
        Py_XDECREF(result);
        Py_DECREF(arg);
        Py_DECREF(update_method);
    }
}

/*
Run a single N-body simulation using leapfrog integration with adaptive timestep.
Yields results in chunks via Python generator to avoid memory overflow.

Args:
    X0_arr: N x 3 initial positions in cm
    V0_arr: N x 3 initial velocities in cm/s
    M_arr: N masses in grams
    perturb_idx_arr: unused (kept for compatibility)
    perturb_pos_arr: unused (kept for compatibility)
    perturb_vel_arr: unused (kept for compatibility)
    sim_id: simulation identifier
    max_step: maximum number of timesteps
    dt: base timestep in seconds
    yr: conversion factor from seconds to years
    collision_radius: minimum distance for collision detection
    chunk_size: number of timesteps to accumulate before yielding
    
Returns:
    Python list of chunk arrays
*/
static PyObject* run_simulation([[maybe_unused]] PyObject* self, PyObject* args) {
    PyArrayObject* X0_arr = nullptr;
    PyArrayObject* V0_arr = nullptr;
    PyArrayObject* M_arr = nullptr;
    PyArrayObject* perturb_idx_arr = nullptr;
    PyArrayObject* perturb_pos_arr = nullptr;
    PyArrayObject* perturb_vel_arr = nullptr;
    
    int sim_id, max_step, chunk_size;
    double dt, yr, collision_radius;
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!iiiddd",
                          &PyArray_Type, &X0_arr,
                          &PyArray_Type, &V0_arr,
                          &PyArray_Type, &M_arr,
                          &PyArray_Type, &perturb_idx_arr,
                          &PyArray_Type, &perturb_pos_arr,
                          &PyArray_Type, &perturb_vel_arr,
                          &sim_id,
                          &max_step,
                          &chunk_size,
                          &dt,
                          &yr,
                          &collision_radius)) {
        return nullptr;
    }
    
    int N = PyArray_DIM(X0_arr, 0);
    
    // Copy initial conditions to C++ vectors for fast access
    std::vector<double> X(N * 3);  // positions flattened [x0,y0,z0,x1,y1,z1,...]
    std::vector<double> V(N * 3);  // velocities flattened
    std::vector<double> M(N);      // masses
    
    double* X0_data = (double*)PyArray_DATA(X0_arr);
    double* V0_data = (double*)PyArray_DATA(V0_arr);
    double* M_data = (double*)PyArray_DATA(M_arr);
    
    // Parallelize initial data copying
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * 3; i++) {
        X[i] = X0_data[i];
        V[i] = V0_data[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        M[i] = M_data[i];
    }
    
    // Create numpy arrays for passing to accel module
    npy_intp pos_dims[2] = {N, 3};
    npy_intp mass_dims[1] = {N};
    
    PyArrayObject* X_arr = (PyArrayObject*)PyArray_SimpleNew(2, pos_dims, NPY_DOUBLE);
    PyArrayObject* M_arr_np = (PyArrayObject*)PyArray_SimpleNew(1, mass_dims, NPY_DOUBLE);
    
    if (!X_arr || !M_arr_np) {
        Py_XDECREF(X_arr);
        Py_XDECREF(M_arr_np);
        return nullptr;
    }
    
    // Copy masses to numpy array (only need to do once since masses don't change)
    double* M_arr_data = (double*)PyArray_DATA(M_arr_np);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        M_arr_data[i] = M[i];
    }
    
    // Copy initial positions and compute initial acceleration
    double* X_arr_data = (double*)PyArray_DATA(X_arr);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N * 3; i++) {
        X_arr_data[i] = X[i];
    }
    
    PyArrayObject* A_arr = call_get_accel(X_arr, M_arr_np, collision_radius);
    if (!A_arr) {
        Py_DECREF(X_arr);
        Py_DECREF(M_arr_np);
        return nullptr;
    }
    
    double* A = (double*)PyArray_DATA(A_arr);
    
    // Virial equilibrium tracking
    std::deque<double> virial_ratios;
    const int check_interval = 100;     // how often it checks for stability, this does not affect the chance just runtime
    const int window_size = 400;        // how many step it averages over for stability to be true
    const double tolerance = 0.03;      // percent deviation
    const int min_steps = 1e5;          // minimum runtime before checking
    const int min_time = 2000 * yr;     // or this
    bool equilibrium_reached = false;
    
    // Track actual simulation time (for adaptive timestep)
    double t = 0.0;
    
    // Base timestep for adaptive calculation
    double base_dt = dt;

    // Create a Python list to accumulate all chunks
    PyObject* all_chunks = PyList_New(0);
    if (!all_chunks) {
        Py_DECREF(X_arr);
        Py_DECREF(M_arr_np);
        Py_DECREF(A_arr);
        return nullptr;
    }
    
    // Allocate buffer for one chunk
    std::vector<std::vector<double>> chunk_buffer;
    chunk_buffer.reserve(chunk_size * N);
    
    // Create progress bar using tqdm
    PyObject* tqdm_module = PyImport_ImportModule("tqdm");
    PyObject* pbar = nullptr;

    if (tqdm_module) {
        PyObject* tqdm_class = PyObject_GetAttrString(tqdm_module, "tqdm");
        if (tqdm_class) {
            char desc_buffer[64];
            snprintf(desc_buffer, sizeof(desc_buffer), "Simulation %d", sim_id);
            
            PyObject* kwargs = Py_BuildValue("{s:s,s:s,s:O,s:s}", 
                                            "desc", desc_buffer,
                                            "unit", "step",
                                            "leave", Py_True,
                                            "bar_format", "{desc}: {n} steps | {postfix}");
            pbar = PyObject_Call(tqdm_class, PyTuple_New(0), kwargs);
            Py_DECREF(kwargs);
            Py_DECREF(tqdm_class);
        }
        Py_DECREF(tqdm_module);
    }

    // Main leapfrog integration loop
    int step = 0;
    while (t < max_step * base_dt) {
        // Compute adaptive timestep based on current state
        double current_dt = compute_adaptive_timestep(X, V, N, base_dt, collision_radius);
        
        // Release GIL for computationally intensive section
        Py_BEGIN_ALLOW_THREADS
        
        // Leapfrog step 1: V(t+dt/2) = V(t) + A(t)*dt/2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            V[i] += A[i] * current_dt / 2.0;
        }
        
        // Leapfrog step 2: X(t+dt) = X(t) + V(t+dt/2)*dt
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            X[i] += V[i] * current_dt;
        }
        
        // Handle particle collisions
        handle_collisions(X, V, M, N, collision_radius);
        
        // Update positions in numpy array for accel calculation
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            X_arr_data[i] = X[i];
        }
        
        // Reacquire GIL before calling Python function
        Py_END_ALLOW_THREADS
        
        // Compute A(t+dt) at new positions
        Py_DECREF(A_arr);
        A_arr = call_get_accel(X_arr, M_arr_np, collision_radius);
        if (!A_arr) {
            Py_DECREF(X_arr);
            Py_DECREF(M_arr_np);
            Py_DECREF(all_chunks);
            Py_XDECREF(pbar);
            return nullptr;
        }
        A = (double*)PyArray_DATA(A_arr);
        
        // Release GIL again for computation
        Py_BEGIN_ALLOW_THREADS
        
        // Leapfrog step 3: V(t+dt) = V(t+dt/2) + A(t+dt)*dt/2
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * 3; i++) {
            V[i] += A[i] * current_dt / 2.0;
        }
        
        // Update time (use adaptive dt)
        t += current_dt;
        step++;
        
        // Reacquire GIL before modifying Python objects
        Py_END_ALLOW_THREADS
        
        // Compute kinetic energy for each particle: KE = 0.5*m*v^2
        std::vector<double> KE(N);
        std::vector<double> PE(N, 0.0);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double v2 = V[i*3+0]*V[i*3+0] + V[i*3+1]*V[i*3+1] + V[i*3+2]*V[i*3+2];
            KE[i] = 0.5 * M[i] * v2;
        }
        
        // Compute potential energy for each particle
        // PE_ij = -G*m_i*m_j / r_ij
        // Each particle gets half of each pair interaction to avoid double counting when summing
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                long double dx = (long double)X[i*3+0] - (long double)X[j*3+0];
                long double dy = (long double)X[i*3+1] - (long double)X[j*3+1];
                long double dz = (long double)X[i*3+2] - (long double)X[j*3+2];
                long double r = sqrtl(dx*dx + dy*dy + dz*dz);
                
                // Apply softening (same as in acceleration calculation)
                long double r_soft = (r < (long double)collision_radius) ? (long double)collision_radius : r;
                
                long double pe_term = -G * (long double)M[i] * (long double)M[j] / r_soft;
                
                // Convert back to double for atomic operations
                double pe_contrib = (double)(pe_term * 0.5L);
                #pragma omp atomic
                PE[i] += pe_contrib;
                #pragma omp atomic
                PE[j] += pe_contrib;
            }
        }
        
        // Store results for this timestep in chunk buffer
        for (int i = 0; i < N; i++) {
            std::vector<double> row(11);
            row[0] = sim_id;
            row[1] = t / yr;
            row[2] = i;
            row[3] = X[i*3 + 0];
            row[4] = X[i*3 + 1];
            row[5] = X[i*3 + 2];
            row[6] = V[i*3 + 0];
            row[7] = V[i*3 + 1];
            row[8] = V[i*3 + 2];
            row[9] = KE[i];
            row[10] = PE[i];
            chunk_buffer.push_back(row);
        }
        
        // When chunk is full or we're done, yield it
        if ((int)chunk_buffer.size() >= chunk_size * N || t >= max_step * base_dt) {
            // Create numpy array for this chunk
            npy_intp chunk_dims[2] = {(npy_intp)chunk_buffer.size(), 11};
            PyArrayObject* chunk_arr = (PyArrayObject*)PyArray_ZEROS(2, chunk_dims, NPY_DOUBLE, 0);
            if (!chunk_arr) {
                Py_DECREF(X_arr);
                Py_DECREF(M_arr_np);
                Py_DECREF(A_arr);
                Py_DECREF(all_chunks);
                Py_XDECREF(pbar);
                return nullptr;
            }
            
            double* chunk_data = (double*)PyArray_DATA(chunk_arr);
            for (size_t i = 0; i < chunk_buffer.size(); i++) {
                for (int j = 0; j < 11; j++) {
                    chunk_data[i*11 + j] = chunk_buffer[i][j];
                }
            }
            
            // Add chunk to list
            PyList_Append(all_chunks, (PyObject*)chunk_arr);
            Py_DECREF(chunk_arr);
            
            // Clear chunk buffer
            chunk_buffer.clear();
        }
        
        // Check virial equilibrium every check_interval steps
        if (step % check_interval == 0 && (step >= min_steps || t >= min_time)) {
            double total_KE = 0.0;
            double total_PE = 0.0;
            
            for (int i = 0; i < N; i++) {
                total_KE += KE[i];
                total_PE += PE[i];
            }
            
            if (std::abs(total_PE) > 1e-30) {
                double virial_ratio = std::abs(2.0 * total_KE / total_PE);
                virial_ratios.push_back(virial_ratio);
                
                if (virial_ratios.size() > (size_t)(window_size * 2)) {
                    virial_ratios.pop_front();
                }
                
                if (check_virial_equilibrium(virial_ratios, window_size, tolerance)) {
                    equilibrium_reached = true;
                    
                    // Update progress bar to completion
                    if (pbar) {
                        int remaining_steps = max_step - (int)(t / base_dt);
                        update_progress_bar(pbar, remaining_steps);
                    }
                    
                    printf("Virial equilibrium reached at step %d (%.2f years)\n", 
                           step, t / yr);
                    break;
                }
            }
        }
        
        // Update progress bar
        if (pbar && step % 100 == 0) {
            update_progress_bar(pbar, 100);
            
            // Update postfix with current simulation time
            PyObject* set_postfix = PyObject_GetAttrString(pbar, "set_postfix_str");
            if (set_postfix) {
                char time_buffer[64];
                snprintf(time_buffer, sizeof(time_buffer), "%.1f yr", t / yr);
                PyObject* time_str = PyUnicode_FromString(time_buffer);
                PyObject* result = PyObject_CallFunctionObjArgs(set_postfix, time_str, nullptr);
                Py_XDECREF(result);
                Py_DECREF(time_str);
                Py_DECREF(set_postfix);
            }
        }
    }
    
    if (!equilibrium_reached) {
        printf("Maximum steps reached without achieving virial equilibrium\n");
    }
    
    // Close progress bar
    if (pbar) {
        PyObject* close_method = PyObject_GetAttrString(pbar, "close");
        if (close_method) {
            PyObject* result = PyObject_CallObject(close_method, nullptr);
            Py_XDECREF(result);
            Py_DECREF(close_method);
        }
        Py_DECREF(pbar);
    }
    
    // Clean up numpy arrays
    Py_DECREF(X_arr);
    Py_DECREF(M_arr_np);
    Py_DECREF(A_arr);
    
    return all_chunks;
}

// Python module method definitions
static PyMethodDef simulator_methods[] = {
    {"run_simulation", run_simulation, METH_VARARGS, 
     "Run a single N-body simulation"},
    {nullptr, nullptr, 0, nullptr}
};

// Python module definition
static struct PyModuleDef simulator_module = {
    PyModuleDef_HEAD_INIT,
    "simulator",
    "N-body simulation integration module",
    -1,
    simulator_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

// Module initialization function called by Python
PyMODINIT_FUNC PyInit_simulator(void) {
    import_array();  // required for numpy C API
    return PyModule_Create(&simulator_module);
}