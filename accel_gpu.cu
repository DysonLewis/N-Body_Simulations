#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "accel_gpu_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <vector>

#define G_GPU 6.67430e-8
#define TILE_SIZE 256

#define GPU_ZERO 0.0  // exact additive identity used in device and host-side staging code.
#define GPU_ONE 1.0   // exact multiplicative identity used in inverse-distance terms.
#define GPU_HALF 0.5  // exact half-step / half-pair factor used by leapfrog and energy code.

__device__ __forceinline__ double4_16a make_tile_entry(double x, double y, double z, double w) {
    return double4_16a{x, y, z, w};
}

#define CUDA_CHECK(call, on_error)                                                        \
    do {                                                                                  \
        const cudaError_t cuda_check_err__ = (call);                                      \
        if (cuda_check_err__ != cudaSuccess) {                                            \
            if (!PyErr_Occurred()) {                                                      \
                PyErr_Format(                                                             \
                    PyExc_RuntimeError,                                                   \
                    "CUDA error %s in %s at %s:%d",                                       \
                    cudaGetErrorString(cuda_check_err__),                                 \
                    #call,                                                                \
                    __FILE__,                                                             \
                    __LINE__);                                                            \
            }                                                                             \
            on_error;                                                                     \
        }                                                                                 \
    } while (0)

/*
Owns all device-resident SoA buffers for one simulation or one get_accel call.
All pointers are either null or point to cudaMalloc-managed storage sized to capacity.
*/
struct DeviceState {
    double* d_x;
    double* d_y;
    double* d_z;
    double* d_vx;
    double* d_vy;
    double* d_vz;
    double* d_m;
    double* d_ax;
    double* d_ay;
    double* d_az;
    double* d_ke;
    double* d_pe;
    int N;
    int capacity;
};

/*
Computes gravitational acceleration for each particle using a shared-memory tiled
O(N^2) interaction loop. Assumes positions and masses are valid SoA arrays and
overwrites d_ax/d_ay/d_az with the current acceleration field.
*/
__global__ void force_kernel(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const double* __restrict__ d_z,
    const double* __restrict__ d_m,
    double* d_ax,
    double* d_ay,
    double* d_az,
    int N,
    double collision_radius) {
    extern __shared__ double4_16a sh_tile[];

    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool active_particle = (i < N);

    double xi = GPU_ZERO;
    double yi = GPU_ZERO;
    double zi = GPU_ZERO;
    if (active_particle) {
        xi = d_x[i];
        yi = d_y[i];
        zi = d_z[i];
    }

    double ax = GPU_ZERO;
    double ay = GPU_ZERO;
    double az = GPU_ZERO;
    double cx = GPU_ZERO;
    double cy = GPU_ZERO;
    double cz = GPU_ZERO;

    for (int tile_base = 0; tile_base < N; tile_base += TILE_SIZE) {
        const int j_global = tile_base + static_cast<int>(threadIdx.x);
        if (j_global < N) {
            sh_tile[threadIdx.x] = make_tile_entry(
                d_x[j_global], d_y[j_global], d_z[j_global], d_m[j_global]);
        } else {
            sh_tile[threadIdx.x] = make_tile_entry(GPU_ZERO, GPU_ZERO, GPU_ZERO, GPU_ZERO);
        }

        // All threads must finish writing the current tile before any thread reads it.
        __syncthreads();

        if (active_particle) {
            const int tile_count = ((tile_base + TILE_SIZE) < N) ? TILE_SIZE : (N - tile_base);
            for (int k = 0; k < tile_count; ++k) {
                const int j_particle = tile_base + k;
                if (j_particle == i) {
                    continue;
                }

                const double dx = xi - sh_tile[k].x;
                const double dy = yi - sh_tile[k].y;
                const double dz = zi - sh_tile[k].z;
                const double mj = sh_tile[k].w;

                const double r2 = dx * dx + dy * dy + dz * dz;
                const double r = sqrt(r2);
                const double r_soft = (r < collision_radius) ? collision_radius : r;
                const double r3 = r_soft * r_soft * r_soft;
                const double factor = G_GPU * mj * (GPU_ONE / r3);

                const double term_x = (-factor * dx) - cx;
                const double next_ax = ax + term_x;
                cx = (next_ax - ax) - term_x;
                ax = next_ax;

                const double term_y = (-factor * dy) - cy;
                const double next_ay = ay + term_y;
                cy = (next_ay - ay) - term_y;
                ay = next_ay;

                const double term_z = (-factor * dz) - cz;
                const double next_az = az + term_z;
                cz = (next_az - az) - term_z;
                az = next_az;
            }
        }

        // No thread may overwrite shared memory with the next tile until all reads are done.
        __syncthreads();
    }

    if (active_particle) {
        d_ax[i] = ax;
        d_ay[i] = ay;
        d_az[i] = az;
    }
}

/*
Applies the leapfrog half-kick update V(t + dt/2) = V(t) + A(t) * dt/2.
Assumes velocity and acceleration SoA arrays are valid and updates velocities in place.
*/
__global__ void leapfrog_halfkick_kernel(
    double* d_vx,
    double* d_vy,
    double* d_vz,
    const double* __restrict__ d_ax,
    const double* __restrict__ d_ay,
    const double* __restrict__ d_az,
    int N,
    double half_dt) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) {
        return;
    }

    d_vx[i] += d_ax[i] * half_dt;
    d_vy[i] += d_ay[i] * half_dt;
    d_vz[i] += d_az[i] * half_dt;
}

/*
Applies the leapfrog drift update X(t + dt) = X(t) + V(t + dt/2) * dt.
Assumes position and velocity SoA arrays are valid and updates positions in place.
*/
__global__ void leapfrog_drift_kernel(
    double* d_x,
    double* d_y,
    double* d_z,
    const double* __restrict__ d_vx,
    const double* __restrict__ d_vy,
    const double* __restrict__ d_vz,
    int N,
    double dt) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) {
        return;
    }

    d_x[i] += d_vx[i] * dt;
    d_y[i] += d_vy[i] * dt;
    d_z[i] += d_vz[i] * dt;
}

/*
Applies the leapfrog full-kick update V(t + dt) = V(t + dt/2) + A(t + dt) * dt/2.
Assumes velocity and acceleration SoA arrays are valid and updates velocities in place.
*/
__global__ void leapfrog_fullkick_kernel(
    double* d_vx,
    double* d_vy,
    double* d_vz,
    const double* __restrict__ d_ax,
    const double* __restrict__ d_ay,
    const double* __restrict__ d_az,
    int N,
    double half_dt) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) {
        return;
    }

    d_vx[i] += d_ax[i] * half_dt;
    d_vy[i] += d_ay[i] * half_dt;
    d_vz[i] += d_az[i] * half_dt;
}

/*
Computes per-particle kinetic energy from the current velocity field.
Assumes masses and velocities are valid SoA arrays and overwrites d_ke.
*/
__global__ void ke_kernel(
    const double* __restrict__ d_vx,
    const double* __restrict__ d_vy,
    const double* __restrict__ d_vz,
    const double* __restrict__ d_m,
    double* d_ke,
    int N) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N) {
        return;
    }

    const double v2 = d_vx[i] * d_vx[i] + d_vy[i] * d_vy[i] + d_vz[i] * d_vz[i];
    d_ke[i] = GPU_HALF * d_m[i] * v2;
}

/*
Computes per-particle potential energy using the same tiled interaction pattern as the
force kernel. Assumes d_pe has been zeroed immediately before launch and splits each
pair contribution evenly between the two participating particles.
*/
__global__ void pe_kernel(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const double* __restrict__ d_z,
    const double* __restrict__ d_m,
    double* d_pe,
    int N,
    double collision_radius) {
    extern __shared__ double4_16a sh_tile[];

    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool active_particle = (i < N);

    double xi = GPU_ZERO;
    double yi = GPU_ZERO;
    double zi = GPU_ZERO;
    double mi = GPU_ZERO;
    if (active_particle) {
        xi = d_x[i];
        yi = d_y[i];
        zi = d_z[i];
        mi = d_m[i];
    }

    for (int tile_base = 0; tile_base < N; tile_base += TILE_SIZE) {
        const int j_global = tile_base + static_cast<int>(threadIdx.x);
        if (j_global < N) {
            sh_tile[threadIdx.x] = make_tile_entry(
                d_x[j_global], d_y[j_global], d_z[j_global], d_m[j_global]);
        } else {
            sh_tile[threadIdx.x] = make_tile_entry(GPU_ZERO, GPU_ZERO, GPU_ZERO, GPU_ZERO);
        }

        // All threads must finish writing the current tile before any thread reads it.
        __syncthreads();

        if (active_particle) {
            const int tile_count = ((tile_base + TILE_SIZE) < N) ? TILE_SIZE : (N - tile_base);
            for (int k = 0; k < tile_count; ++k) {
                const int j_particle = tile_base + k;
                if (j_particle <= i) {
                    continue;
                }

                const double dx = xi - sh_tile[k].x;
                const double dy = yi - sh_tile[k].y;
                const double dz = zi - sh_tile[k].z;
                const double mj = sh_tile[k].w;

                const double r2 = dx * dx + dy * dy + dz * dz;
                const double r = sqrt(r2);
                const double r_soft = (r < collision_radius) ? collision_radius : r;
                const double pair_energy = -G_GPU * mi * mj / r_soft;
                const double half_pair_energy = GPU_HALF * pair_energy;

                atomicAdd(&d_pe[i], half_pair_energy);
                atomicAdd(&d_pe[j_particle], half_pair_energy);
            }
        }

        // No thread may overwrite shared memory with the next tile until all reads are done.
        __syncthreads();
    }
}

static bool validate_capacity(const DeviceState* ds, int N, const char* function_name) {
    if (ds == nullptr) {
        PyErr_Format(PyExc_ValueError, "%s requires a non-null DeviceState", function_name);
        return false;
    }
    if (N < 0) {
        PyErr_Format(PyExc_ValueError, "%s requires N >= 0", function_name);
        return false;
    }
    if (N > ds->capacity) {
        PyErr_Format(
            PyExc_ValueError,
            "%s requested N=%d but DeviceState capacity is %d",
            function_name,
            N,
            ds->capacity);
        return false;
    }
    return true;
}

/*
Allocates a DeviceState and all associated SoA buffers for N particles.
Assumes N > 0 and leaves all device pointers ready for upload and kernel launches.
*/
extern "C" DeviceState* gpu_alloc(int N) {
    if (N <= 0) {
        PyErr_SetString(PyExc_ValueError, "gpu_alloc requires N > 0");
        return nullptr;
    }

    DeviceState* ds = new DeviceState{
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr,
        nullptr, nullptr, nullptr,
        nullptr, nullptr,
        N, N};

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_x), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_y), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_z), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_vx), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_vy), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_vz), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_m), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_ax), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_ay), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_az), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_ke), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ds->d_pe), bytes), {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return nullptr;
    });

    return ds;
}

/*
Frees all cudaMalloc-managed buffers owned by DeviceState and deletes the struct itself.
Assumes ds is either null or was returned by gpu_alloc and leaves no device storage behind.
*/
extern "C" bool gpu_free(DeviceState* ds) {
    if (ds == nullptr) {
        return true;
    }

    if (ds->d_x != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_x), { return false; });
    }
    if (ds->d_y != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_y), { return false; });
    }
    if (ds->d_z != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_z), { return false; });
    }
    if (ds->d_vx != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_vx), { return false; });
    }
    if (ds->d_vy != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_vy), { return false; });
    }
    if (ds->d_vz != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_vz), { return false; });
    }
    if (ds->d_m != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_m), { return false; });
    }
    if (ds->d_ax != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_ax), { return false; });
    }
    if (ds->d_ay != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_ay), { return false; });
    }
    if (ds->d_az != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_az), { return false; });
    }
    if (ds->d_ke != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_ke), { return false; });
    }
    if (ds->d_pe != nullptr) {
        CUDA_CHECK(cudaFree(ds->d_pe), { return false; });
    }

    delete ds;
    return true;
}

/*
Uploads host AoS position and velocity arrays plus host mass array into device SoA buffers.
Assumes all host pointers are valid for N particles and performs the only host AoS->SoA conversion.
*/
extern "C" bool gpu_upload(
    DeviceState* ds,
    const double* X_aos,
    const double* V_aos,
    const double* M,
    int N) {
    if (!validate_capacity(ds, N, "gpu_upload")) {
        return false;
    }
    if (X_aos == nullptr || V_aos == nullptr || M == nullptr) {
        PyErr_SetString(PyExc_ValueError, "gpu_upload requires non-null host buffers");
        return false;
    }
    if (N == 0) {
        ds->N = 0;
        return true;
    }

    std::vector<double> hx(static_cast<std::size_t>(N));
    std::vector<double> hy(static_cast<std::size_t>(N));
    std::vector<double> hz(static_cast<std::size_t>(N));
    std::vector<double> hvx(static_cast<std::size_t>(N));
    std::vector<double> hvy(static_cast<std::size_t>(N));
    std::vector<double> hvz(static_cast<std::size_t>(N));

    for (int i = 0; i < N; ++i) {
        hx[static_cast<std::size_t>(i)] = X_aos[i * 3 + 0];
        hy[static_cast<std::size_t>(i)] = X_aos[i * 3 + 1];
        hz[static_cast<std::size_t>(i)] = X_aos[i * 3 + 2];

        hvx[static_cast<std::size_t>(i)] = V_aos[i * 3 + 0];
        hvy[static_cast<std::size_t>(i)] = V_aos[i * 3 + 1];
        hvz[static_cast<std::size_t>(i)] = V_aos[i * 3 + 2];
    }

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(ds->d_x, hx.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_y, hy.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_z, hz.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vx, hvx.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vy, hvy.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vz, hvz.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_m, M, bytes, cudaMemcpyHostToDevice), { return false; });

    ds->N = N;
    return true;
}

/*
Uploads host AoS positions and host masses into device SoA buffers while zeroing velocities.
Assumes host buffers are valid for N particles and is intended for standalone force evaluation.
*/
static bool gpu_upload_positions(DeviceState* ds, const double* X_aos, const double* M, int N) {
    if (!validate_capacity(ds, N, "gpu_upload_positions")) {
        return false;
    }
    if (X_aos == nullptr || M == nullptr) {
        PyErr_SetString(PyExc_ValueError, "gpu_upload_positions requires non-null host buffers");
        return false;
    }
    if (N == 0) {
        ds->N = 0;
        return true;
    }

    std::vector<double> hx(static_cast<std::size_t>(N));
    std::vector<double> hy(static_cast<std::size_t>(N));
    std::vector<double> hz(static_cast<std::size_t>(N));
    std::vector<double> zeros(static_cast<std::size_t>(N), GPU_ZERO);

    for (int i = 0; i < N; ++i) {
        hx[static_cast<std::size_t>(i)] = X_aos[i * 3 + 0];
        hy[static_cast<std::size_t>(i)] = X_aos[i * 3 + 1];
        hz[static_cast<std::size_t>(i)] = X_aos[i * 3 + 2];
    }

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(ds->d_x, hx.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_y, hy.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_z, hz.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vx, zeros.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vy, zeros.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vz, zeros.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_m, M, bytes, cudaMemcpyHostToDevice), { return false; });

    ds->N = N;
    return true;
}

/*
Uploads corrected host AoS positions and velocities back into the existing device SoA buffers.
Assumes masses are already resident on the device and only the mutable state needs refreshing.
*/
extern "C" bool gpu_upload_positions_velocities(
    DeviceState* ds,
    const double* X_aos,
    const double* V_aos,
    int N) {
    if (!validate_capacity(ds, N, "gpu_upload_positions_velocities")) {
        return false;
    }
    if (X_aos == nullptr || V_aos == nullptr) {
        PyErr_SetString(
            PyExc_ValueError,
            "gpu_upload_positions_velocities requires non-null host buffers");
        return false;
    }
    if (N == 0) {
        ds->N = 0;
        return true;
    }

    std::vector<double> hx(static_cast<std::size_t>(N));
    std::vector<double> hy(static_cast<std::size_t>(N));
    std::vector<double> hz(static_cast<std::size_t>(N));
    std::vector<double> hvx(static_cast<std::size_t>(N));
    std::vector<double> hvy(static_cast<std::size_t>(N));
    std::vector<double> hvz(static_cast<std::size_t>(N));

    for (int i = 0; i < N; ++i) {
        hx[static_cast<std::size_t>(i)] = X_aos[i * 3 + 0];
        hy[static_cast<std::size_t>(i)] = X_aos[i * 3 + 1];
        hz[static_cast<std::size_t>(i)] = X_aos[i * 3 + 2];

        hvx[static_cast<std::size_t>(i)] = V_aos[i * 3 + 0];
        hvy[static_cast<std::size_t>(i)] = V_aos[i * 3 + 1];
        hvz[static_cast<std::size_t>(i)] = V_aos[i * 3 + 2];
    }

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(ds->d_x, hx.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_y, hy.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_z, hz.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vx, hvx.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vy, hvy.data(), bytes, cudaMemcpyHostToDevice), { return false; });
    CUDA_CHECK(cudaMemcpy(ds->d_vz, hvz.data(), bytes, cudaMemcpyHostToDevice), { return false; });

    ds->N = N;
    return true;
}

/*
Downloads device SoA positions, velocities, and energies back into host AoS/vector buffers.
Assumes host buffers are valid for N particles and performs the only device SoA->AoS conversion.
*/
extern "C" bool gpu_download(
    DeviceState* ds,
    double* X_aos,
    double* V_aos,
    double* KE,
    double* PE,
    int N) {
    if (!validate_capacity(ds, N, "gpu_download")) {
        return false;
    }
    if (X_aos == nullptr || V_aos == nullptr || KE == nullptr || PE == nullptr) {
        PyErr_SetString(PyExc_ValueError, "gpu_download requires non-null host buffers");
        return false;
    }
    if (N == 0) {
        return true;
    }

    std::vector<double> hx(static_cast<std::size_t>(N));
    std::vector<double> hy(static_cast<std::size_t>(N));
    std::vector<double> hz(static_cast<std::size_t>(N));
    std::vector<double> hvx(static_cast<std::size_t>(N));
    std::vector<double> hvy(static_cast<std::size_t>(N));
    std::vector<double> hvz(static_cast<std::size_t>(N));

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(hx.data(), ds->d_x, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(hy.data(), ds->d_y, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(hz.data(), ds->d_z, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(hvx.data(), ds->d_vx, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(hvy.data(), ds->d_vy, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(hvz.data(), ds->d_vz, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(KE, ds->d_ke, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(PE, ds->d_pe, bytes, cudaMemcpyDeviceToHost), { return false; });

    for (int i = 0; i < N; ++i) {
        X_aos[i * 3 + 0] = hx[static_cast<std::size_t>(i)];
        X_aos[i * 3 + 1] = hy[static_cast<std::size_t>(i)];
        X_aos[i * 3 + 2] = hz[static_cast<std::size_t>(i)];

        V_aos[i * 3 + 0] = hvx[static_cast<std::size_t>(i)];
        V_aos[i * 3 + 1] = hvy[static_cast<std::size_t>(i)];
        V_aos[i * 3 + 2] = hvz[static_cast<std::size_t>(i)];
    }

    return true;
}

/*
Downloads device SoA accelerations back into a host AoS acceleration buffer.
Assumes A_aos is valid for N particles and performs the device SoA->AoS conversion for force output.
*/
extern "C" bool gpu_download_accel(DeviceState* ds, double* A_aos, int N) {
    if (!validate_capacity(ds, N, "gpu_download_accel")) {
        return false;
    }
    if (A_aos == nullptr) {
        PyErr_SetString(PyExc_ValueError, "gpu_download_accel requires a non-null host buffer");
        return false;
    }
    if (N == 0) {
        return true;
    }

    std::vector<double> hax(static_cast<std::size_t>(N));
    std::vector<double> hay(static_cast<std::size_t>(N));
    std::vector<double> haz(static_cast<std::size_t>(N));
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);

    CUDA_CHECK(cudaMemcpy(hax.data(), ds->d_ax, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(hay.data(), ds->d_ay, bytes, cudaMemcpyDeviceToHost), { return false; });
    CUDA_CHECK(cudaMemcpy(haz.data(), ds->d_az, bytes, cudaMemcpyDeviceToHost), { return false; });

    for (int i = 0; i < N; ++i) {
        A_aos[i * 3 + 0] = hax[static_cast<std::size_t>(i)];
        A_aos[i * 3 + 1] = hay[static_cast<std::size_t>(i)];
        A_aos[i * 3 + 2] = haz[static_cast<std::size_t>(i)];
    }

    return true;
}

/*
Launches the force kernel on the current DeviceState without synchronizing the stream.
Assumes positions and masses are current on the device and leaves d_ax/d_ay/d_az updated.
*/
extern "C" bool gpu_launch_force(DeviceState* ds, double collision_radius) {
    if (!validate_capacity(ds, ds != nullptr ? ds->N : 0, "gpu_launch_force")) {
        return false;
    }
    if (ds->N == 0) {
        return true;
    }

    const int block_size = TILE_SIZE;  // block = 256 threads, matching the fixed tile width.
    const int grid_size = (ds->N + TILE_SIZE - 1) / TILE_SIZE;  // grid = ceil(N / 256).
    const std::size_t shared_bytes =
        static_cast<std::size_t>(TILE_SIZE) * sizeof(double4_16a);  // one shared tile per block.

    force_kernel<<<grid_size, block_size, shared_bytes>>>(
        ds->d_x, ds->d_y, ds->d_z, ds->d_m,
        ds->d_ax, ds->d_ay, ds->d_az,
        ds->N, collision_radius);
    CUDA_CHECK(cudaGetLastError(), { return false; });
    return true;
}

/*
Launches the leapfrog half-kick kernel on the current DeviceState without synchronizing.
Assumes d_ax/d_ay/d_az already contain A(t) and updates velocities to the half step.
*/
extern "C" bool gpu_launch_halfkick(DeviceState* ds, double dt) {
    if (!validate_capacity(ds, ds != nullptr ? ds->N : 0, "gpu_launch_halfkick")) {
        return false;
    }
    if (ds->N == 0) {
        return true;
    }

    const int block_size = TILE_SIZE;  // block = 256 threads for the bandwidth-bound vector update.
    const int grid_size = (ds->N + TILE_SIZE - 1) / TILE_SIZE;  // grid = ceil(N / 256).
    leapfrog_halfkick_kernel<<<grid_size, block_size>>>(
        ds->d_vx, ds->d_vy, ds->d_vz,
        ds->d_ax, ds->d_ay, ds->d_az,
        ds->N, dt * GPU_HALF);
    CUDA_CHECK(cudaGetLastError(), { return false; });
    return true;
}

/*
Launches the leapfrog drift kernel on the current DeviceState without synchronizing.
Assumes velocities are already at the half step and updates positions in place.
*/
extern "C" bool gpu_launch_drift(DeviceState* ds, double dt) {
    if (!validate_capacity(ds, ds != nullptr ? ds->N : 0, "gpu_launch_drift")) {
        return false;
    }
    if (ds->N == 0) {
        return true;
    }

    const int block_size = TILE_SIZE;  // block = 256 threads for the bandwidth-bound vector update.
    const int grid_size = (ds->N + TILE_SIZE - 1) / TILE_SIZE;  // grid = ceil(N / 256).
    leapfrog_drift_kernel<<<grid_size, block_size>>>(
        ds->d_x, ds->d_y, ds->d_z,
        ds->d_vx, ds->d_vy, ds->d_vz,
        ds->N, dt);
    CUDA_CHECK(cudaGetLastError(), { return false; });
    return true;
}

/*
Launches the leapfrog full-kick kernel on the current DeviceState without synchronizing.
Assumes d_ax/d_ay/d_az already contain A(t + dt) and updates velocities to the full step.
*/
extern "C" bool gpu_launch_fullkick(DeviceState* ds, double dt) {
    if (!validate_capacity(ds, ds != nullptr ? ds->N : 0, "gpu_launch_fullkick")) {
        return false;
    }
    if (ds->N == 0) {
        return true;
    }

    const int block_size = TILE_SIZE;  // block = 256 threads for the bandwidth-bound vector update.
    const int grid_size = (ds->N + TILE_SIZE - 1) / TILE_SIZE;  // grid = ceil(N / 256).
    leapfrog_fullkick_kernel<<<grid_size, block_size>>>(
        ds->d_vx, ds->d_vy, ds->d_vz,
        ds->d_ax, ds->d_ay, ds->d_az,
        ds->N, dt * GPU_HALF);
    CUDA_CHECK(cudaGetLastError(), { return false; });
    return true;
}

/*
Launches the per-particle kinetic energy kernel on the current DeviceState without synchronizing.
Assumes masses and velocities are current on the device and overwrites d_ke.
*/
extern "C" bool gpu_launch_ke(DeviceState* ds) {
    if (!validate_capacity(ds, ds != nullptr ? ds->N : 0, "gpu_launch_ke")) {
        return false;
    }
    if (ds->N == 0) {
        return true;
    }

    const int block_size = TILE_SIZE;  // block = 256 threads for the bandwidth-bound vector update.
    const int grid_size = (ds->N + TILE_SIZE - 1) / TILE_SIZE;  // grid = ceil(N / 256).
    ke_kernel<<<grid_size, block_size>>>(
        ds->d_vx, ds->d_vy, ds->d_vz, ds->d_m, ds->d_ke, ds->N);
    CUDA_CHECK(cudaGetLastError(), { return false; });
    return true;
}

/*
Zeroes d_pe and launches the tiled potential energy kernel without synchronizing.
Assumes positions and masses are current on the device and leaves d_pe holding fresh PE values.
*/
extern "C" bool gpu_launch_pe(DeviceState* ds, double collision_radius) {
    if (!validate_capacity(ds, ds != nullptr ? ds->N : 0, "gpu_launch_pe")) {
        return false;
    }
    if (ds->N == 0) {
        return true;
    }

    const std::size_t bytes = static_cast<std::size_t>(ds->N) * sizeof(double);
    CUDA_CHECK(cudaMemset(ds->d_pe, 0, bytes), { return false; });

    const int block_size = TILE_SIZE;  // block = 256 threads, matching the fixed tile width.
    const int grid_size = (ds->N + TILE_SIZE - 1) / TILE_SIZE;  // grid = ceil(N / 256).
    const std::size_t shared_bytes =
        static_cast<std::size_t>(TILE_SIZE) * sizeof(double4_16a);  // one shared tile per block.

    pe_kernel<<<grid_size, block_size, shared_bytes>>>(
        ds->d_x, ds->d_y, ds->d_z, ds->d_m, ds->d_pe, ds->N, collision_radius);
    CUDA_CHECK(cudaGetLastError(), { return false; });
    return true;
}

/*
Synchronizes the default CUDA stream so the host can safely consume the current chunk state.
Assumes all prior kernels were launched on the default stream and blocks until they finish.
*/
extern "C" bool gpu_sync(void) {
    CUDA_CHECK(cudaDeviceSynchronize(), { return false; });
    return true;
}

/*
Runs a complete standalone force evaluation from host AoS inputs to host AoS output.
Assumes R_aos and M describe N particles and leaves A_aos filled with the force result.
*/
extern "C" bool gpu_get_accel(
    const double* R_aos,
    const double* M,
    double* A_aos,
    int N,
    double collision_radius) {
    if (N < 0) {
        PyErr_SetString(PyExc_ValueError, "gpu_get_accel requires N >= 0");
        return false;
    }
    if (R_aos == nullptr || M == nullptr || A_aos == nullptr) {
        PyErr_SetString(PyExc_ValueError, "gpu_get_accel requires non-null host buffers");
        return false;
    }
    if (N == 0) {
        return true;
    }

    DeviceState* ds = gpu_alloc(N);
    if (ds == nullptr) {
        return false;
    }

    if (!gpu_upload_positions(ds, R_aos, M, N)) {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return false;
    }
    if (!gpu_launch_force(ds, collision_radius)) {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return false;
    }
    if (!gpu_sync()) {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return false;
    }
    if (!gpu_download_accel(ds, A_aos, N)) {
        const bool ignored = gpu_free(ds);
        (void)ignored;
        return false;
    }

    return gpu_free(ds);
}

/*
Python wrapper for gpu_get_accel. Validates NumPy inputs, preserves the CPU module's
get_accel(R, M, collision_radius) signature, and returns an N x 3 float64 array.
*/
static PyObject* py_get_accel([[maybe_unused]] PyObject* self, PyObject* args) {
    PyObject* R_obj = nullptr;
    PyObject* M_obj = nullptr;
    double collision_radius = GPU_ZERO;

    if (!PyArg_ParseTuple(args, "OOd", &R_obj, &M_obj, &collision_radius)) {
        return nullptr;
    }

    PyArrayObject* R_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(R_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* M_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(M_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* A_arr = nullptr;

    if (R_arr == nullptr || M_arr == nullptr) {
        Py_XDECREF(R_arr);
        Py_XDECREF(M_arr);
        return nullptr;
    }

    if (PyArray_NDIM(R_arr) != 2 || PyArray_DIM(R_arr, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "R must be an N x 3 float64 array");
        Py_DECREF(R_arr);
        Py_DECREF(M_arr);
        return nullptr;
    }
    if (PyArray_NDIM(M_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "M must be a 1D float64 array");
        Py_DECREF(R_arr);
        Py_DECREF(M_arr);
        return nullptr;
    }

    if (PyArray_DIM(M_arr, 0) != PyArray_DIM(R_arr, 0)) {
        PyErr_SetString(PyExc_ValueError, "R and M must describe the same number of particles");
        Py_DECREF(R_arr);
        Py_DECREF(M_arr);
        return nullptr;
    }
    const int N = static_cast<int>(PyArray_DIM(R_arr, 0));

    npy_intp dims[2] = {PyArray_DIM(R_arr, 0), 3};
    A_arr = reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2, dims, NPY_DOUBLE, 0));
    if (A_arr == nullptr) {
        Py_DECREF(R_arr);
        Py_DECREF(M_arr);
        return nullptr;
    }

    if (N > 0) {
        const double* R_data = static_cast<const double*>(PyArray_DATA(R_arr));
        const double* M_data = static_cast<const double*>(PyArray_DATA(M_arr));
        double* A_data = static_cast<double*>(PyArray_DATA(A_arr));
        if (!gpu_get_accel(R_data, M_data, A_data, N, collision_radius)) {
            Py_DECREF(R_arr);
            Py_DECREF(M_arr);
            Py_DECREF(A_arr);
            return nullptr;
        }
    }

    Py_DECREF(R_arr);
    Py_DECREF(M_arr);
    return reinterpret_cast<PyObject*>(A_arr);
}

/*
Python diagnostic wrapper that reports the SM count of the currently selected CUDA device.
Assumes a CUDA-capable device is available and leaves CUDA runtime state otherwise unchanged.
*/
static PyObject* py_get_num_sm([[maybe_unused]] PyObject* self, [[maybe_unused]] PyObject* args) {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id), { return nullptr; });

    cudaDeviceProp device_props{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id), { return nullptr; });
    return PyLong_FromLong(device_props.multiProcessorCount);
}

static PyMethodDef accel_gpu_methods[] = {
    {"get_accel", py_get_accel, METH_VARARGS,
     "Compute gravitational acceleration for N bodies on the GPU."},
    {"get_num_sm", py_get_num_sm, METH_NOARGS,
     "Return the number of streaming multiprocessors on the active CUDA device."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef accel_gpu_module = {
    PyModuleDef_HEAD_INIT,
    "accel_gpu",
    "CUDA-accelerated N-body gravitational acceleration module.",
    -1,
    accel_gpu_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit_accel_gpu(void) {
    import_array();
    return PyModule_Create(&accel_gpu_module);
}
