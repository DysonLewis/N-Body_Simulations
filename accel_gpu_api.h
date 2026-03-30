#pragma once

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeviceState DeviceState;

/*
Allocates one DeviceState with SoA buffers sized for N particles.
Returns nullptr and sets a Python exception on allocation failure.
*/
DeviceState* gpu_alloc(int N);

/*
Frees all device buffers owned by ds and deletes the wrapper struct.
Returns false and sets a Python exception if any CUDA cleanup step fails.
*/
bool gpu_free(DeviceState* ds);

/*
Uploads host AoS positions/velocities plus masses into device SoA buffers.
Returns false and sets a Python exception if validation or H->D transfer fails.
*/
bool gpu_upload(
    DeviceState* ds,
    const double* X_aos,
    const double* V_aos,
    const double* M,
    int N);

/*
Downloads device SoA positions/velocities/energies into host AoS buffers.
Returns false and sets a Python exception if validation or D->H transfer fails.
*/
bool gpu_download(
    DeviceState* ds,
    double* X_aos,
    double* V_aos,
    double* KE,
    double* PE,
    int N);

/*
Re-uploads corrected host AoS positions and velocities after CPU collision handling.
Masses remain resident on the device and are not modified by this call.
*/
bool gpu_upload_positions_velocities(
    DeviceState* ds,
    const double* X_aos,
    const double* V_aos,
    int N);

/*
Launches the tiled force kernel and refreshes device accelerations in-place.
Returns false and sets a Python exception if the kernel launch fails.
*/
bool gpu_launch_force(DeviceState* ds, double collision_radius);

/*
Launches the leapfrog half-kick kernel with the supplied full-step dt.
The wrapper applies the dt/2 factor internally.
*/
bool gpu_launch_halfkick(DeviceState* ds, double dt);

/*
Launches the leapfrog full-kick kernel with the supplied full-step dt.
The wrapper applies the dt/2 factor internally.
*/
bool gpu_launch_fullkick(DeviceState* ds, double dt);

/*
Launches the leapfrog drift kernel with the supplied full-step dt.
Returns false and sets a Python exception if the kernel launch fails.
*/
bool gpu_launch_drift(DeviceState* ds, double dt);

/*
Launches the kinetic-energy kernel and overwrites ds->d_ke.
Returns false and sets a Python exception if the kernel launch fails.
*/
bool gpu_launch_ke(DeviceState* ds);

/*
Zeroes ds->d_pe, launches the potential-energy kernel, and overwrites ds->d_pe.
Returns false and sets a Python exception if zeroing or launch fails.
*/
bool gpu_launch_pe(DeviceState* ds, double collision_radius);

/*
Synchronizes the default CUDA stream exactly when the caller needs a chunk boundary.
Returns false and sets a Python exception if cudaDeviceSynchronize fails.
*/
bool gpu_sync(void);

#ifdef __cplusplus
}
#endif
