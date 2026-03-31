import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from accel_dispatcher import get_backend, get_num_threads, run_simulation
from visualize_live import SimulationVisualizer
from ensemble_analysis import analyze_ensemble
import os
import tempfile
import threading
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import warnings

# Suppress matplotlib tight_layout warning
warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')

# Suppress Qt warnings
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm
Msol = 1.989e33    # solar mass in grams
yr = 3.15576e7     # year in seconds
G = 6.6743e-8      # gravitational constant in cm^3 g^-1 s^-2

def get_project_root():
    """Resolve the project root from the script location or current working directory."""
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.getcwd())


plot_dir = os.path.join(get_project_root(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# Visualization parameters
TRAIL_LENGTH = 50  # Number of historical positions to keep per particle
TARGET_FPS = 30    # Target frames per second for animation
WINDOW_WIDTH = 1400  # Total window width in pixels
WINDOW_HEIGHT = 800  # Total window height in pixels
MAX_PLOT_TIME_POINTS = 500  # Cap static plots to a manageable number of sampled time points.

DEFAULT_PARTICLES = 20
DEFAULT_SPHERE_RADIUS_AU = 0.001
DEFAULT_TOTAL_MASS_MSOL = 1e-18
DEFAULT_MAX_YEARS = 25000.0
DEFAULT_N_SIMULATIONS = 1
DEFAULT_COLLISION_RADIUS_FACTOR = 0.01
DEFAULT_CHUNK_STEPS = 10000
DEFAULT_TIMESTEP_YEARS = 0.1
DEFAULT_FITS_BASENAME = "nbody_simulations.fits"


def parse_args():
    """Parse runtime configuration so notebook cells can override defaults."""
    parser = argparse.ArgumentParser(
        description="Run the N-body simulation with optional runtime overrides."
    )
    parser.add_argument(
        "-render",
        "--render",
        action="store_true",
        help=(
            "Skip the simulation and render an existing FITS file instead. "
            "Defaults to the most recently modified FITS file in the project directory."
        ),
    )
    parser.add_argument(
        "--fits-file",
        type=str,
        default=None,
        help="Path to an existing FITS file for --render mode.",
    )
    parser.add_argument(
        "--render-sim",
        type=int,
        default=None,
        help="Simulation ID to render from the FITS file (default: highest completed SIMID).",
    )
    parser.add_argument(
        "-particles",
        "--particles",
        type=int,
        default=None,
        help=(
            f"Number of particles to simulate (default: {DEFAULT_PARTICLES}). "
            "In --render mode, limits how many particles are displayed."
        ),
    )
    parser.add_argument(
        "--sphere-radius-au",
        type=float,
        default=DEFAULT_SPHERE_RADIUS_AU,
        help=f"Initial sphere radius in AU (default: {DEFAULT_SPHERE_RADIUS_AU})",
    )
    parser.add_argument(
        "--total-mass-msol",
        type=float,
        default=DEFAULT_TOTAL_MASS_MSOL,
        help=f"Total mass in solar masses (default: {DEFAULT_TOTAL_MASS_MSOL})",
    )
    parser.add_argument(
        "--max-years",
        type=float,
        default=DEFAULT_MAX_YEARS,
        help=f"Simulation duration in years (default: {DEFAULT_MAX_YEARS})",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help=f"Number of simulations to run (default: {DEFAULT_N_SIMULATIONS})",
    )
    parser.add_argument(
        "--collision-radius-factor",
        type=float,
        default=DEFAULT_COLLISION_RADIUS_FACTOR,
        help=(
            "Collision radius as a fraction of sphere radius "
            f"(default: {DEFAULT_COLLISION_RADIUS_FACTOR})"
        ),
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=DEFAULT_CHUNK_STEPS,
        help=f"Rows per FITS write chunk (default: {DEFAULT_CHUNK_STEPS})",
    )
    parser.add_argument(
        "--dt-years",
        type=float,
        default=DEFAULT_TIMESTEP_YEARS,
        help=f"Timestep in years (default: {DEFAULT_TIMESTEP_YEARS})",
    )
    return parser.parse_args()


args = parse_args()
N = DEFAULT_PARTICLES
sphere_radius = DEFAULT_SPHERE_RADIUS_AU * AU
total_mass = DEFAULT_TOTAL_MASS_MSOL * Msol
max_years = DEFAULT_MAX_YEARS
n_simulations = DEFAULT_N_SIMULATIONS
collision_radius_factor = DEFAULT_COLLISION_RADIUS_FACTOR
chunk_steps = DEFAULT_CHUNK_STEPS

# # Get simulation parameters from user
# N = int(input("Enter number of particles: "))
# sphere_radius = float(input("Enter sphere radius [AU]: ")) * AU
# total_mass = float(input("Enter total system mass [solar masses]: ")) * Msol
# n_years = float(input("Enter simulation time [years]: "))
# n_simulations = int(input("Enter number of simulations to run: "))

# Calculate derived parameters
particle_mass = total_mass / N
dt = DEFAULT_TIMESTEP_YEARS * yr
max_step = int((max_years * yr) / dt)
collision_radius = collision_radius_factor * sphere_radius


def validate_simulation_args(parsed_args):
    """Validate CLI arguments used for simulation runs."""
    effective_particles = (
        DEFAULT_PARTICLES if parsed_args.particles is None else parsed_args.particles
    )
    if effective_particles <= 0:
        raise ValueError("--particles must be positive")
    if parsed_args.sphere_radius_au <= 0:
        raise ValueError("--sphere-radius-au must be positive")
    if parsed_args.total_mass_msol <= 0:
        raise ValueError("--total-mass-msol must be positive")
    if parsed_args.max_years <= 0:
        raise ValueError("--max-years must be positive")
    if parsed_args.n_simulations <= 0:
        raise ValueError("--n-simulations must be positive")
    if parsed_args.collision_radius_factor <= 0:
        raise ValueError("--collision-radius-factor must be positive")
    if parsed_args.chunk_steps <= 0:
        raise ValueError("--chunk-steps must be positive")
    if parsed_args.dt_years <= 0:
        raise ValueError("--dt-years must be positive")
    if parsed_args.render_sim is not None and parsed_args.render_sim <= 0:
        raise ValueError("--render-sim must be positive")


def configure_simulation(parsed_args):
    """Populate module-level simulation parameters from validated CLI args."""
    global N, sphere_radius, total_mass, max_years, n_simulations
    global collision_radius_factor, chunk_steps, particle_mass, dt
    global max_step, collision_radius

    validate_simulation_args(parsed_args)

    N = DEFAULT_PARTICLES if parsed_args.particles is None else parsed_args.particles
    sphere_radius = parsed_args.sphere_radius_au * AU
    total_mass = parsed_args.total_mass_msol * Msol
    max_years = parsed_args.max_years
    n_simulations = parsed_args.n_simulations
    collision_radius_factor = parsed_args.collision_radius_factor
    chunk_steps = parsed_args.chunk_steps

    particle_mass = total_mass / N
    dt = parsed_args.dt_years * yr
    max_step = int((max_years * yr) / dt)
    collision_radius = collision_radius_factor * sphere_radius


def get_default_fits_path():
    """Store the main FITS output alongside the simulation scripts."""
    return os.path.join(get_project_root(), DEFAULT_FITS_BASENAME)


def resolve_fits_path(fits_filename):
    """Resolve an explicit FITS path against cwd and the project directory."""
    if os.path.isabs(fits_filename):
        return fits_filename

    cwd_candidate = os.path.abspath(fits_filename)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    project_candidate = os.path.join(get_project_root(), fits_filename)
    if os.path.exists(project_candidate):
        return project_candidate

    return cwd_candidate


def find_latest_fits_file():
    """Find the most recently modified FITS file in cwd or the project directory."""
    candidate_dirs = [os.getcwd(), get_project_root()]
    fits_paths = []
    seen = set()

    for directory in candidate_dirs:
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            if not entry.lower().endswith(".fits"):
                continue
            path = os.path.abspath(os.path.join(directory, entry))
            if path in seen or not os.path.isfile(path):
                continue
            fits_paths.append(path)
            seen.add(path)

    if not fits_paths:
        raise FileNotFoundError(
            "No FITS files found in the current working directory or project directory."
        )

    return max(fits_paths, key=os.path.getmtime)


def get_render_target(fits_filename=None, sim_id=None):
    """Resolve the FITS file and simulation ID for render-only mode."""
    resolved_fits = resolve_fits_path(fits_filename) if fits_filename else find_latest_fits_file()
    if not os.path.exists(resolved_fits):
        raise FileNotFoundError(f"FITS file not found: {resolved_fits}")

    with fits.open(resolved_fits, memmap=True) as hdul:
        header = hdul[0].header
        sphere_radius_cm = header.get("SPHRAD")
        if sphere_radius_cm is None:
            raise ValueError(f"FITS header missing SPHRAD: {resolved_fits}")

        completed_sim_ids = []
        fallback_sim_ids = []
        for i in range(1, len(hdul)):
            if "SIMID" not in hdul[i].header:
                continue
            current_sim_id = int(hdul[i].header["SIMID"])
            fallback_sim_ids.append(current_sim_id)
            if "CHUNKIDX" not in hdul[i].header:
                completed_sim_ids.append(current_sim_id)

    available_sim_ids = sorted(set(completed_sim_ids or fallback_sim_ids))
    if not available_sim_ids:
        raise ValueError(f"No simulation HDUs found in FITS file: {resolved_fits}")

    selected_sim_id = sim_id if sim_id is not None else available_sim_ids[-1]
    if selected_sim_id not in available_sim_ids:
        raise ValueError(
            f"Simulation {selected_sim_id} not found in {resolved_fits}. "
            f"Available SIMIDs: {available_sim_ids}"
        )

    return resolved_fits, selected_sim_id, float(sphere_radius_cm)


def render_existing_simulation(fits_filename=None, sim_id=None):
    """Launch the visualizer without running any new simulation."""
    resolved_fits, selected_sim_id, sphere_radius_cm = get_render_target(
        fits_filename=fits_filename,
        sim_id=sim_id,
    )
    render_particle_limit = abs(args.particles) if args.particles is not None else None

    print("\n" + "=" * 60)
    print("Render-only mode")
    print("=" * 60)
    print(f"FITS file: {resolved_fits}")
    print(f"Simulation ID: {selected_sim_id}")
    print(f"Initial sphere radius: {sphere_radius_cm / AU:.6e} AU")
    if render_particle_limit is not None:
        print(f"Render particle limit: {render_particle_limit}")
    print("Close the visualization window to exit")
    print("=" * 60)

    visualizer = SimulationVisualizer(
        resolved_fits,
        sim_id=selected_sim_id,
        sphere_radius=sphere_radius_cm,
        trail_length=TRAIL_LENGTH,
        target_fps=TARGET_FPS,
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        max_particles=render_particle_limit,
    )
    visualizer.run()


def print_simulation_banner():
    """Log the simulation configuration before compute starts."""
    print(f"Compute backend: {get_backend()}")
    print(f"Compute units: {get_num_threads()}")
    print("=" * 60)
    print("N-body simulation")
    print("=" * 60)
    print(f"Particles: {N}")
    print(f"Total mass: {total_mass/Msol:.6e} solar masses")
    print(f"Particle mass: {particle_mass/Msol:.6e} solar masses")
    print(f"Initial sphere radius: {sphere_radius/AU:.6e} AU")
    print(f"Collision radius: {collision_radius/AU:.6e} AU")
    print(f"Timestep: {dt/yr:.4f} years")
    print(f"Nominal maximum steps: {max_step}")
    print(f"Target simulation duration: {max_years} years")
    print(f"Number of simulations: {n_simulations}")
    print("=" * 60)


def generate_sphere_particles(N, radius):
    """
    Generate N particles uniformly distributed in a sphere.
    Uses rejection sampling: generate random points in cube, keep those inside sphere.
    
    Args:
        N: number of particles
        radius: sphere radius in cm
        
    Returns:
        positions: N x 3 array of particle positions in cm
        velocities: N x 3 array of particle velocities in cm/s (all zero)
    """
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    
    for i in range(N):
        while True:
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            z = np.random.uniform(-radius, radius)
            r = np.sqrt(x**2 + y**2 + z**2)
            if r <= radius:
                positions[i] = [x, y, z]
                break
    
    return positions, velocities


def get_simulation_chunk_hdus(hdul, sim_id):
    """Return sorted temporary FITS chunk HDUs for a simulation."""
    chunk_hdus = []
    for i in range(1, len(hdul)):
        header = hdul[i].header
        if header.get('SIMID') == sim_id and 'CHUNKIDX' in header:
            chunk_hdus.append((int(header['CHUNKIDX']), i))
    chunk_hdus.sort(key=lambda item: item[0])
    return chunk_hdus


def combine_simulation_chunks(fits_filename, sim_id, expected_chunk_count=None):
    """
    Combine temporary per-chunk HDUs into one simulation HDU using a disk-backed buffer.

    The old implementation loaded every chunk into a Python list and then called
    ``np.concatenate``. That doubled peak memory usage right at the end of the run.
    Here we copy chunk HDUs sequentially into a memmap on disk, append the final HDU,
    and only then remove the temporary chunk HDUs.
    """
    import gc

    print(f"Combining temporary chunks into single HDU for simulation {sim_id}...")

    temp_buffer_path = None
    combined_data = None
    try:
        with fits.open(fits_filename, memmap=True) as hdul:
            chunk_hdus = get_simulation_chunk_hdus(hdul, sim_id)
            if not chunk_hdus:
                raise ValueError(f"No temporary chunk HDUs found for simulation {sim_id}")
            if expected_chunk_count is not None and len(chunk_hdus) != expected_chunk_count:
                raise ValueError(
                    f"Expected {expected_chunk_count} chunk HDUs for simulation {sim_id}, "
                    f"found {len(chunk_hdus)}"
                )

            first_chunk = hdul[chunk_hdus[0][1]].data
            total_rows = sum(len(hdul[hdu_idx].data) for _, hdu_idx in chunk_hdus)

            with tempfile.NamedTemporaryFile(
                prefix=f"sim_{sim_id}_combine_",
                suffix=".dat",
                dir=os.path.dirname(os.path.abspath(fits_filename)) or None,
                delete=False,
            ) as temp_buffer:
                temp_buffer_path = temp_buffer.name

            combined_data = np.memmap(
                temp_buffer_path,
                dtype=first_chunk.dtype,
                mode='w+',
                shape=(total_rows,),
            )

            write_row = 0
            with tqdm(
                total=len(chunk_hdus),
                desc=f"Combining chunks (sim {sim_id})",
                unit="chunk",
                leave=True,
            ) as pbar:
                for _, hdu_idx in chunk_hdus:
                    chunk_data = hdul[hdu_idx].data
                    next_row = write_row + len(chunk_data)
                    combined_data[write_row:next_row] = chunk_data
                    write_row = next_row
                    pbar.update(1)

            combined_data.flush()
            del first_chunk
            gc.collect()

        with fits.open(fits_filename, mode='append', memmap=True) as hdul:
            final_hdu = fits.BinTableHDU(combined_data)
            final_hdu.header['SIMID'] = (sim_id, 'Simulation ID number')
            final_hdu.header['EXTNAME'] = f'SIM_{sim_id}'
            hdul.append(final_hdu)
            hdul.flush()

    finally:
        if combined_data is not None:
            del combined_data
            gc.collect()
        if temp_buffer_path and os.path.exists(temp_buffer_path):
            os.remove(temp_buffer_path)


def run_and_save_simulation(sim_id, fits_filename, append=False):
    """
    Run a single simulation and save to FITS file incrementally.
    All chunks from this simulation go into a single HDU.
    
    Args:
        sim_id: simulation identifier (1-indexed)
        fits_filename: path to FITS file
        append: if True, append to existing file; if False, create new file
        
    Returns:
        DataFrame with simulation results (only first chunk for visualization)
    """
    import gc
    
    print(f"\nSimulation {sim_id}/{n_simulations}")
    
    # Generate initial conditions
    X0, V0 = generate_sphere_particles(N, sphere_radius)
    M = np.full(N, particle_mass)
    
    # Empty perturbation arrays (not used)
    perturb_indices = np.array([], dtype=np.int32)
    perturb_pos = np.zeros((0, 3))
    perturb_vel = np.zeros((0, 3))
    
    # Run simulation - returns list of chunk arrays
    result_chunks = run_simulation(
        X0, V0, M,
        perturb_indices,
        perturb_pos,
        perturb_vel,
        sim_id,
        max_step,
        chunk_steps,
        dt,
        yr,
        collision_radius
    )

    if result_chunks:
        final_time_yr = float(result_chunks[-1][0, 1])
        print(
            f"Simulation {sim_id} produced {len(result_chunks)} stored snapshots "
            f"through {final_time_yr:.1f} years"
        )
    
    # Create FITS file with header on first simulation
    if not append:
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['N_BODIES'] = N
        primary_hdu.header['SPHRAD'] = (sphere_radius, 'Initial sphere radius [cm]')
        primary_hdu.header['TOTMASS'] = (total_mass, 'Total system mass [g]')
        primary_hdu.header['DT'] = (dt, 'Timestep [s]')
        primary_hdu.header['MAXSTEP'] = max_step
        primary_hdu.header['COLLRAD'] = (collision_radius, 'Collision radius [cm]')
        primary_hdu.header['NSIMS'] = n_simulations
        
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(fits_filename, overwrite=True)
        del hdul
        gc.collect()
    
    # Keep first chunk for visualization
    first_chunk_df = None
    chunk_count = len(result_chunks)

    # Process and write chunks one at a time, releasing each chunk array immediately
    # after its FITS HDU has been flushed to disk.
    with tqdm(total=chunk_count, desc=f"Writing chunks (sim {sim_id})", 
              unit="chunk", leave=True) as pbar:
        for chunk_idx in range(chunk_count):
            chunk_array = result_chunks[chunk_idx]
            # Convert chunk to DataFrame
            df_chunk = pd.DataFrame(chunk_array, columns=[
                "simulation", "time_yr", "body_idx",
                "x_cm", "y_cm", "z_cm",
                "vx_cm_s", "vy_cm_s", "vz_cm_s",
                "KE", "PE"
            ])
            
            df_chunk["E_tot"] = df_chunk["KE"] + df_chunk["PE"]
            
            # Keep first chunk for return value
            if chunk_idx == 0:
                first_chunk_df = df_chunk.copy()
            
            # Convert to astropy Table
            table = Table.from_pandas(df_chunk)
            
            # Write as temporary HDU
            table_hdu = fits.BinTableHDU(table)
            table_hdu.header['SIMID'] = (sim_id, 'Simulation ID number')
            table_hdu.header['CHUNKIDX'] = (chunk_idx, 'Temporary chunk index')
            table_hdu.header['EXTNAME'] = f'SIM_{sim_id}_TEMP_{chunk_idx}'
            
            with fits.open(fits_filename, mode='append') as hdul:
                hdul.append(table_hdu)
                hdul.flush()
            
            # Drop all references to this chunk before moving to the next one.
            result_chunks[chunk_idx] = None
            del chunk_array
            del df_chunk
            del table
            del table_hdu
            gc.collect()
            
            pbar.update(1)

    del result_chunks
    gc.collect()

    # Now combine all temporary chunk HDUs into one final HDU without ever keeping the
    # whole simulation in RAM.
    combine_simulation_chunks(fits_filename, sim_id, chunk_count)
    
    # Remove temporary chunk HDUs
    print(f"Cleaning up temporary chunks for simulation {sim_id}...")
    with fits.open(fits_filename, mode='update') as hdul:
        # Collect indices of temp HDUs to remove (backwards so indices don't shift)
        indices_to_remove = []
        for i in range(len(hdul) - 1, 0, -1):
            if ('SIMID' in hdul[i].header and 
                hdul[i].header['SIMID'] == sim_id and
                'CHUNKIDX' in hdul[i].header):
                indices_to_remove.append(i)
        
        # Remove temp HDUs
        for idx in indices_to_remove:
            del hdul[idx]
        
        hdul.flush()
    
    print(f"Simulation {sim_id} complete")
    
    # Return first chunk for visualization
    return first_chunk_df


def background_simulations(fits_filename, start_sim, end_sim):
    """
    Run simulations in background thread and append to FITS file.
    
    Args:
        fits_filename: path to FITS file
        start_sim: first simulation ID to run (inclusive)
        end_sim: last simulation ID to run (inclusive)
    """
    import os
    
    # Set lower priority for this thread to not interfere with visualization
    try:
        os.nice(10)  # Lower priority (higher niceness value)
    except:
        pass  # If it fails (e.g., on Windows), just continue
    
    # Limit OpenMP threads for background simulations to leave cores for visualization
    original_threads = get_num_threads()
    background_threads = max(1, original_threads // 2)  # Use half the threads
    os.environ['OMP_NUM_THREADS'] = str(background_threads)
    print(f"Background simulations using {background_threads}/{original_threads} threads")
    
    for sim_id in range(start_sim, end_sim + 1):
        run_and_save_simulation(sim_id, fits_filename, append=True)
    
    # Restore thread count (though thread is about to end anyway)
    os.environ['OMP_NUM_THREADS'] = str(original_threads)
    
    print("\nAll background simulations complete")


def create_static_plots(df, fits_filename):
    """
    Generate static analysis plots from all simulation data.
    Loads data in subsampled chunks to avoid memory overflow.
    
    Args:
        df: UNUSED - kept for compatibility
        fits_filename: name of FITS file (for plot filenames)
    """
    import gc
    
    print("\nCreating static plots...")
    
    # Read subsampled data directly from FITS file
    # Now each simulation is in a single HDU
    all_times = []
    with fits.open(fits_filename) as hdul:
        # Find HDU for simulation 1
        sim_hdu_idx = None
        for i in range(1, len(hdul)):
            if 'SIMID' in hdul[i].header and hdul[i].header['SIMID'] == 1:
                sim_hdu_idx = i
                break
        
        if sim_hdu_idx is None:
            print("Error: Could not find simulation 1 data")
            return
        
        # Load all data for simulation 1
        print("Loading simulation 1 data...")
        sim_data = Table(hdul[sim_hdu_idx].data).to_pandas()
        all_times = sorted(sim_data['time_yr'].unique())
    
    # Subsample time points
    plot_stride = max(1, len(all_times) // MAX_PLOT_TIME_POINTS)
    plot_times = set(all_times[::plot_stride])
    print(
        f"Subsampled to {len(plot_times)} time points from {len(all_times)} total "
        f"(stride {plot_stride})"
    )
    
    # Filter to only subsampled times
    first_sim = sim_data[sim_data['time_yr'].isin(plot_times)]
    del sim_data
    gc.collect()
    
    print(f"Loaded {len(first_sim)} rows for plotting")
    
    plot_particles = min(10, N)
    plot_radius_factor = 3.0
    plot_limit = plot_radius_factor * sphere_radius / AU
    
    # First figure: particle-level plots (2x2)
    print("Generating particle trajectory plots...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    
    for i in range(plot_particles):
        particle_df = first_sim[first_sim["body_idx"] == i]
        
        r = np.sqrt(particle_df["x_cm"]**2 + particle_df["y_cm"]**2 + particle_df["z_cm"]**2)
        axes1[0, 0].plot(particle_df["time_yr"], r/AU, label=f"Particle {i}")
        
        v = np.sqrt(particle_df["vx_cm_s"]**2 + particle_df["vy_cm_s"]**2 + particle_df["vz_cm_s"]**2)
        axes1[0, 1].plot(particle_df["time_yr"], v/1e5, label=f"Particle {i}")
        
        scatter = axes1[1, 0].scatter(particle_df["x_cm"]/AU, particle_df["y_cm"]/AU, 
                                      c=particle_df["time_yr"], cmap='viridis', 
                                      s=2, alpha=0.6, label=f"Particle {i}")
        
        axes1[1, 1].plot(particle_df["time_yr"], particle_df["E_tot"], label=f"Particle {i}")
    
    axes1[0, 0].axhline(y=sphere_radius/AU, color='k', linestyle='--', linewidth=1, label='Initial radius')
    axes1[0, 0].set_xlabel("Time [yr]")
    axes1[0, 0].set_ylabel("Radial Distance [AU]")
    axes1[0, 0].set_title("Radial Distance vs Time")
    axes1[0, 0].legend(fontsize='small')
    
    axes1[0, 1].set_xlabel("Time [yr]")
    axes1[0, 1].set_ylabel("Speed [km/s]")
    axes1[0, 1].set_title("Speed vs Time")
    axes1[0, 1].legend(fontsize='small')
    
    circle = plt.Circle((0, 0), sphere_radius/AU, color='k', fill=False, linestyle='--', linewidth=1, label='Initial sphere')
    axes1[1, 0].add_patch(circle)
    axes1[1, 0].set_xlabel("X [AU]")
    axes1[1, 0].set_ylabel("Y [AU]")
    axes1[1, 0].set_title("XY Trajectories (colored by time)")
    axes1[1, 0].axis('equal')
    axes1[1, 0].set_xlim(-plot_limit, plot_limit)
    axes1[1, 0].set_ylim(-plot_limit, plot_limit)
    cbar = plt.colorbar(scatter, ax=axes1[1, 0], label='Time [yr]')
    axes1[1, 0].legend(fontsize='small')
    
    axes1[1, 1].set_xlabel("Time [yr]")
    axes1[1, 1].set_ylabel("Total Energy [erg]")
    axes1[1, 1].set_title("Total Energy vs Time")
    axes1[1, 1].legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"system_of_{N}_particles.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # Second figure: system-level plots (1x2)
    print("Generating energy and virial plots...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    system_energy = first_sim.groupby('time_yr').agg({
        'KE': 'sum',
        'PE': 'sum',
        'E_tot': 'sum'
    }).reset_index()
    
    axes2[0].plot(system_energy['time_yr'], system_energy['KE'], label='Total KE', linewidth=1.5)
    axes2[0].plot(system_energy['time_yr'], system_energy['PE'], label='Total PE', linewidth=1.5)
    axes2[0].plot(system_energy['time_yr'], system_energy['E_tot'], label='Total Energy', linewidth=1.5, linestyle='--')
    axes2[0].set_xlabel("Time [yr]")
    axes2[0].set_ylabel("Energy [erg]")
    axes2[0].set_title("System Energy vs Time")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    virial_ratio = np.abs(2.0 * system_energy['KE'] / system_energy['PE'])
    axes2[1].plot(system_energy['time_yr'], virial_ratio, linewidth=1.5, color='purple')
    axes2[1].axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Ideal virial (ratio=1)')
    axes2[1].fill_between(system_energy['time_yr'], 0.95, 1.05, alpha=0.2, color='green', label='Equilibrium zone')
    axes2[1].set_xlabel("Time [yr]")
    axes2[1].set_ylabel("Virial Ratio |2*KE/PE|")
    axes2[1].set_title("Virial Equilibrium Check")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Virial_for_{N}_particles.png"), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # Final cleanup
    del first_sim
    gc.collect()
    
    print("Static plots complete!")


def main():
    """
    Main execution flow:
    1. Run simulation 1 and save to FITS
    2. Start background thread for simulations 2-N
    3. Visualize simulation 1 (blocking on main thread)
    4. Wait for background simulations to complete
    5. Generate static plots
    """
    if args.render:
        render_existing_simulation(
            fits_filename=args.fits_file,
            sim_id=args.render_sim,
        )
        return

    configure_simulation(args)
    print_simulation_banner()

    fits_filename = get_default_fits_path()
    
    # Run first simulation and save to FITS
    print("\n" + "=" * 60)
    print("Running first simulation...")
    print("=" * 60)
    df_sim1 = run_and_save_simulation(1, fits_filename, append=False)
    
    # Start background thread for remaining simulations
    background_thread = None
    if n_simulations > 1:
        print("\n" + "=" * 60)
        print(f"Starting background thread for simulations 2-{n_simulations}...")
        print("=" * 60)
        background_thread = threading.Thread(
            target=background_simulations,
            args=(fits_filename, 2, n_simulations),
            daemon=False
        )
        background_thread.start()
    
    # Visualize first simulation on main thread
    print("\n" + "=" * 60)
    print("Starting visualization of simulation 1...")
    print("Close the visualization window to continue")
    print("=" * 60)
    
    visualizer = SimulationVisualizer(
        fits_filename,
        sim_id=1,
        sphere_radius=sphere_radius,
        trail_length=TRAIL_LENGTH,
        target_fps=TARGET_FPS,
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT
    )
    visualizer.run()
    
    print("\nVisualization window closed")
    
    # Wait for background simulations to complete
    if background_thread is not None:
        print("\nWaiting for background simulations to complete...")
        background_thread.join()
    
    # Read all data from FITS file for final plots
    print("\n" + "=" * 60)
    print("Generating static plots...")
    print("=" * 60)
    
    # Pass fits_filename to plotting function - it will handle memory-efficient loading
    create_static_plots(None, fits_filename)
    
    if n_simulations > 1:
        analyze_ensemble(fits_filename)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
