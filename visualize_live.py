import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from collections import deque
import time
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import warnings
import os

# Suppress matplotlib tight_layout warning
warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')

# Suppress Qt warnings
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

# Physical constants in CGS units
AU = 1.496e13      # astronomical unit in cm
DEFAULT_PLAYBACK_SECONDS = 30.0


class SimulationVisualizer:
    """
    Playback visualizer for pre-computed N-body simulation data.
    
    Displays particles in XY plane with trailing motion blur effect and
    a virial ratio indicator bar showing system energy balance.
    
    Reads directly from FITS file in streaming fashion to minimize memory usage.
    Now assumes one HDU per simulation (not one per chunk).
    """
    
    def __init__(self, fits_filename, sim_id, sphere_radius, visualization_interval=1,
                 trail_length=50, target_fps=30, window_width=1400, window_height=800,
                 chunk_cache_size=5, max_particles=None):
        """
        Initialize the visualization window and data structures.
        
        Args:
            fits_filename: path to FITS file containing simulation data
            sim_id: which simulation to visualize
            sphere_radius: initial sphere radius in cm
            visualization_interval: UNUSED - kept for compatibility
            trail_length: number of historical positions to keep per particle
            target_fps: target frames per second for animation
            window_width: total window width in pixels
            window_height: total window height in pixels
            chunk_cache_size: number of time-based chunks to keep in memory at once
            max_particles: optional cap on how many bodies to display
        """
        self.fits_filename = fits_filename
        self.sim_id = sim_id
        self.sphere_radius = sphere_radius
        self.trail_length = trail_length
        self.target_fps = target_fps
        self.window_width = window_width
        self.window_height = window_height
        self.chunk_cache_size = chunk_cache_size
        self.max_particles = max_particles
        
        # Read FITS file to get metadata
        print("Loading simulation metadata from FITS...")
        with fits.open(fits_filename, memmap=True) as hdul:
            # Get number of bodies from header
            self.total_bodies = int(hdul[0].header['N_BODIES'])
            if max_particles is None:
                self.N = self.total_bodies
            else:
                self.N = min(int(max_particles), self.total_bodies)
                if self.N <= 0:
                    raise ValueError("max_particles must be positive")
            
            # Find HDU for this simulation
            self.sim_hdu_idx = None
            for i in range(1, len(hdul)):
                if 'SIMID' in hdul[i].header and hdul[i].header['SIMID'] == sim_id:
                    self.sim_hdu_idx = i
                    break
            
            if self.sim_hdu_idx is None:
                raise ValueError(f"Could not find simulation {sim_id} in FITS file")
            
            # Get all unique time points efficiently by sampling
            # Read just the time_yr column to build index
            print(f"Building time index for simulation {sim_id}...")
            time_col = hdul[self.sim_hdu_idx].data['time_yr']
            self.all_time_points = sorted(list(set(time_col)))
            self.n_total_frames = len(self.all_time_points)
            
            # Build index: time_val -> row range for fast lookup
            # Group consecutive rows by time to create chunks
            self.time_to_row_range = {}
            current_time = None
            start_row = 0
            
            for row_idx, time_val in enumerate(time_col):
                if time_val != current_time:
                    if current_time is not None:
                        self.time_to_row_range[current_time] = (start_row, row_idx)
                    current_time = time_val
                    start_row = row_idx
            
            # Don't forget the last time point
            if current_time is not None:
                self.time_to_row_range[current_time] = (start_row, len(time_col))
            
            print(f"Found {self.n_total_frames} time points")

        if self.N < self.total_bodies:
            print(f"Rendering first {self.N} of {self.total_bodies} particles")
        else:
            print(f"Rendering all {self.total_bodies} particles")

        if self.n_total_frames > 1:
            frame_deltas = np.diff(self.all_time_points)
            positive_deltas = frame_deltas[frame_deltas > 0.0]
            if len(positive_deltas) > 0:
                self.median_frame_delta_yr = float(np.median(positive_deltas))
            else:
                self.median_frame_delta_yr = 1.0
            self.total_time_span_yr = float(self.all_time_points[-1] - self.all_time_points[0])
        else:
            self.median_frame_delta_yr = 1.0
            self.total_time_span_yr = 0.0
        
        self.current_time_index = 0
        
        # Chunk cache: {time_val: dataframe}
        self.chunk_cache = {}
        self.chunk_lru = deque(maxlen=chunk_cache_size)
        
        # Preload first few time points
        print("Preloading initial frames...")
        frames_to_preload = min(chunk_cache_size, len(self.all_time_points))
        for i in range(frames_to_preload):
            self._load_frame(self.all_time_points[i])
        print(f"Preloaded {frames_to_preload} frames")
        
        # Trail data: each particle has deque of (x, y) positions
        self.particle_trails = [deque(maxlen=trail_length) for _ in range(self.N)]
        
        # Particle colors: use colormap for distinct colors
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.N, endpoint=False))
        
        # Playback control - speed in years per second
        self.default_years_per_second = max(
            1.0,
            self.total_time_span_yr / DEFAULT_PLAYBACK_SECONDS if self.total_time_span_yr > 0.0 else 1.0,
        )
        self.min_years_per_second = max(1.0e-3, self.default_years_per_second / 10000.0)
        self.max_years_per_second = max(
            self.default_years_per_second * 100.0,
            self.median_frame_delta_yr * self.target_fps,
        )
        self.years_per_second = self.default_years_per_second
        self.last_real_time = time.time()
        self.accumulated_sim_time = 0.0  # How much simulation time we should have shown
        
        # Playback control
        self.paused = False
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Plot limits (auto-scale based on initial sphere)
        self.plot_limit = 2.0 * sphere_radius / AU
        
        self._setup_figure()
    
    def _load_frame(self, time_val):
        """
        Load data for a specific time value from FITS file into cache.
        
        Args:
            time_val: time in years
        """
        # Check if already cached
        if time_val in self.chunk_cache:
            return
        
        # If cache is full, remove oldest
        if len(self.chunk_cache) >= self.chunk_cache_size:
            if len(self.chunk_lru) > 0:
                old_time = self.chunk_lru.popleft()
                if old_time in self.chunk_cache:
                    del self.chunk_cache[old_time]
        
        # Get row range for this time point
        if time_val not in self.time_to_row_range:
            return
        
        start_row, end_row = self.time_to_row_range[time_val]
        
        # Load only these rows from FITS
        with fits.open(self.fits_filename, memmap=True) as hdul:
            # Use array slicing to read only needed rows
            subset_data = hdul[self.sim_hdu_idx].data[start_row:end_row]
            frame_df = Table(subset_data).to_pandas()
            
            self.chunk_cache[time_val] = frame_df
            self.chunk_lru.append(time_val)
    
    def _get_frame_data(self, time_val):
        """
        Get data for a specific time value, loading from FITS if needed.
        
        Args:
            time_val: time in years
            
        Returns:
            DataFrame for that time point
        """
        # Load frame if not in cache
        if time_val not in self.chunk_cache:
            self._load_frame(time_val)
            
            # Predictive preloading: load next frame too if not already loaded
            try:
                current_idx = self.all_time_points.index(time_val)
                if current_idx + 1 < len(self.all_time_points):
                    next_time = self.all_time_points[current_idx + 1]
                    if next_time not in self.chunk_cache:
                        self._load_frame(next_time)
            except (ValueError, IndexError):
                pass  # No next frame or already at end
        
        return self.chunk_cache.get(time_val)
        
    def _setup_figure(self):
        """
        Create the matplotlib figure with two panels: XY plot and virial bar.
        """
        # Create figure with custom size
        self.fig = plt.figure(figsize=(self.window_width/100, self.window_height/100), dpi=100)
        
        # Create grid: main plot takes 85% width, virial bar takes 15%
        gs = self.fig.add_gridspec(2, 2, width_ratios=[85, 15], height_ratios=[9, 1],
                                   hspace=0.15, wspace=0.15)
        
        # Left panel: XY trajectory plot
        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.ax_xy.set_xlabel("X [AU]", fontsize=12)
        self.ax_xy.set_ylabel("Y [AU]", fontsize=12)
        self.ax_xy.set_title("Particle Trajectories (XY Plane)", fontsize=14, fontweight='bold')
        self.ax_xy.set_xlim(-self.plot_limit, self.plot_limit)
        self.ax_xy.set_ylim(-self.plot_limit, self.plot_limit)
        self.ax_xy.set_aspect('equal')
        self.ax_xy.grid(True, alpha=0.3, linestyle='--')
        
        # Draw initial sphere boundary as reference
        initial_circle = plt.Circle((0, 0), self.sphere_radius/AU, 
                                   color='gray', fill=False, 
                                   linestyle='--', linewidth=2, 
                                   label='Initial sphere', alpha=0.5)
        self.ax_xy.add_patch(initial_circle)
        
        # Initialize line collections for trails (one per particle)
        self.trail_collections = []
        for i in range(self.N):
            lc = LineCollection([], linewidths=2, colors=self.colors[i])
            self.ax_xy.add_collection(lc)
            self.trail_collections.append(lc)
        
        # Current particle positions (scatter plot)
        self.particle_scatter = self.ax_xy.scatter([], [], s=30, c=[],
                                                   edgecolors='black', linewidths=1.5,
                                                   zorder=10, label='Particles')
        self.particle_scatter.set_offsets(np.empty((0, 2)))
        
        # Stats text overlay (top-left corner)
        self.stats_text = self.ax_xy.text(0.02, 0.98, '', 
                                          transform=self.ax_xy.transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                          fontsize=11, family='monospace')
        
        # Right panel: Virial ratio bar
        self.ax_virial = self.fig.add_subplot(gs[0, 1])
        self.ax_virial.set_xlim(0, 1)
        self.ax_virial.set_ylim(0.5, 1.5)
        self.ax_virial.set_ylabel("Virial Ratio |2KE/PE|", fontsize=12)
        self.ax_virial.set_title("Energy Balance", fontsize=12, fontweight='bold')
        self.ax_virial.set_xticks([])
        self.ax_virial.grid(True, axis='y', alpha=0.3)
        
        # Equilibrium zone (0.95 to 1.05) highlighted in green
        equilibrium_zone = patches.Rectangle((0, 0.95), 1, 0.1, 
                                            linewidth=0, edgecolor=None,
                                            facecolor='green', alpha=0.2,
                                            label='Equilibrium zone')
        self.ax_virial.add_patch(equilibrium_zone)
        
        # Center line at ratio = 1.0
        self.ax_virial.axhline(y=1.0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        
        # Virial ratio indicator (sliding horizontal bar)
        self.virial_indicator = patches.Rectangle((0.1, 0.98), 0.8, 0.04,
                                                  linewidth=2, edgecolor='black',
                                                  facecolor='blue', alpha=0.8)
        self.ax_virial.add_patch(self.virial_indicator)
        
        # Virial ratio value text
        self.virial_text = self.ax_virial.text(0.5, 1.0, '1.00',
                                              ha='center', va='bottom',
                                              fontsize=14, fontweight='bold',
                                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Labels for top/bottom of bar
        self.ax_virial.text(0.5, 1.48, 'KE Dominated', ha='center', fontsize=9, style='italic')
        self.ax_virial.text(0.5, 0.52, 'PE Dominated', ha='center', fontsize=9, style='italic')
        
        # Speed control slider (bottom panel, spans both columns)
        self.ax_slider = self.fig.add_subplot(gs[1, :])
        self.ax_slider.set_title("Playback Speed Control (years per second)", fontsize=10)
        
        slider_min = np.log10(self.min_years_per_second)
        slider_max = np.log10(self.max_years_per_second)
        slider_init = np.log10(self.default_years_per_second)

        # Use a log-speed slider so both dense CPU output and sparse GPU chunk output
        # remain navigable without hard-coding a single years/second range.
        self.speed_slider = Slider(
            ax=self.ax_slider,
            label='log10(speed)',
            valmin=slider_min,
            valmax=slider_max,
            valinit=slider_init,
            color='skyblue'
        )
        
        # Custom slider labels
        midpoint_speed = 10.0 ** ((slider_min + slider_max) / 2.0)
        self.ax_slider.text(
            0.0, -0.5, self._format_speed_label(self.min_years_per_second),
            transform=self.ax_slider.transAxes, ha='left', va='top', fontsize=9)
        self.ax_slider.text(
            0.5, -0.5, self._format_speed_label(midpoint_speed),
            transform=self.ax_slider.transAxes, ha='center', va='top', fontsize=9)
        self.ax_slider.text(
            1.0, -0.5, self._format_speed_label(self.max_years_per_second),
            transform=self.ax_slider.transAxes, ha='right', va='top', fontsize=9)
        
        # Connect slider to callback
        self.speed_slider.on_changed(self._update_speed)

    def _format_speed_label(self, speed_yr_per_s):
        """
        Format playback speed labels compactly across small and very large year ranges.

        Args:
            speed_yr_per_s: playback speed in years per second

        Returns:
            Human-readable label string
        """
        if speed_yr_per_s >= 1000.0:
            return f"{speed_yr_per_s:.0f} yr/s"
        if speed_yr_per_s >= 10.0:
            return f"{speed_yr_per_s:.1f} yr/s"
        return f"{speed_yr_per_s:.2f} yr/s"
        
    def _update_speed(self, val):
        """
        Callback for speed slider changes.
        
        Args:
            val: slider value (log10 of years per second)
        """
        self.years_per_second = 10.0 ** val
        
    def _calculate_fps(self):
        """
        Calculate current frames per second based on recent frame times.
        
        Returns:
            fps: current FPS, or 0 if not enough data
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate FPS from time span of stored frames
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span > 0:
            fps = (len(self.frame_times) - 1) / time_span
            return fps
        return 0.0
    
    def _update_trails(self, positions):
        """
        Update particle trail positions and render with fading alpha.
        
        Args:
            positions: N x 2 array of current particle positions in AU
        """
        for i in range(self.N):
            # Add current position to this particle's trail
            x = positions[i, 0]
            y = positions[i, 1]
            self.particle_trails[i].append((x, y))
            
            # Convert deque to line segments for rendering
            if len(self.particle_trails[i]) > 1:
                trail_points = list(self.particle_trails[i])
                segments = [trail_points[j:j+2] for j in range(len(trail_points)-1)]
                
                # Create alpha gradient (older = more transparent)
                n_segments = len(segments)
                alphas = np.linspace(0.1, 1.0, n_segments)
                
                # Set colors with alpha gradient
                colors = np.zeros((n_segments, 4))
                colors[:, :3] = self.colors[i, :3]
                colors[:, 3] = alphas
                
                # Update line collection
                self.trail_collections[i].set_segments(segments)
                self.trail_collections[i].set_colors(colors)
    
    def _update_virial_bar(self, virial_ratio):
        """
        Update the virial ratio indicator bar position.
        
        Args:
            virial_ratio: current virial ratio value |2*KE/PE|
        """
        # Clamp to display range
        display_ratio = np.clip(virial_ratio, 0.5, 1.5)
        
        # Update indicator position (centered vertically at ratio value)
        self.virial_indicator.set_y(display_ratio - 0.02)
        
        # Color code: green near equilibrium, blue if PE dominated, red if KE dominated
        if 0.95 <= virial_ratio <= 1.05:
            color = 'green'
        elif virial_ratio < 0.95:
            color = 'blue'
        else:
            color = 'red'
        self.virial_indicator.set_facecolor(color)
        
        # Update text value
        self.virial_text.set_text(f'{virial_ratio:.2f}')
        self.virial_text.set_position((0.5, display_ratio))
    
    def update_frame(self, frame_num):
        """
        Animation update function called by FuncAnimation.
        
        Advances through pre-computed simulation data based on real time elapsed
        and current speed setting.
        
        Args:
            frame_num: frame number from FuncAnimation (unused - we track time ourselves)
        """
        # Check if we've reached the end
        if self.current_time_index >= self.n_total_frames - 1:
            return [self.particle_scatter, self.virial_indicator, 
                    self.virial_text, self.stats_text] + self.trail_collections
        
        # Calculate how much real time has passed
        current_real_time = time.time()
        real_dt = current_real_time - self.last_real_time
        self.last_real_time = current_real_time
        
        # Calculate how much simulation time should have passed
        sim_dt = real_dt * self.years_per_second
        self.accumulated_sim_time += sim_dt
        
        # Find the time index we should be at
        target_sim_time = self.all_time_points[0] + self.accumulated_sim_time
        
        # Find closest time point in data (binary search would be better but this works)
        while (self.current_time_index < self.n_total_frames - 1 and 
               self.all_time_points[self.current_time_index] < target_sim_time):
            self.current_time_index += 1
        
        # Clamp to valid range
        self.current_time_index = min(self.current_time_index, self.n_total_frames - 1)
        
        # Get data for current time point
        current_time = self.all_time_points[self.current_time_index]
        frame_data = self._get_frame_data(current_time)
        
        if frame_data is None or len(frame_data) == 0:
            return [self.particle_scatter, self.virial_indicator, 
                    self.virial_text, self.stats_text] + self.trail_collections
        
        display_frame_data = frame_data.sort_values("body_idx")
        if self.N < self.total_bodies:
            display_frame_data = display_frame_data[display_frame_data["body_idx"] < self.N]

        if len(display_frame_data) != self.N:
            return [self.particle_scatter, self.virial_indicator,
                    self.virial_text, self.stats_text] + self.trail_collections

        # Extract positions and energies
        positions_cm = display_frame_data[['x_cm', 'y_cm', 'z_cm']].values
        positions_au = positions_cm / AU
        KE = frame_data['KE'].values
        PE = frame_data['PE'].values
        
        # Update particle trails with fading effect
        self._update_trails(positions_au[:, :2])
        
        # Update current particle positions
        xy_positions = positions_au[:, :2]
        self.particle_scatter.set_offsets(xy_positions)
        self.particle_scatter.set_facecolors(self.colors)
        
        # Calculate and update virial ratio
        total_KE = np.sum(KE)
        total_PE = np.sum(PE)
        if np.abs(total_PE) > 1e-30:
            virial_ratio = np.abs(2.0 * total_KE / total_PE)
        else:
            virial_ratio = 1.0
        self._update_virial_bar(virial_ratio)
        
        # Update stats text
        fps = self._calculate_fps()
        stats_str = (f"Time: {current_time:.1f} yr\n"
                    f"Particles: {self.N}/{self.total_bodies}\n"
                    f"Speed: {self.years_per_second:.1f} yr/s\n"
                    f"Data dt: ~{self.median_frame_delta_yr:.1f} yr/frame\n"
                    f"FPS: {fps:.1f}\n"
                    f"Frame: {self.current_time_index+1}/{self.n_total_frames}")
        self.stats_text.set_text(stats_str)
        
        return [self.particle_scatter, self.virial_indicator, 
                self.virial_text, self.stats_text] + self.trail_collections
    
    def run(self):
        """
        Start the animation loop.
        """
        from matplotlib.animation import FuncAnimation
        
        # Initialize timing
        self.last_real_time = time.time()
        self.accumulated_sim_time = 0.0
        
        # Create animation with target FPS
        self.anim = FuncAnimation(self.fig, self.update_frame,
                                 interval=1000/self.target_fps,
                                 blit=True, 
                                 repeat=False,
                                 cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
