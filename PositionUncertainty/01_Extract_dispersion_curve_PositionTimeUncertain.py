import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset # type: ignore
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from itertools import combinations
from tqdm import tqdm
import os
from pathlib import Path
from scipy.stats import truncnorm
from scipy.spatial.distance import pdist
import re
from matplotlib.ticker import MaxNLocator

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_coordinates(origin, angles, distances):
    """
    Compute the coordinates of stations based on their angles and distances from a reference point.

    Parameters:
    - origin: Reference point coordinates.
    - angles: Dictionary of station angles in degrees.
    - distances: Dictionary of distances from the origin to each station.

    Returns:
    - Dictionary of station coordinates.
    """
    coords = {}
    for key, angle in angles.items():
        angle_rad = np.deg2rad(angle)
        x = origin[0] + distances[key] * np.cos(angle_rad)
        y = origin[1] + distances[key] * np.sin(angle_rad)
        coords[key] = np.array([x, y])
    return coords

def simulate_coordinates(coords, rmse, num_samples):
    """
    Simulate coordinates with Gaussian noise added to the original positions.

    Parameters:
    - coords: Dictionary of original station coordinates.
    - rmse: Root Mean Square Error for Gaussian noise.
    - num_samples: Number of random samples to generate per station.

    Returns:
    - Dictionary of simulated coordinates for each station.
    """
    rng = np.random.default_rng()
    cov = [[rmse ** 2, 0], [0, rmse ** 2]]
    simulated_coords = {
        key: rng.multivariate_normal(mean, cov, size=num_samples)
        for key, mean in coords.items()
    }
    return simulated_coords

def read_and_process_reference_stacks(stations, filter_ids, component):
    """
    Read and process reference stacks from netCDF files.

    Parameters:
    - stations: List of station names.
    - filter_ids: List of filter identifiers.
    - component: Component identifier (e.g., 'ZZ').

    Returns:
    - List of station pairs used.
    - List of raw cross-correlation data.
    - List of symmetric cross-correlation data.
    - List of time axes.
    """
    st_combi = []
    cc_dat_ref_list, cc_dat_ref_sym_list, time_list = [], [], []
    
    for st1, st2 in combinations(stations, 2):
        st_combi.append([st1, st2])
        for filter_id in filter_ids:
            path = f"./STACKS2/{filter_id:02}/REF/{component}/XA.{st1}.--_XA.{st2}.--.nc"
            with Dataset(path) as st_ref:
                cc_ref = st_ref.variables['CCF'][:]
                ax_ref = st_ref.variables['taxis'][:]
                cc_ref_sym = (cc_ref + np.flip(cc_ref)) / 2

                cc_dat_ref_list.append(cc_ref)
                cc_dat_ref_sym_list.append(cc_ref_sym)
                time_list.append(ax_ref)

    return st_combi, cc_dat_ref_list, cc_dat_ref_sym_list, time_list


def compute_peak_indices(data, min_idx, max_idx, lag, percentile):
    """
    Identify peak indices and corresponding values around the peak.

    Parameters:
    - data: Scalogram data.
    - min_idx, max_idx: Bounds for the time window.
    - lag: Lag times for the data.
    - percentile: Threshold for values around the peak.

    Returns:
    - Lists of peak indice per freq, peak value per freq, lag value at peak, 
    values and lags around the peak.
    """

    peak_indices = []
    peak_values = []
    values_around_max = []
    lags_around_max = []
    picked_lags = []

    for row in data:
        # Select the segment within the specified time window
        segment = row[min_idx:max_idx]
        segment_lag = lag[min_idx:max_idx]

        # Find the index of the maximum value in the segment
        max_idx_in_segment = np.argmax(segment)
        peak_indices.append(max_idx_in_segment)
        peak_values.append(segment[max_idx_in_segment])

        # Calculate the threshold for the specified percentile
        threshold = np.percentile(segment, percentile)

        # Now, select the continuous block around the peak index
        # Initialize the block with the peak index
        block_indices = [max_idx_in_segment]

        # Initialize counters for missing indices
        missing_indices_allowed = 1  # Allow at most one missing index
        missing_indices_left = 0
        missing_indices_right = 0

        # Expand to the left
        i = max_idx_in_segment - 1
        while i >= 0:
            if segment[i] >= threshold:
                if missing_indices_left <= missing_indices_allowed:
                    block_indices.insert(0, i)
                    i -= 1
                else:
                    break
            else:
                missing_indices_left += 1
                if missing_indices_left > missing_indices_allowed:
                    break
                i -= 1

        # Expand to the right
        i = max_idx_in_segment + 1
        while i < len(segment):
            if segment[i] >= threshold:
                if missing_indices_right <= missing_indices_allowed:
                    block_indices.append(i)
                    i += 1
                else:
                    break
            else:
                missing_indices_right += 1
                if missing_indices_right > missing_indices_allowed:
                    break
                i += 1

        values_above_threshold = segment[block_indices]
        lags_above_threshold = segment_lag[block_indices]
        values_around_max.append(values_above_threshold)
        lags_around_max.append(lags_above_threshold)

        # The actual time-lag at the peak
        picked_lags.append(segment_lag[max_idx_in_segment])


    return peak_indices, peak_values, picked_lags, values_around_max, lags_around_max  



def compute_group_velocities(dist, lag, peaks, offset):
    """
    Convert lag indices to velocities.

    Parameters:
    - dist: Distance between stations.
    - lag: Lag times.
    - peaks: Peak indices.
    - offset: Time offset for the lag values.

    Returns:
    - Group velocities
    """

    velocities = []
    for peak_idx in peaks:
        velocities.append(dist / abs(lag[peak_idx + offset]))
    return velocities

# Function to get parameters for truncnorm
def get_truncnorm_params(mean, sigma, lower_bound, upper_bound):
    """
    Calculate parameters for a truncated normal distribution.

    Parameters:
    - mean: Mean of the distribution.
    - sigma: Standard deviation of the distribution.
    - lower_bound: Lower limit of truncation.
    - upper_bound: Upper limit of truncation.

    Returns:
    - Parameters `a` and `b` for truncation.
    """
    a, b = (lower_bound - mean) / sigma, (upper_bound - mean) / sigma
    return a, b


def process_lags_and_values(picked_lags, lags, values):
    """
    Prepare lag times and values for processing.

    Ensures they are arrays, converts lag times to positive, and sorts values accordingly.

    Parameters:
    - picked_lags: Lag time at maximum amplitude.
    - lags: Lag times to process.
    - values: Amplitudes corresponding to the lag times.

    Returns:
    - Processed lag times and corresponding values.
    """
    lags = np.array(lags)
    values = np.array(values)
    if lags[0] < 0:
        lags = np.flip(np.abs(lags))
        values = np.flip(values)

    picked_lags = np.abs(picked_lags)
    return picked_lags, lags, values

    
def sample_lag_times(nf, freqs, peak_values, picked_lags, values_around, lags_around, heisenberg_box, plot_truncated=False):

    """
    Generate sampled lag times using:
      - "heisenberg": wavelet-based box for std
    Returns an array of sampled_lag_times of length nf (one per frequency).
    """

    sampled_lag_times_heisenberg = np.empty(nf)  

    for idx in range(nf):
        max_value = peak_values[idx]

        # The amplitude / lag arrays near the peak
        values_around_max = values_around[idx]
        lags_around_max = lags_around[idx]

        picked_lags, lags_around_max, values_around_max = process_lags_and_values(picked_lags, lags_around_max, values_around_max)
        # We'll interpret "mean" as the peak lag (the maximum amplitude's lag)        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Heisenberg approach
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mean_heis = picked_lags[idx]
        sigma_heis = heisenberg_box[idx]  # from wavelet scale

        if np.isnan(mean_heis) or np.isnan(sigma_heis) or sigma_heis <= 0:
            sampled_lag_times_heisenberg[idx] = np.nan
        else:
            # Truncate to [0.055 ,"infinity"], as only positive lag times are allowed and 0.06 might allow for calculing moments (mean of at most 1.5s)
            lower_bound_h = 0.055
            upper_bound_h = np.inf
            ah, bh = get_truncnorm_params(mean_heis, sigma_heis, lower_bound_h, upper_bound_h)
            rv_heis = truncnorm(ah, bh, loc=mean_heis, scale=sigma_heis)
            sampled_lag_times_heisenberg[idx] = rv_heis.rvs(size=1)[0]

            # Plot only selected frequencies
            target_freqs = np.array([3.6, 5.004, 6.956, 11.4])
            freq_val = freqs[idx]
            plotting = plot_truncated and np.any(np.isclose(freq_val, target_freqs, atol=1e-3))

            if plotting:
                x_heis = np.linspace(0, 3, 1000)
                heis_values = rv_heis.pdf(x_heis)

                plt.figure(figsize=(6, 4))
                plt.plot(lags_around_max, values_around_max * np.max(heis_values)/max_value, 'o', label="Amplitude-Scaled Original Data", color = "C0")
                plt.plot(x_heis, heis_values, label="Truncated Normal PDF", color='C1', linewidth=2.5)
                plt.title(f"Truncated Normal Distribution (heisenberg) @ {freq_val:.3f} Hz")
                plt.xlabel("Lag Time [s]")
                plt.ylabel("Probability Density")
                plt.legend()
                plt.xlim((0,3))
                plt.grid(True)
                plt.savefig(OUTPUT_DIR / f"Truncated_heis_{freq_val:.3f}.png")
                plt.close()


    return sampled_lag_times_heisenberg
  

def group_vel(cc_tr, lag, dist, neg_min, neg_max, pos_min, pos_max, sampling, fmin, fmax, nf, w0, plot_truncated=False):
    """
    Compute group velocities from cross-correlation data.

    Parameters:
    - cc_tr: Cross-correlation data.
    - lag: Lag times for the data.
    - dist: Distance between stations.
    - neg_min, neg_max: Bounds for the negative lag time window.
    - pos_min, pos_max: Bounds for the positive lag time window.
    - sampling: Sampling rate of the data.
    - fmin, fmax: Frequency range for the calculation.
    - nf: Number of frequency steps.
    - w0: Wavelet parameter (replaces the fixed integer previously used in cwt).

    Returns:
    - Results dependent on the flag `all`, including group velocities, errors, and scalograms.
    """
    dt = 1 / sampling  # Time step

    # Compute the scalogram with continuous wavelet transform using w0
    scalogram, scales = cwt(cc_tr, dt, w0, fmin, fmax, nf)
    scal = np.abs(scalogram)**2

    # For Morlet wavelet, std(t) is 1/np.sqrt(2)
    heisenberg_box = scales * 1/np.sqrt(2)
    filename_heis = OUTPUT_DIR / f"heisenberg_std_w{w0}.npy"
    if not filename_heis.exists():
        np.save(filename_heis, heisenberg_box)

    
    # Frequency axis
    freq2 = np.logspace(np.log10(fmin), np.log10(fmax), scalogram.shape[0])
    
    
    # Compute peak indices and values around the maximum to find Gaussian fit (on negative time only)
    peak_indices_neg, peak_values_neg, picked_lags_neg, values_around_max_neg, lags_around_max_neg = compute_peak_indices(scal, neg_min, neg_max, lag, percentile=20)

    filename_lag = OUTPUT_DIR / f"true_mean_lag_w{w0}.npy"
    if not filename_lag.exists():
        np.save(filename_lag, np.abs(np.array(picked_lags_neg)))

    sampled_lag_times_heisenberg = sample_lag_times(
        nf, freq2, peak_values_neg, picked_lags_neg, 
        values_around_max_neg, lags_around_max_neg, 
        heisenberg_box,
        plot_truncated=plot_truncated
    )

    # for each distance, I get for each frequency x time lag samples
    group_vel_neg_heisenberg = dist / sampled_lag_times_heisenberg 

    # Scalograms for visualization
    scalogram_neg, _ = cwt(cc_tr[:neg_max], dt, w0, fmin, fmax, nf)
    scalogram_scal_neg = np.abs(scalogram_neg)**2
    
    return freq2, scalogram_scal_neg, group_vel_neg_heisenberg, sampled_lag_times_heisenberg


# Helper function to check if the simulation result file exists and load it
def load_existing_simulation(result_file, num_samples, description):
    if os.path.isfile(result_file):
        print(f"Simulation with {num_samples} {description} samples already performed")
        return np.load(result_file, allow_pickle=True)
    return None

def ensure_support_files(cc_tr, lag, dist, neg_min, neg_max, pos_min, pos_max, sampling, fmin, fmax, nf, w0):
    """
    Make sure auxiliary outputs (true_mean_lag, heisenberg_std) exist even when
    we reuse cached simulation results.
    """
    heis_file = OUTPUT_DIR / f"heisenberg_std_w{w0}.npy"
    lag_file = OUTPUT_DIR / f"true_mean_lag_w{w0}.npy"
    if heis_file.exists() and lag_file.exists():
        return

    # Run a single group_vel pass to populate the missing files
    group_vel(
        cc_tr=cc_tr,
        lag=lag,
        dist=dist,
        neg_min=neg_min,
        neg_max=neg_max,
        pos_min=pos_min,
        pos_max=pos_max,
        sampling=sampling,
        fmin=fmin,
        fmax=fmax,
        nf=nf,
        w0=w0
    )

# plot dispersion curves
def plot_dispersion_curves(i, freq, fmin, fmax, group_vel_neg_lagUncertain_heisenberg, w0, label):
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(7,5), constrained_layout=True)
    plt.errorbar(
        freq,
        np.mean(group_vel_neg_lagUncertain_heisenberg[:i, :], axis=0),
        np.std(group_vel_neg_lagUncertain_heisenberg[:i, :], axis=0),
        color="red",
        alpha=0.6,
        label=label
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Group Velocity (m/s)")
    plt.xlim(fmin, fmax)
    plt.ylim(25,55)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.legend()
    plt.savefig(OUTPUT_DIR / f"group_dis_w{w0}.png", dpi = 1000)
    plt.close()

    
#plot scalogram with dispersion curve for negative lag time
def plot_scalogram_with_dispersion(i, freq, time, dist, neg_max, scalogram_scal_neg, fmin, fmax, obspy_sequential, group_vel_neg_lagUncertain_heisenberg, w0):
    vel_ax = dist / np.abs(time[:neg_max])
    freq_ax, vel_ax2 = np.meshgrid(freq, vel_ax)

    color_val = np.transpose(scalogram_scal_neg)

    plt.rcParams.update({'font.size': 23})
    # create figure & axes
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    # draw the scalogram; pass through vmin/vmax
    pcm = ax.pcolormesh(
        freq_ax,
        vel_ax2,
        color_val,
        cmap=obspy_sequential,
        vmin=0,
        vmax=1.8e-17
    )

    # overlay the mean group velocity
    mean_curve = np.mean(group_vel_neg_lagUncertain_heisenberg[:i, :], axis=0)
    ax.plot(freq, mean_curve, color="red", alpha=0.6, linewidth = 2.2, label=rf"$\omega_0$ = {w0}")

    # axes labels and limits
    ax.set_xlabel("Frequency (Hz)", fontsize=23)
    ax.set_ylabel("Group Velocity (m/s)", fontsize=23)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(25, 55)

    # tick formatting
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    # add the colorbar
    cbar_label="Scalogram amplitude"
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=23)
    cbar.ax.tick_params(labelsize=23)

    ax.legend(loc='upper right', fontsize=23)

    # save and close
    plt.savefig(OUTPUT_DIR / f"scalogram_w{w0}.png", dpi=1000)
    plt.close(fig)
     

def main():
    # -----------------------------
    # Runtime user prompts
    # -----------------------------
    # Ask whether plots should be produced
    plot_input = input("Produce plots? (y/n): ").strip().lower()
    if plot_input == "y":
        dispersion_with_plots = True
    else:
        dispersion_with_plots = False

    # Ask for simulation mode: 'time' for only time simulation, 'pos' for only pos simulation, or 'both' for neither flag true
    sim_mode = input("Enter simulation mode ('time' for only time simulation, 'pos' for only pos simulation, or 'both' for both): ").strip().lower()
    if sim_mode == "time":
        only_time_simulation = True
        only_pos_simulation = False
    elif sim_mode == "pos":
        only_time_simulation = False
        only_pos_simulation = True
    elif sim_mode == "both":
        only_time_simulation = False
        only_pos_simulation = False
    else:
        print("Invalid simulation mode. Defaulting to 'both'.")
        only_time_simulation = False
        only_pos_simulation = False

    # Ask for wavelet parameter w0 (default is 6)
    w0_input = input("Enter w0 value (default 6): ").strip()
    if w0_input:
        try:
            if re.match(r'^\d+$', w0_input):
                w0 = int(w0_input)
            else:
                w0 = float(w0_input)
        except ValueError:
            print("Invalid input for w0, using default 6")
            w0 = 6
    else:
        w0 = 6


    # Parameters
    origin = np.array([100, 100])
    angles = {'G1': 0, 'G2': 122.88, 'G3': 0, 'G4': 122.88 + 122.38}
    distances = {'G1': 57.5, 'G2': 56.4, 'G3': 0., 'G4': 56.9}
    rmse = 0.9
    num_samples = 100000
    stations = ["G1", "G2", "G3", "G4"]
    filter_ids = [1]
    component = "ZZ"
    neg_min, neg_max = 13390, 13730
    pos_min, pos_max = 13870, 14210
    sampling = 115
    fmin, fmax = 3.6, 11.4
    nf = 50
    dist = 56.9

    assert not (only_pos_simulation and only_time_simulation), "Both only_pos_simulation and only_time_simulation cannot be true at the same time"

    # Step 1: Compute coordinates
    if only_time_simulation:
        simulated_coords = {}
    else:
        coords = compute_coordinates(origin, angles, distances)
        simulated_coords = simulate_coordinates(coords, rmse, num_samples)

    # Step 2: Read and process reference stacks
    st_combi, cc_dat_ref_list, _, time_list = read_and_process_reference_stacks(stations, filter_ids, component)

    # Reference stack for G3-G4
    ind = 5
    stations_used = st_combi[ind]
    print("Using reference stack for station pair ", stations_used)
    time = time_list[ind]
    cc_dat_ref = cc_dat_ref_list[ind]

    # Step 3: Compute dispersion curves
    # check if file exist, otherwise rerun simulation
    if only_pos_simulation:
        label_dispersion = "Only localization (std)"
        result_file = OUTPUT_DIR / f"neg_group_vel_p_{num_samples}_w{w0}.npy"
        existing_data = load_existing_simulation(result_file, num_samples, "distance only")
        if existing_data is not None:
            group_vel_neg_lagUncertain = existing_data
            if dispersion_with_plots:
                i = num_samples
                freq = np.load(OUTPUT_DIR / "freqs.npy")
                plot_dispersion_curves(i, freq, fmin, fmax, group_vel_neg_lagUncertain, w0, label_dispersion)
                # scalograms are not saved, therefore get one once
                results = group_vel(
                    cc_tr=cc_dat_ref,
                    lag=time,
                    dist=dist,
                    neg_min=neg_min,
                    neg_max=neg_max,
                pos_min=pos_min,
                pos_max=pos_max,
                sampling=sampling,
                fmin=fmin,
                fmax=fmax,
                nf=nf,
                w0=w0,
                plot_truncated=False
            )

                _, scalogram_scal_neg, _, _ = results
                plot_scalogram_with_dispersion(i, freq, time, dist, neg_max, scalogram_scal_neg, fmin, fmax, obspy_sequential, group_vel_neg_lagUncertain, w0)
            return
        
        coords_g3 = simulated_coords["G3"]
        coords_g4 = simulated_coords["G4"]

        # Compute the pairwise distances between all points in G3 and G4
        station_dist = np.linalg.norm(coords_g3 - coords_g4, axis=1)

        dt = 1 / sampling  # Time step

        # Compute the scalogram with continuous wavelet transform using w0
        scalogram, _ = cwt(cc_dat_ref[:neg_max], dt, w0, fmin, fmax, nf)
        scal = np.abs(scalogram)**2
        # Frequency axis
        freq2 = np.logspace(np.log10(fmin), np.log10(fmax), scalogram.shape[0])
    
        _, _, picked_lags_neg, _, _ = compute_peak_indices(scal, neg_min, neg_max, time, percentile=95)

        group_vel_neg_lagUncertain = station_dist[:, np.newaxis] / np.transpose(np.abs(picked_lags_neg))[np.newaxis,:]

        ensure_support_files(
            cc_tr=cc_dat_ref,
            lag=time,
            dist=dist,
            neg_min=neg_min,
            neg_max=neg_max,
            pos_min=pos_min,
            pos_max=pos_max,
            sampling=sampling,
            fmin=fmin,
            fmax=fmax,
            nf=nf,
            w0=w0
        )

        np.save(OUTPUT_DIR / "freqs.npy", freq2)
        np.save(result_file, group_vel_neg_lagUncertain)

        if dispersion_with_plots:
            plot_dispersion_curves(num_samples, freq2, fmin, fmax, group_vel_neg_lagUncertain, w0, label_dispersion)
            plot_scalogram_with_dispersion(num_samples, freq2, time, dist, neg_max, scal, fmin, fmax, obspy_sequential, group_vel_neg_lagUncertain, w0)
    else:
        if only_time_simulation:
            result_file = OUTPUT_DIR / f"neg_group_vel_t_{num_samples}_w{w0}.npy"
            description = "time only"
            label_dispersion = "Only time lag (std)"
        else:
            result_file = OUTPUT_DIR / f"neg_group_vel_pt_{num_samples}_w{w0}.npy"
            description = "distance and time"
            label_dispersion = "Combined (std)"
            
        existing_data = load_existing_simulation(result_file, num_samples, description)
        if existing_data is not None:
            group_vel_neg_lagUncertain_heisenberg, sampled_lag_times_heisenberg = existing_data
            if dispersion_with_plots:
                i = num_samples
                freq = np.load(OUTPUT_DIR / "freqs.npy")
                plot_dispersion_curves(i, freq, fmin, fmax, group_vel_neg_lagUncertain_heisenberg, w0, label_dispersion)
                # scalograms are not saved, therefore get one once
                station_dist = dist
                results = group_vel(
                    cc_tr=cc_dat_ref,
                    lag=time,
                    dist=station_dist,
                    neg_min=neg_min,
                    neg_max=neg_max,
                pos_min=pos_min,
                pos_max=pos_max,
                sampling=sampling,
                fmin=fmin,
                fmax=fmax,
                nf=nf,
                w0=w0,
                plot_truncated=(dispersion_with_plots and only_time_simulation)
            )

                _, scalogram_scal_neg, _, _ = results
                plot_scalogram_with_dispersion(i, freq, time, dist, neg_max, scalogram_scal_neg, fmin, fmax, obspy_sequential, group_vel_neg_lagUncertain_heisenberg, w0)
            return
        
        group_vel_neg_lagUncertain_heisenberg = np.zeros((num_samples, nf))
        sampled_lag_times_heisenberg = np.zeros((num_samples, nf))
        
        for i in tqdm(range(num_samples)):
            if not only_time_simulation:
                coords_array_rand = np.vstack([simulated_coords["G3"][i, :], simulated_coords["G4"][i, :]])
                station_dist = pdist(coords_array_rand)
            else: 
                station_dist = dist
            
            results = group_vel(
                cc_tr=cc_dat_ref,
                lag=time,
                dist=station_dist,
                neg_min=neg_min,
                neg_max=neg_max,
                pos_min=pos_min,
                pos_max=pos_max,
                sampling=sampling,
                fmin=fmin,
                fmax=fmax,
                nf=nf,
                w0=w0,
                plot_truncated=(dispersion_with_plots and only_time_simulation)
            )
            
            freq, scalogram_scal_neg, group_vel_neg_heisenberg_tmp, sampled_lag_times_heisenberg_tmp = results

            if i == 0: 
                np.save(OUTPUT_DIR / "freqs.npy", freq)

            group_vel_neg_lagUncertain_heisenberg[i, :] = group_vel_neg_heisenberg_tmp
            sampled_lag_times_heisenberg[i, :] = sampled_lag_times_heisenberg_tmp

            # This will create plots in each iteration, so quite expensive
            if dispersion_with_plots:
                plot_dispersion_curves(i, freq, fmin, fmax, group_vel_neg_lagUncertain_heisenberg, w0, label_dispersion)
                plot_scalogram_with_dispersion(i, freq, time, dist, neg_max, scalogram_scal_neg, fmin, fmax, obspy_sequential, group_vel_neg_lagUncertain_heisenberg, w0)

        
        np.save(result_file, [group_vel_neg_lagUncertain_heisenberg, sampled_lag_times_heisenberg])    
        

if __name__ == "__main__":
    main()
