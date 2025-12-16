import os
from pathlib import Path
import numpy as np
from obspy import read, Stream, Trace
from obspy.signal.tf_misfit import cwt
from itertools import combinations
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from netCDF4 import Dataset # type: ignore

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"  # final npy/png outputs for clock workflow
SAMPLES_DIR = BASE_DIR / "output" / "samples" # per-sample modified MSEED + stacks
INPUT_DIR = BASE_DIR.parent / "PositionUncertainty" / "DATA" / "MSEED" # raw MSEED input

OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

########## Using realistic 2-state clock model with drift from Trainotti et al, 2019 #######################

# ---------------------------------------------------------------------
def phase_deviation(sigma1_sq,
                    sigma2_sq,
                    drift,
                    npts,
                    dt,                # sampling period [s]
                    sync_tau,
                    rng = np.random.default_rng()):
    """
    Simulate the two-state clock over `npts` EQUALLY-SPACED samples
    (spacing = `dt`) while resetting τ and f to zero every `sync_tau`
    seconds.  Returns a 1-D NumPy array of length `npts`
    containing the phase deviation at every sample.
    """
    steps_per_block = int(round(sync_tau / dt))
    n_blocks        = int(np.ceil(npts / steps_per_block))

    # --- draw ALL noise vectors once ----------------------------------
    Q = np.array([[sigma1_sq * dt + sigma2_sq * dt**3 / 3.0,
                    sigma2_sq * dt**2 / 2.0],
                    [sigma2_sq * dt**2 / 2.0,
                    sigma2_sq * dt]])

    w = rng.multivariate_normal([0.0, 0.0], Q,
                                size=n_blocks * steps_per_block).T   # shape (2, …)

    # pre-allocate output
    phase = np.empty(n_blocks * steps_per_block)
    freq  = np.empty_like(phase)

    # --- vectorised work inside each 50-s block -----------------------
    idx0 = 0
    for _ in range(n_blocks):
        idx1 = idx0 + steps_per_block
        w1, w2 = w[:, idx0:idx1]                     # slice of length m

        # frequency FIRST  : f_{k+1} = f_k + drift·dt + w2_k
        freq_inc  = drift * dt + w2
        freq[idx0:idx1] = np.cumsum(freq_inc)

        # phase increments : τ_{k+1} = τ_k + dt·f_k + ½·d·dt² + w1_k
        phase_inc = dt * np.concatenate(([0.0], freq[idx0:idx1-1])) \
                    + 0.5 * drift * dt**2 + w1
        phase[idx0:idx1] = np.cumsum(phase_inc)

        # perfect sync ⇒ next block starts from τ = f = 0
        idx0 = idx1

    # trim to exactly npts in case the last block was partial
    return phase[:npts]


# Function for dispersion curve
def group_vel(cc_tr, lag, dist, neg_min, neg_max, pos_min, pos_max, sampling, fmin, fmax, nf, w0):
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
    """
    
    dt = 1/sampling
    
    #compute scalogram with continuous wavelet transform
    scalogram, scales = cwt(cc_tr, dt, w0, fmin, fmax, nf) 
    scal = np.abs(scalogram)**2

    freq2=np.logspace(np.log10(fmin), np.log10(fmax),scalogram.shape[0])

    t_max_neg=[]
    for n in range(len(scal)):
        dat=scal[n][neg_min:neg_max]
        indx=np.argmax(dat)
        t_max_neg.append(indx)


    group_vel_neg_ref=[]
    for n in range(len(t_max_neg)):
        val=dist/abs(lag[t_max_neg[n]+neg_min])
        group_vel_neg_ref.append(val)
    

    return freq2, group_vel_neg_ref
    

# Function to calculate dispersion curve
def calculate_dispersion_curve(stations, sample_output_dir: Path):
    # Set parameters for the dispersion curve calculation
    filterid = [1]
    comp = "ZZ"
    st_ref_list = []
    st_combi = []
    # Build station combinations and load reference stacks
    for st1, st2 in combinations(stations, 2):
        st_combi.append([st1, st2])
        for filter in filterid:
            path = sample_output_dir / f"STACKS2/{filter:02}/REF/{comp}/XA.{st1}.--_XA.{st2}.--.nc"
            st_ref_tmp = Dataset(path)
            st_ref_list.append(st_ref_tmp)

    cc_ref_list = []
    ax_ref_list = []
    time_list = []
    cc_dat_ref_list = []
    cc_dat_ref_sym_list = []
    
    # Extract cross-correlation function and time axes
    for i in range(len(st_ref_list)):
        cc_ref_list.append(st_ref_list[i].variables['CCF'])
        ax_ref_list.append(st_ref_list[i].variables['taxis'])
        cc_dat_ref_list.append(cc_ref_list[i][:])
        cc_dat_ref_sym_list.append(cc_ref_list[i][:])
        cc_dat_ref_sym_list[i] = (cc_dat_ref_sym_list[i] + np.flip(cc_dat_ref_sym_list[i])) / 2
        time_list.append(ax_ref_list[0][:])

    # Use the first combination for this example
    ind = 0
    time = time_list[ind]
    cc_dat_ref = cc_dat_ref_list[ind]
    sampling = 115  # Depends on your sampling rate during CC calculation
    w0 = 8.0
    fmin, fmax = 3.6, 11.4
    nf = 50
    dist = 56.9

    # Time windows for dispersion curve extraction
    neg_min = 13390
    neg_max = 13730
    pos_min = 13870
    pos_max = 14210

    
    # Compute the dispersion curve (using the group_vel function you already have)
    freq, group_vel_neg = group_vel(
        cc_tr=cc_dat_ref, lag=time, dist=dist,
        neg_min=neg_min, neg_max=neg_max, pos_min=pos_min, pos_max=pos_max,
        sampling=sampling, fmin=fmin, fmax=fmax, nf=nf, w0=w0
    )
    
    return freq, group_vel_neg


# Define the task to be parallelized
def process_sample(i, sample_len, files_gp3, files_gp4, base_dir: Path, input_directory: Path, samples_dir: Path, output_dir: Path, synchronization_interval, sigma1_sq, sigma2_sq, drift):
    print(f"Processing sample {i} out of {sample_len-1}")
    rng = np.random.default_rng(seed=i) # Create a random number generator instance

    sample_output_dir = samples_dir / f"sample_{i}"
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Loop through the files
    for file_gp3, file_gp4 in zip(files_gp3, files_gp4):
        file_path_gp3 = input_directory / file_gp3
        file_path_gp4 = input_directory / file_gp4
        
        # Read the MSEED files
        st_gp3 = read(file_path_gp3)
        st_gp4 = read(file_path_gp4)

        # if either stream is empty, skip
        if not st_gp3 or not st_gp4:
            print("Empty stream! Check!")
            continue

        # use the first trace as our template for stats and start_time when creating the modified streams
        template_tr3 = st_gp3[0]
        template_tr4 = st_gp4[0]
        start_time_tr3   = template_tr3.stats.starttime
        start_time_tr4   = template_tr4.stats.starttime
        if not (start_time_tr3 == start_time_tr4):
            print("Start times of streams are not equal. Interpolation might return wrong results")
            continue

        modified_data_gp3 = []  # To store modified traces for GP3
        modified_data_gp4 = []  # To store modified traces for GP4

        drift_gp3_total = []  # To store total drift for GP3 for plotting
        drift_gp4_total = []  # To store total drift for GP4 for plotting

        # Process each trace
        for tr_gp3, tr_gp4 in zip(st_gp3, st_gp4):
            start_time = tr_gp4.stats.starttime
            end_time = tr_gp4.stats.endtime
            current_time = start_time
            
            while current_time < end_time:
                interval_end = current_time + synchronization_interval
                if interval_end > end_time:
                    interval_end = end_time

                gp3_slice = tr_gp3.slice(current_time, interval_end)
                gp4_slice = tr_gp4.slice(current_time, interval_end)

                if gp3_slice.stats.npts == 0 or gp4_slice.stats.npts == 0:
                    break

                # Simulate clock instability for both GP3 and GP4
                npts = len(gp3_slice.times())
                sampling_rate_original = gp3_slice.stats.sampling_rate 
                dt               = 1.0 / sampling_rate_original  #dt for clock simulation
                drift_gp3 = phase_deviation(sigma1_sq, sigma2_sq, drift, npts, dt, synchronization_interval, rng)
                drift_gp4 = phase_deviation(sigma1_sq, sigma2_sq, drift, npts, dt, synchronization_interval, rng)

                # Append drift for plotting
                drift_gp3_total.extend(drift_gp3)
                drift_gp4_total.extend(drift_gp4)

                # Interpolate to synchronize drifted data
                gp3_resampled = np.interp(gp3_slice.times(), gp3_slice.times() + drift_gp3, gp3_slice.data)
                gp4_resampled = np.interp(gp3_slice.times(), gp3_slice.times() + drift_gp4, gp4_slice.data)

                modified_data_gp3.append(gp3_resampled)
                modified_data_gp4.append(gp4_resampled)

                current_time = interval_end + 1 / gp3_slice.stats.sampling_rate

        # Combine all modified data into single arrays
        all_modified_data_gp3 = np.concatenate(modified_data_gp3)
        all_modified_data_gp4 = np.concatenate(modified_data_gp4)

        # Create a single modified trace for GP3 and GP4
        modified_trace_gp3 = Trace()
        modified_trace_gp3.data = all_modified_data_gp3.astype(np.float64)
        modified_trace_gp3.stats = template_tr3.stats.copy()
        modified_trace_gp3.stats.starttime = start_time_tr3
        modified_trace_gp3.stats.mseed['encoding'] = 'FLOAT64'

        modified_trace_gp4 = Trace()
        modified_trace_gp4.data = all_modified_data_gp4.astype(np.float64)
        modified_trace_gp4.stats = template_tr4.stats.copy()
        modified_trace_gp4.stats.starttime = start_time_tr4
        modified_trace_gp4.stats.mseed['encoding'] = 'FLOAT64'

        # Create streams with the modified traces
        modified_stream_gp3 = Stream(traces=[modified_trace_gp3])
        modified_stream_gp4 = Stream(traces=[modified_trace_gp4])

        # Save the modified Stream to new MSEED files
        output_path_gp3 = sample_output_dir / file_gp3
        output_path_gp4 = sample_output_dir / file_gp4
        modified_stream_gp3.write(str(output_path_gp3), format="MSEED")
        modified_stream_gp4.write(str(output_path_gp4), format="MSEED")
        
        print(f"Processed {file_gp3} and {file_gp4} for sample {i}")

    ########## Step 2: Run external script for stacking ##########
    print(f"Started stacking for sample {i}")
    bash_script_path = os.path.join(base_dir, "stacking.sh")
    result = subprocess.run(f"{bash_script_path} {sample_output_dir}", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in sample {i}: {result.stderr}")


    ########## Step 3: Calculate dispersion curve ##########
    print(f"Started calculating dispersion curve for sample {i}")
    stations = ["G3", "G4"]
    freq, group_vel_neg_clock = calculate_dispersion_curve(stations, sample_output_dir)

    ########## Step 4: Save dispersion curve with index ##########
    np.save(output_dir / f"neg_group_vel_clock_{i}.npy", group_vel_neg_clock)
    np.save(output_dir / f"sampled_freqs_{i}.npy", freq)

    print(f"Stacked and calculated dispersion curve for sample {i}")

    return i


def aggregate_results(sample_len, output_directory: Path, clock_type, sync_interval):
    all_results = []

    # Loop through all saved .npy files and aggregate them
    for i in range(sample_len):
        result = np.load(output_directory / f"neg_group_vel_clock_{i}.npy")
        all_results.append(result)

    # Convert list to numpy array and save the final aggregated .npy file in the output directory
    final_results = np.array(all_results)
    np.save(output_directory / f"neg_group_vel_{sample_len}_clock_{clock_type}_{sync_interval}sInterval.npy", final_results)



def main():
    # Number of samples
    sample_len = 1000

    # Paths
    base_dir = BASE_DIR
    input_directory = INPUT_DIR
    samples_dir = SAMPLES_DIR
    output_directory = OUTPUT_DIR

    # Parameters for the 2-state clock model with drift from Trainotti et al. and 
    # SELF-ORGANIZED ULTRA-WIDE BAND LOCALIZATION FOR SWARM (Master's thesis)
    sigma1_sq = 1.02e-15  # Variance of phase noise (σ1^2)
    sigma2_sq = 1.29e-18  # Variance of frequency noise (σ2^2)
    drift = 1.47e-22  # Constant drift term (d)
    clock_type = "uwb"

    # Find and separate files for each station
    files_gp3 = sorted([f for f in os.listdir(input_directory) if 'GP3' in f])
    files_gp4 = sorted([f for f in os.listdir(input_directory) if 'GP4' in f])

    synchronization_interval = 50 #s
   
    # Parallel loop to process each sample
    max_processes = 25
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = [executor.submit(process_sample, i, sample_len, files_gp3, files_gp4, base_dir, 
                                   input_directory, samples_dir, output_directory, synchronization_interval,
                                   sigma1_sq, sigma2_sq, drift) for i in range(sample_len)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                print(f"Sample {result + 1} completed")
    
    # Aggregate results after parallel execution is complete
    aggregate_results(sample_len, output_directory, clock_type, synchronization_interval)

if __name__ == "__main__":
    main()
