#!/bin/bash

# Folder for this process
PROCESS_FOLDER=$1

# Navigate to the process folder
cd "$PROCESS_FOLDER" || exit 1

# Activate the Anaconda environment (abort if conda cannot be initialized)
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" || { echo "Error: failed to initialize conda." >&2; exit 1; }
    conda activate PassiveSeismic || { echo "Error: failed to activate conda env PassiveSeismic." >&2; exit 1; }
else
    echo "Error: conda not found. Aborting stacking." >&2
    exit 1
fi

# Check if the db.ini and db.sqlite files exist and remove them
if [ -f "db.ini" ]; then
    rm -rf db.ini
fi

if [ -f "db.sqlite" ]; then
    rm -rf db.sqlite
fi

if [ -d "STACKS" ]; then
    rm -rf STACKS
fi

if [ -d "STACKS2" ]; then
    rm -rf STACKS2
fi

# Initialize and configure msnoise
msnoise db init --tech=1
msnoise config set data_folder=$PROCESS_FOLDER
msnoise config set startdate=1976-09-19
msnoise config set enddate=1976-10-19

# Scan archive and populate database
msnoise scan_archive --init --path $PROCESS_FOLDER
msnoise populate --fromDA

# Initialize new jobs
msnoise new_jobs --init

# Insert custom filter into the database
msnoise db execute "insert into filters (ref, low, mwcs_low, high, mwcs_high, mwcs_wlen, mwcs_step, used) values (1, 3, 3, 30, 30, 2.0, 1.0, 1)"

# Reset various stages of processing
msnoise reset CC --all
msnoise reset STACK --all
msnoise reset MWCS --all
msnoise reset DTT --all

# Configure msnoise for processing
msnoise config set resampling_method=Lancoz
msnoise config set cc_sampling_rate=115
msnoise config set preprocess_highpass=1
msnoise config set preprocess_lowpass=40
# ! msnoise config set stack_method='pws'

# Compute cross-correlation
msnoise -v cc compute_cc 

# Sync the config and stack cross-correlation results
msnoise config sync
msnoise cc stack -r

# delete data that is not necessary for dispersion curve calculation
rm -rf XA.*
rm -rf STACKS
rm -rf CROSS_CORRELATIONS