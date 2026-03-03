## Evaluating Multi-Agent and Wavelet-Transform Uncertainties in Lunar Seismic Ambient Noise Exploration
This repository contains the code that was used for the study on how uncertainties in multi-agent systems and the processing pipeline propagates into the results of ambient noise interferometry ([Citation info](CITATION.cff)):

Passive seismic ambient noise interferometry (ANI) has shown potential for lunar seismic exploration, offering the capability to detect near-surface subsurface structures critical for future lunar mission, such as near-surface ice deposits and lava tubes, without the need for active seismic sources. Performing ANI on the Moon can be realized with a multi-agent system, in which a network of individual rovers either carry or deploy seismic receivers. However, these systems have inherent uncertainties in localization and timing. Additionally, methods used to extract dispersion curves from cross-correlations are fundamentally limited in achievable time–frequency resolution, which we demonstrate for the continuous wavelet transform (CWT). Quantifying how these factors propagate into Rayleigh wave velocity estimates is essential for accurate detection of lunar subsurface features. In this study, analytical error formulas are derived and validated through Monte-Carlo simulations using passive seismic data from the Apollo 17 Lunar Seismic Profiling Experiment (LSPE). 

### Data aquisition
The decoded CSV data files were downloaded from the Data Archives and Transmission System (DARTS): https://darts.isas.jaxa.jp/app/apollo/. Individual CSV files (each containing up to 6 hours of data) have to be merged into daily continuous streams and converted to MiniSEED format using the python toolbox ObsPy. No additional pre-processing is applied during this stage. The daily files for the 4 seismometers shall be placed in `PositionUncertainty/DATA/MSEED` with format `XA.GP{station_number}..Z.1976.{day_number}`.

### Environment
Use conda to create the environment (will also install pip-only deps such as msnoise from the `pip:` section):
```bash
conda env create -f environment_new.yml
conda activate PassiveSeismic
```

### Repository layout (current)
```
.
├── ClockInstability/
│   ├── 01_parallel_clockDrift_completeWorkflow.py              # main file for clock instability impact simulation             
│   ├── 02_plotting_results.ipynb                               # visualization of results
│   ├── stacking.sh                                             # script called in 01_interferometry_moon.ipynb to stack modified traces
│   └── output/                                                 # simulation results
│       └── samples/                                            # stacks from the MC simulation (netCDF)
├── PositionUncertainty/
│   ├── 00_interferometry_moon.ipynb                            # stacking using msnoise
│   ├── 01_Extract_dispersion_curve_PositionTimeUncertain.py    # main file
│   ├── 02_plotting_results.ipynb                               # visualization of results
│   ├── 03_ANI_Overview.ipynb                                   # visualization of ANI process
│   ├── output/                                                 # simulation results and plots
│   ├── STACKS2/                                                # input stacks (netCDF)
│   └── DATA/                                                   # raw data (MSEED)
```


### Usage
The [00_interferometry_moon.ipynb](PositionUncertainty/00_interferometry_moon.ipynb) notebook that should be run first to create the stacks used for the dispersion analysis. As in the paper, all the following code will only use the station pair G1, G4 with a hard-coded distance.

The impact of UBW clock uncertainties under the assumption of perfect syncronization every 50 seconds can be simulated with [01_parallel_clockDrift_completeWorkflow.py](ClockInstability/01_parallel_clockDrift_completeWorkflow.py). This script runs parallel Monte Carlo draws, the number of which can be set via `max_processes` in `main()`. Results are visualized with
[02_plotting_results.ipynb](ClockInstability/02_plotting_results.ipynb).

The impact of localization errors, CWT-uncertainty, and a combination of both can be evaluated using [01_Extract_dispersion_curve_PositionTimeUncertainty.py](PositionUncertainty/01_Extract_dispersion_curve_PositionTimeUncertainty.py). Running it in the terminal will ask for user prompts. The results can be visualized with [02_plotting_results.ipynb](PositionUncertainty/02_plotting_results.ipynb).

[03_ANI_Overview.ipynb](PositionUncertainty/03_ANI_Overview.ipynb) reproduces the figures used to explain the concept of ANI in the paper.

### Citing
If you use this code, please cite the associated paper for the scientific method and results, and cite this repository for the software implementation.

```
@article{NierulaPaper2026,
  author = {Nierula, Kai and Keil, Sabrina and Shutin, Dmitriy and Shin, Ban-Sok and Igel, Heiner},
  title = {Evaluating Multi-Agent and Wavelet-Transform Uncertainties in Lunar Seismic Ambient Noise Exploration},
  journal = {Earth and Space Science},
  volume = {13},
  number = {2},
  pages = {e2025EA004631},
  doi = {https://doi.org/10.1029/2025EA004631},
  year = {2026}
}
```

```
@misc{NierulaCode2025,
  author = {Nierula, Kai and Keil, Sabrina},
  title = {Code for Evaluating Multi-Agent and Wavelet-Transform Uncertainties in Lunar Seismic Ambient Noise Exploration},
  publisher = {Deutsches Zentrum für Luft- und Raumfahrt e. V. (DLR)},
  doi = {10.26090/CKHM-HN79},
  url = {https://github.com/DLR-KN/LunarANI-UQ},
  year = {2025},
  type = {Software}
}
```
