# SingleCapillaryReporters

A MATLAB tool for detecting and analyzing capillary segments in ultrasound particle tracking data using Hidden Markov Models (HMM).

## Overview

SingleCapillaryReporters (SCaRe) automatically identifies capillary segments from ultrasound particle tracks by analyzing velocity patterns and spectral features. The tool uses an HMM-based approach to distinguish between arteriole, capillary, and venule regions, then generates dwell time maps for visualization.

## Features

- **Automated capillary detection** using HMM with Viterbi algorithm
- **Velocity-based state classification** (flowing vs. non-flowing regions)
- **Spectral feature extraction** from acceleration signals
- **Dwell time map generation** from identified capillary tracks
- **Visualization tools** for 3D track display and dwell map overlay

## Requirements

- MATLAB (tested on R2020b or later)
- Signal Processing Toolbox (for `spectrogram`, `smooth`, `mad`)
- Statistics and Machine Learning Toolbox (for `hmmtrain`, `hmmviterbi`)

## Installation

1. Clone or download this repository:
   ```bash
   git clone https://github.com/provostultrasoundlab/SingleCapillaryReporters.git
   cd SingleCapillaryReporters
   ```

2. Ensure you have the required MATLAB toolboxes installed.

## Quick Start

Run the example script to process sample data:

```matlab
example
```

This will:
1. Load particle tracking data from the `data/` directory
2. Train an HMM on all tracks
3. Identify capillary segments
4. Display tracks with highlighted capillaries
5. Generate and display a dwell time map

## Usage

### Basic Workflow

```matlab
% Add source directory to path
addpath(genpath('src'));

% Configure parameters
config.numStates = 2;                      % HMM states (1: non-cap, 2: cap)
config.numObservations = 4;                 % Observation symbols
config.transProbs = [0.9 0.1; 0.1 0.9];    % Initial transition probabilities
config.emissionProbs = [0.45 0.45 0.05 0.05; 0.05 0.05 0.45 0.45];
config.cap_thresh = 0.1;                    % Min capillary length [s]
config.vel_thresh = 1;                      % Velocity threshold [mm/s]
config.grid_spacing = 1.7523e-5;            % Grid spacing [m]
config.PRF = 1000;                          % Pulse repetition frequency [Hz]
config.tracks = load('data/LongEnsemble_1.mat').tracks;

% Initialize and run analysis
SCaRe = SingleCapillaryReporter(config);
SCaRe.train();                              % Train HMM
SCaRe.analyze();                            % Detect capillaries
SCaRe.display_capillaries();                % Visualize results
```

### Input Data Format

Tracks should be a cell array where each cell contains an Nx4 matrix:
- Column 1: x position (in grid units)
- Column 2: y position (in grid units)
- Column 3: frame number
- Column 4: z position (in grid units)

### Output

The `analyze()` method populates `SCaRe.capillaries` with:
- `iscap`: Classification vector (0 = none, 1 = full capillary, 2 = start only, 3 = end only)
- `indices`: Cell array of frame indices for each capillary segment

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `numStates` | Number of HMM hidden states | 2 |
| `numObservations` | Number of observation symbols | 4 |
| `transProbs` | State transition probability matrix | `[0.9 0.1; 0.1 0.9]` |
| `emissionProbs` | Emission probability matrix | See `example.m` |
| `cap_thresh` | Minimum capillary segment duration [s] | 0.1 |
| `vel_thresh` | Velocity threshold for flow detection [mm/s] | 1 |
| `grid_spacing` | Spatial resolution of tracking grid [m] | 1.7523e-5 |
| `PRF` | Pulse repetition frequency [Hz] | 1000 |

## Methods

### SingleCapillaryReporter Class

- `train()` - Train HMM parameters using Baum-Welch algorithm
- `analyze()` - Detect capillary segments using trained HMM
- `display_capillaries()` - 3D visualization of tracks with highlighted capillaries
- `tracks2dwellmap(grid)` - Convert capillary tracks to dwell time map
- `overlay_dwellmap(dwellMap, denseMap, grid)` - Overlay dwell map on dense reconstruction

## Example Output

The tool provides:
1. Console output showing the number of detected capillary segments
2. 3D visualization (Figure 42) of all tracks with capillaries in red
3. Dwell time map overlay (Figure 10) showing capillary dwell times

## File Structure

```
SingleCapillaryReporters/
├── data/
│   ├── LongEnsemble_1.mat      # Sample tracking data
│   └── DenseMap.mat            # Dense reconstruction for overlay
├── src/
│   └── SingleCapillaryReporter.m  # Main class file
├── example.m                    # Example script
├── LICENSE
└── README.md
```

## Algorithm Overview

1. **Feature Extraction**: Calculate velocity magnitude and acceleration spectrogram from particle tracks
2. **Observation Sequence**: Combine velocity and spectral features into discrete observations
3. **HMM Training**: Learn transition and emission probabilities using Baum-Welch algorithm
4. **State Decoding**: Apply Viterbi algorithm to find most likely state sequence
5. **Post-processing**: Merge small groups, identify capillary regions, filter by velocity
6. **Classification**: Categorize segments as full capillary, start, or end

## Citation

If you use this software in your research, please cite:

```
[Citation information to be added]
```

## License

See `LICENSE` file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the Provost Ultrasound Lab.

## Acknowledgments

Developed by the Provost Ultrasound Lab for automated capillary analysis in ultrasound localization microscopy studies.
