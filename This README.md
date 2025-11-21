This README.txt file was generated on 2025-11-21 by Stephen Alexander Lee

--------------------
GENERAL INFORMATION
--------------------

1. Dataset Title: SingleCapillaryReporters 

2. Author Information
	A. Principal Investigator Contact Information
		Name: Jean Provost
		Institution: Polytechnique Montreal
		Email: jean.provost@polymtl.ca

	B. Associate or Co-investigator Contact Information
		Name: Stephen A. Lee
		Institution: Polytechnique Montreal
		Email: stephen.lee@polymtl.ca


3. Date of data collection (single date, range, approximate date): 2025-03-20

4. Geographic location of data collection: Montreal, QC, Canada

5. Information about funding sources that supported the collection of the data: Vanier Banting Postdoctoral Fellowship (NSERC), CIHR Postdoctoral Fellowship, TransMedTech Postdoctoral Scholarship, the Institute for Data Valorization (IVADO), Canadian Foundation for Innovation (38095), the CIHR (452530), New Frontiers in Research Fund (NFRFE-2018-01312), the NSERC(RGPIN-2019-04982), Fonds de Recherche de Quebec - Nature et Technologies, the Quebec Bio-Imaging Network, CONAHCYT, the Insightec, the Healthy Brains Healthy Lives, the Canada First Research Excellence Fund, the NIH National Institutes of Aging (1R01AG079894).


---------------------------
SHARING/ACCESS INFORMATION
---------------------------

1. Licenses/restrictions placed on the data: These data are available under a Creative Commons Public Domain Dedication (CC0 1.0) license.

2. Links to publications that cite or use the data: https://doi.org/10.48550/arXiv.2407.07857, https://doi.org/10.48550/arXiv.2509.08149

3. Links/relationships to ancillary data sets or software packages: https://github.com/provostultrasoundlab/SingleCapillaryReporters

4. Was data derived from another source? no

5. Recommended citation for this dataset: Stephen A. Lee, Alexis Leconte, Alice Wu, Joshua Kinugasa, Gerardo R. Palacios, Jonathan Poree, Abbas F. Sadikot, Andreas Linninger, Jean Provost. (2025). SingleCapillaryReporters. Federated Research Data Repository. doi:10.20383/103.01507

---------------------
DATA & FILE OVERVIEW
---------------------

1. File List

   A. Folder: data_SingleCapillaryReporters
      Short description: Data containing (1) pre-computed ultrasound localization microscopy density map and (295) datasets containing all trajectories (N x 5: [x y z frames backscatter_intensity]) for use with the SingleCapillaryReporters example script.

   B. Folder: data_synthetic_L22_2D_brain
      Short description: Dataset containing (179) folders with pre-computed microbubble ground truth flow datasets in a synthetic mouse brain cortex. Each folder contains the raw RF data (binary file), the beamformed data using delay-and-sum beamforming, and the ground truth microbubble positions and vessel radius from the synthetic cortex.

2. Relationship between files, if important: The two datasets are independent. data_SingleCapillaryReporters contains real in vivo mouse data for capillary detection validation. data_synthetic_L22_2D_brain contains synthetic data with ground truth for algorithm validation and benchmarking.

3. Additional related data collected that was not included in the current data package: N/A

4. Are there multiple versions of the dataset? no 


---------------------------
METHODOLOGICAL INFORMATION
---------------------------

1. Description of methods used for collection/generation of data: The in vivo data was acquired in a wild-type 6-month-old mouse using continuous perfusion of Definity microbubbles via tail-vein catheterization. The scan was performed for 5 minutes with a gap-less acquisition scheme at 1000 frames per second, 9-angle plane wave compounding on a 2D 16 MHz ultrasound linear array. Details can be found here (https://doi.org/10.48550/arXiv.2407.07857).

2. Methods for processing the data: Synthetic mouse brain cortex datasets were computed through sequential Monte Carlo simulations of microbubbles flowing through vessel trajectories. These individual microbubbles were combined into a 3D dataset where an example 16 MHz linear ultrasound transducer was used to insonify a slice of the microbubbles. The data contained include 3D microbubble flow through the field of view of the 2D ultrasound probe. More information is detailed in (https://doi.org/10.48550/arXiv.2509.08149).

3. Instrument- or software-specific information needed to interpret the data: MATLAB (tested on R2020b or later) with the Signal Processing Toolbox (for `spectrogram`, `smooth`, `mad`) and the Statistics and Machine Learning Toolbox (for `hmmtrain`, `hmmviterbi`)


4. Standards and calibration information, if appropriate: N/A

5. Environmental/experimental conditions: Head-fixed, isoflurane anesthetized animals

6. Describe any quality-assurance procedures performed on the data: N/A

7. People involved with sample collection, processing, analysis and/or submission: Stephen A. Lee, Alexis Leconte, Alice Wu, Joshua Kinugasa, Gerardo R. Palacios, Jonathan Poree, Abbas F. Sadikot, Andreas Linninger, Jean Provost.


-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: data_SingleCapillaryReporters
-----------------------------------------------------------------

1. Number of files: 296 total
   - 295 LongEnsemble files (LongEnsemble_1.mat through LongEnsemble_295.mat)
   - 1 DenseMap.mat file

2. File format: MATLAB .mat files (version 7.3 or compatible)

3. Missing data codes: N/A (no missing data encoding)

4. LongEnsemble_*.mat files (particle trajectory datasets):

   A. Variable name: tracks
      Description: Cell array containing particle trajectories from ultrasound localization microscopy
      Format: Each cell contains an N×5 matrix where N is the number of time points
      Columns:
         Column 1: x position (in grid units, multiply by grid_spacing for meters)
         Column 2: y position (in grid units, multiply by grid_spacing for meters)  
         Column 3: z position (in grid units, multiply by grid_spacing for meters)
         Column 4: frame number (time index)
         Column 5: backscatter intensity (arbitrary units)
      Units: Grid units (convert using grid_spacing = wavelength/(4*sqrt(2)) ≈ 1.75e-5 m)
      Typical number of tracks per file: Variable (depends on microbubble detection in that time window)
      Typical track length: Variable (N ranges from tens to hundreds of frames)

5. DenseMap.mat file (ultrasound localization microscopy density map):

   A. Variable name: density
      Description: Pre-computed ultrasound localization microscopy (ULM) density map showing accumulated microbubble detections
      Format: 3D matrix 
      Dimensions: Z × X × (channels/RGB) where Z and X correspond to imaging depth and lateral position
      Units: Accumulated counts or normalized intensity
      Purpose: Used as background image for dwell time map overlay visualization

6. Spatial grid parameters (for coordinate interpretation):
   - Grid spacing (beamforming): lambda/(2sqrt(2)) ≈ 3.50e-5 m
   - Grid spacing (superlocalization): lambda/24 ≈ 4.11e-6 m  
   - Wavelength: lambda = 1540 m/s / 15.625 MHz ≈ 9.86e-5 m
   - X bounds: [-6.4 mm, 6.4 mm] (lateral, centered on transducer)
   - Z bounds: [1.6 mm, 8.7 mm] (axial depth from transducer face)
   - Temporal resolution: 1000 frames per second (PRF = 1000 Hz)

7. Data collection timespan: 
   - Total acquisition: 5 minutes continuous recording
   - Frame rate: 1000 Hz
   - Total frames: ~300,000 frames
   - Divided into 295 ensemble windows for processing


-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: data_synthetic_L22_2D_brain
-----------------------------------------------------------------

1. Number of folders: 179 total
   - Each folder contains synthetic ultrasound data from Monte Carlo simulations of microbubble flow through mouse brain cortex vasculature

2. File format: Mixed - MATLAB .mat files and binary raw RF data files

3. Missing data codes: N/A (no missing data encoding)

4. Files per folder:

   A. Raw RF data file (binary format):
      Description: Simulated raw radio-frequency (RF) ultrasound data before beamforming
      Format: Binary file
      Dimensions: Typically channel × sample × frame
      Purpose: Raw ultrasound signals as would be acquired by transducer elements
      Note: Requires beamforming parameters to reconstruct images

   B. BF.mat (beamformed ultrasound data):
      Variable name: p (structure)
      Description: Delay-and-sum beamformed ultrasound data
      Structure fields:
         - xrange: Lateral positions [m] (1×517 double array)
                   Range: approximately [-6.4 mm, +6.4 mm]
         - zrange: Axial positions [m] (1×325 double array)
                   Range: approximately [1.0 mm, 17.0 mm]
         - image: Beamformed ultrasound image stack [325×517×500 single]
                  Dimensions: depth × lateral × frames
                  Data type: Single precision floating point
                  Typical values: Complex or magnitude ultrasound signal intensities
      Spatial resolution: 
         - Lateral spacing: ~25 μm (derived from xrange)
         - Axial spacing: ~50 μm (derived from zrange)
      Temporal dimension: 500 frames per dataset

   C. GT_microbubble.mat (ground truth data):
      Variable name: GT (structure)
      Description: Ground truth microbubble positions and vessel geometry from Monte Carlo simulation
      
      Structure fields:
         GT.MB (microbubble data):
            - idx: Microbubble indices [1×194×3500 double]
                   Identifies each microbubble across frames
            - pos: Microbubble 3D positions [3×194×3500 double]
                   Dimensions: [x; y; z] × bubbles × frames
                   Units: meters
            - vel: Microbubble velocities [1×194×3500 double]
                   Units: m/s (magnitude)
            - ecg: ECG-related timing or cardiac phase [1×194×3500 double]
                   Normalized or phase values
            - siz: Microbubble sizes [1×194×3500 double]
                   Units: meters (diameter or radius)
         
         GT.VSL (vessel geometry data):
            - pos: Vessel centerline positions [3×194×3500 double]
                   Dimensions: [x; y; z] × points × frames
                   Units: meters
            - rad: Vessel radius at each centerline point [1×194×3500 double]
                   Units: meters
      
      Array dimensions:
         - 194: Maximum number of microbubbles present in a single frame
         - 3500: Total number of simulated frames
      
      Notes: 
         - Not all 194 positions contain valid microbubbles in every frame (sparse representation)
         - Use idx field to identify valid microbubbles
         - GT.VSL provides the underlying vessel geometry through which microbubbles flow

5. Simulation parameters:
   - Transducer: 16 MHz linear array (L22 or equivalent)
   - Imaging mode: 2D plane wave or focused imaging
   - Tissue model: Synthetic mouse brain cortex vasculature
   - Simulation method: Sequential Monte Carlo for microbubble trajectories
   - Frames per dataset: 500 (beamformed), 3500 (ground truth)
   
6. Coordinate system:
   - x: Lateral dimension (perpendicular to beam propagation)
   - y: Elevation dimension (perpendicular to imaging plane, limited in 2D)
   - z: Axial dimension (along beam propagation, depth)
   - Origin: Transducer face center

7. Use cases:
   - Algorithm validation against known ground truth
   - Microbubble tracking algorithm development
   - Ultrasound localization microscopy (ULM) method benchmarking
   - Vessel reconstruction accuracy assessment
   - Flow velocity estimation validation