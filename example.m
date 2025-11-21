
addpath(genpath('src'));

%% User Configuration for Single Capillary Analysis
datapath = 'data';
% ----- transducer parameters ----- %
Trans.frequency     = 15.625e6;                                             % transducer frequency [Hz]
Trans.aperture      = 12.8e-3;                                              % transducer aperture [m]

% ----- grid parameters ----- %
grid.startDepth     = 1.6e-3;                                               % start depth of imaging [m]
grid.endDepth       = 8.7e-3;                                               % end depth of imaging [m]
grid.SoS            = 1540;                                                 % speed of sound [m/s]
grid.wavelength     = grid.SoS/Trans.frequency;                             % wavelength [m]
grid.resBf          = grid.wavelength/(2*sqrt(2));                          % beamforming grid resolution [m] this dataset was beamfored with 1/ (2*sqrt(2)) resolution
grid.resSuperloc    = grid.wavelength/24;                                   % superlocalization grid resolution [m]
grid.x_bounds       = [-Trans.aperture/2 Trans.aperture/2];                 % x grid boundaries
grid.z_bounds       = [grid.startDepth grid.endDepth];                      % z grid boundaries
grid.xbins          = grid.x_bounds(1):grid.resSuperloc:grid.x_bounds(2);   % x grid points
grid.zbins          = grid.z_bounds(1):grid.resSuperloc:grid.z_bounds(2);   % z grid points
grid.imXsize        = length(grid.xbins);                                   % size of image in x dimension
grid.imZsize        = length(grid.zbins);                                   % size of image in z dimension

% ----- Single Capillary Reporter parameters ----- %
config.numStates        = 2;                                                % number of hidden states 
                                                                            %   (1: non-capillary, 2: capillary)
config.numObservations  = 4;                                                % number of observation symbols 
                                                                            %   1: low speed, low acceleration, 
                                                                            %   2: low speed, high acceleration, 
                                                                            %   3: high speed, low acceleration, 
                                                                            %   4: high speed, high acceleration
config.transProbs       = [0.9 0.1; 0.1 0.9];                               % initial state transition probabilities
config.emissionProbs    = [0.45 0.45 0.05 0.05; ...                         % initial emission probabilities
                           0.05 0.05 0.45 0.45]; 
config.cap_thresh       = 0.1;                                              % threshold for minimum length of capillary segment [s]
config.vel_thresh       = 1;                                                % threshold for minimum velocity to consider as flowing [mm/s]
config.grid_spacing     = grid.resBf/2;                                     % spacing of grid points in capillary [m]
config.PRF              = 1000;                                             % pulse repetition frequency [Hz]

config.debug.verbose    = false;                                            % verbose output for debugging

%% load tracks
% load precomputed dense map
denseMap = load([datapath '/DenseMap.mat']).density;

nfiles = dir([datapath '/LongEnsemble_*.mat']);

dwellMap = zeros(grid.imZsize, grid.imXsize,length(nfiles));

for fidx = 1:length(nfiles)
    % retrain HMM every 50 buffers to decrease processing time
    if mod(fidx-1,100)==0
        config.tracks = load([datapath '/' nfiles(fidx).name]).tracks;

        % Initialize SingleCapillaryReporter Class
        SCaRe = SingleCapillaryReporter(config);

        % train Hidden Markov Model on all tracks
        SCaRe.train();
    else
        % deposit new tracks into existing SCaRe object
        SCaRe.tracks = load([datapath '/' nfiles(fidx).name]).tracks;
        SCaRe.N = length(SCaRe.tracks);
    end

    % Analyze all tracks using trained HMM
    SCaRe.analyze();

    % Dwell Map
    dwellMap(:,:,fidx) = SCaRe.tracks2dwellmap(grid);
    % display
    if config.debug.verbose
        SCaRe.display_capillaries();
    end
end
dwellMap(dwellMap==0) = nan;
dwellMap = mean(dwellMap,3,'omitmissing');

 
% display overlay of dwell map on dense reconstruction
SCaRe.overlay_dwellmap(dwellMap, denseMap, grid);
clim([0 2])
