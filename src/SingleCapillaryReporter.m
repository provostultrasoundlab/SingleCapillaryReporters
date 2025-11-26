classdef SingleCapillaryReporter < handle
    % SingleCapillaryReporter - Class for detecting and analyzing capillary segments in ultrasound tracks
    %
    % This class uses Hidden Markov Models (HMM) to identify capillary segments
    % based on velocity and spectral features extracted from particle tracks.
    % Author: Stephen A. Lee, PhD
    % Release Date: November 19th, 2025
    % Updated: November 19th, 2025
    
    % =====================================================================
    % PROPERTIES
    % =====================================================================
    properties
        % HMM parameters
        numStates           % Number of hidden states in the HMM
        numObservations     % Number of possible observations
        transProbs          % Transition probability matrix
        emissionProbs       % Emission probability matrix
        initProbs           % Initial state probabilities
        
        % Capillary detection parameters
        cap_thresh          % Minimum length threshold for capillary segments
        vel_thresh          % Velocity threshold for state classification
        
        % Track data
        tracks              % Cell array of particle tracks
        N                   % Number of tracks
        observations        % Processed observations from training
        
        % Imaging parameters
        PRF                 % Pulse repetition frequency (Hz)
        grid_spacing        % Spatial grid spacing
        
        % Results
        capillaries         % Structure containing detection results

        % Debugging options
        debug               % Debugging options
    end
    
    % =====================================================================
    % PUBLIC METHODS
    % =====================================================================
    methods
        % -----------------------------------------------------------------
        % Constructor
        % -----------------------------------------------------------------
        function obj = SingleCapillaryReporter(params)
            % Initialize SingleCapillaryReporter with parameter structure
            %
            % Args:
            %   params: Structure containing initialization parameters
            
            fields = fieldnames(params);
            for idx = 1:length(fields)
                obj.(fields{idx}) = params.(fields{idx});
            end
            obj.N = length(obj.tracks);
            obj.initProbs = ones(1, obj.numStates) / obj.numStates;
            if ~isfield(obj, 'debug')
                obj.debug.verbose = false;
            end
        end
        
        % -----------------------------------------------------------------
        % Train HMM
        % -----------------------------------------------------------------
        function obj = train(obj)
            % Train the HMM using the provided tracks
            %
            % This method:
            %   1. Calculates observations from all tracks
            %   2. Normalizes track lengths via interpolation
            %   3. Trains HMM parameters using Baum-Welch algorithm
            
            % Calculate observations for all tracks
            observe = cellfun(@(track) obj.calculateObservations(track, 0.01), ...
                             obj.tracks(1:obj.N), 'UniformOutput', false);
            
            % Remove empty observations
            not_empty = cellfun(@(x) ~isempty(x), observe);
            observe = observe(not_empty);
            
            % Normalize all observations to same length via interpolation
            maxLength = max(cellfun(@length, observe));
            for idx = 1:length(observe)
                observe{idx} = round(interp1(1:length(observe{idx}), observe{idx}, ...
                                             linspace(1, length(observe{idx}), maxLength), 'linear'));
            end
            
            % Train HMM parameters
            fprintf('Training HMM with %d sequences of length %d...\n', ...
                    length(observe), maxLength);
            [obj.transProbs, obj.emissionProbs] = hmmtrain(observe, obj.transProbs, ...
                                                           obj.emissionProbs, 'Verbose', obj.debug.verbose);
            obj.observations = observe;
        end
        
        % -----------------------------------------------------------------
        % Analyze Tracks
        % -----------------------------------------------------------------
        function analyze(obj)
            % Analyze all tracks to identify capillary segments
            %
            % Uses Viterbi algorithm to find most likely state sequence,
            % then post-processes to identify capillary regions based on
            % velocity and state transition patterns.
            
            iscap = zeros(obj.N, 1);
            indices = cell(obj.N, 1);
            
            for idx = 1:obj.N
                % Calculate observations and velocities for current track
                [observe, vel] = obj.calculateObservations(obj.tracks{idx}, 0.01);
                if isempty(observe)
                    continue
                end
                
                % Run Viterbi algorithm to find most likely state sequence
                viterbiPath = hmmviterbi(observe, obj.transProbs, obj.emissionProbs);
                
                % Skip if entire track is in state 1
                if all(viterbiPath == 1)
                    continue
                end
                
                % --- Post-process Viterbi path ---
                % Step 1: Remove small isolated state groups
                flag = true;
                while flag
                    % Find state change indices
                    changeIndices = [1, find(diff(viterbiPath)) + 1, length(viterbiPath) + 1];
                    
                    % Split path into groups of consecutive states
                    groups = arrayfun(@(i) viterbiPath(changeIndices(i):changeIndices(i+1)-1), ...
                                     1:length(changeIndices)-1, 'UniformOutput', false);
                    n_groups = length(groups);
                    [smallest_group, minIdx] = min(cellfun(@length, groups));
                    
                    % Stop if too few groups or smallest group is large enough
                    if n_groups <= 3 || smallest_group >= 2
                        break
                    end
                    
                    % Merge smallest group with neighbors
                    if minIdx == 1
                        % Merge first group with next
                        groups{minIdx} = [groups{minIdx+1}(1) * ones(size(groups{minIdx})), groups{minIdx+1}];
                        groups(minIdx+1) = [];
                    elseif minIdx == length(groups)
                        % Merge last group with previous
                        groups{minIdx-1} = [groups{minIdx-1}, groups{minIdx-1}(1) * ones(size(groups{minIdx}))];
                        groups(minIdx) = [];
                    else
                        % Merge middle group with adjacent groups
                        groups{minIdx-1} = [groups{minIdx-1}, groups{minIdx-1}(1) * ones(size(groups{minIdx})), groups{minIdx+1}];
                        groups(minIdx:minIdx+1) = [];
                    end
                    viterbiPath = fillmissing(cell2mat(groups), "linear");
                end
                
                % Step 2: Identify capillary groups based on weights
                changeIndices = [1, find(diff(viterbiPath)) + 1, length(viterbiPath) + 1];
                groups = arrayfun(@(i) viterbiPath(changeIndices(i):changeIndices(i+1)-1), ...
                                 1:length(changeIndices)-1, 'UniformOutput', false);
                
                % Calculate mean velocity for each group
                mean_vel = zeros(1, length(groups));
                for i = 1:length(groups)
                    mean_vel(i) = mean(vel(changeIndices(i):changeIndices(i+1)-1));
                end
                
                % Calculate weights: favor state-2 groups with low velocity
                weight = zeros(1, length(groups));
                for i = 1:length(groups)
                    weight(i) = sum(groups{i} == 2) + 1/(mean_vel(i) + 1);
                end
                
                % Only consider weights for state-2 groups
                weight = cellfun(@(x) all(x == 2), groups) .* weight;
                weight = weight / sum(weight);
                
                % Identify significant capillary groups
                vP = ones(size(viterbiPath));
                cap_groups = find(weight > 0.05);
                if isempty(cap_groups)
                    cap_groups = find(weight > mean(weight(weight ~= 0)));
                end
                
                if length(cap_groups) < 2
                    % Single capillary group
                    first_cap_group = cap_groups(1);
                    vP(changeIndices(first_cap_group):changeIndices(first_cap_group+1)-1) = 2;
                else
                    % Multiple groups: mark region between first and last as capillary
                    first_cap_group = cap_groups(1);
                    last_cap_group = cap_groups(end);
                    vP(changeIndices(first_cap_group):changeIndices(last_cap_group+1)-1) = 2;
                end
                viterbiPath = vP;
                
                % Step 3: Final velocity-based filtering
                changeIndices = [1, find(diff(viterbiPath)) + 1, length(viterbiPath) + 1];
                groups = arrayfun(@(i) viterbiPath(changeIndices(i):changeIndices(i+1)-1), ...
                                 1:length(changeIndices)-1, 'UniformOutput', false);
                
                mean_vel = zeros(1, length(groups));
                for i = 1:length(groups)
                    mean_vel(i) = mean(vel(changeIndices(i):changeIndices(i+1)-1));
                end
                
                % Remove high-velocity groups from capillary classification
                mean_vel = cellfun(@(x) all(x == 2), groups) .* mean_vel;
                group_vel = mean_vel(mean_vel ~= 0);
                cap_groups = find(mean_vel ~= 0);
                if group_vel > obj.vel_thresh
                    first_cap_group = cap_groups(1);
                    viterbiPath(changeIndices(first_cap_group):changeIndices(first_cap_group+1)-1) = 1;
                end
                
                % Analyze final path to determine capillary status
                [iscap(idx), loc] = obj.analyzeViterbiPath(viterbiPath, vel, ...
                                                           obj.cap_thresh * obj.PRF);
                indices{idx} = loc;
            end
            
            % Store results
            cap.iscap = iscap;
            cap.indices = indices;
            obj.capillaries = cap;
            
            fprintf('Tracks contain: [%d full caps | %d starts | %d ends]\n', ...
                    sum(iscap == 1), sum(iscap == 2), sum(iscap == 3));
        end

        % -----------------------------------------------------------------
        % Calculate Observations
        % -----------------------------------------------------------------
        function [observe, vel] = calculateObservations(obj, track, smoothing)
            % Calculate observation sequence from particle track
            %
            % Args:
            %   track: Nx4 matrix [x, y, frame, z]
            %   smoothing: Smoothing factor for velocity (set to [] to disable)
            %
            % Returns:
            %   observe: Observation sequence for HMM
            %   vel: Velocity magnitude at each frame
            
            % Convert track positions to mm and calculate velocity
            trk = track(:, 1:2) * (obj.grid_spacing) * 1e3;  % Convert to mm
            diff_trk = diff(trk) * obj.PRF;                   % Velocity in mm/s
            vel = sqrt(sum(diff_trk.^2, 2));                  % Magnitude
            
            % Apply smoothing if requested
            if ~isempty(smoothing)
                vel = smooth(vel, smoothing);
            end
            
            % Binary classification: slow (2) vs fast (1)
            velo = double(vel <= obj.vel_thresh) + 1;
            
            % --- Extract spectral features from acceleration ---
            frames = track(:, 3);
            t_frames = frames / obj.PRF;               % Time in seconds
            acc = diff(vel);                           % Acceleration (length n-2)
            t_a = t_frames(2:end-1);                   % Align acc to middle frames
            
            % Prepare signal for spectrogram
            x = acc(:);
            Fs = obj.PRF;
            
            % Spectrogram parameters (adaptive to signal length)
            win = max(8, round(0.2 * Fs));
            win = min(win, numel(x));
            noverlap = min(round(0.9 * win), win - 1);
            nfft = max(256, 2^nextpow2(win));
            
            % Compute spectrogram power
            [S, F, T] = spectrogram(x, win, noverlap, nfft, Fs);
            P = abs(S).^2;
            
            if size(P, 2) > 1
                % Calculate high-frequency energy (>= 1 Hz)
                hf_idx = F >= 1;
                if ~any(hf_idx)
                    hf_energy = zeros(1, size(P, 2));
                else
                    hf_energy = sum(P(hf_idx, :), 1);
                end
                
                % Interpolate HF energy to acceleration timepoints
                hf_interp = interp1(T, hf_energy, t_a, 'linear', 'extrap');
                
                % Threshold using robust statistics (median + scaled MAD)
                thr = median(hf_interp) + 1.5 * mad(hf_interp, 1);
                high_energy_mask = double(hf_interp < thr);
                
                % Map mask back to velocity frames
                frame_mask_vals = interp1(1:length(high_energy_mask), high_energy_mask, ...
                                         linspace(1, length(high_energy_mask), length(velo)));
                acc = frame_mask_vals(:) * 0.5;
            else
                acc = zeros(size(velo));
            end
            
            % Combine velocity and spectral features into observations
            if sum(velo == 2) > 1
                observe = floor((velo + acc) / (2.5/4));
            else
                observe = {};
            end
        end


        % -----------------------------------------------------------------
        % Display Capillaries
        % -----------------------------------------------------------------
        function display_capillaries(obj)
            % Visualize tracks with identified capillary segments
            %
            % Creates a 3D plot showing all tracks (black) with capillary
            % segments highlighted (red points)
            
            figure(42);
            clf;
            hold on
            
            for i = 1:length(obj.tracks)
                % Plot full track in black
                plot3(obj.tracks{i}(:, 1), obj.tracks{i}(:, 3), obj.tracks{i}(:, 4), 'k');
                
                % Highlight capillary segments in red
                if obj.capillaries.iscap(i) == 1
                    loc = cell2mat(obj.capillaries.indices(i));
                    scatter3(obj.tracks{i}(loc, 1), obj.tracks{i}(loc, 3), ...
                            obj.tracks{i}(loc, 4), '.', 'r')
                end
            end
            
            hold off
            set(gca, 'YDir', 'reverse')
        end

        % -----------------------------------------------------------------
        % Tracks to Dwell Map
        % -----------------------------------------------------------------
        function dwellMap = tracks2dwellmap(obj,grid)
            % Convert tracks to dwell time map
            %
            % Args:
            %   grid: Structure containing grid parameters
            %
            % Returns:
            %   dwellMap: 2D dwell time map

            n_segments = sum(obj.capillaries.iscap == 1);
            cidx = find(obj.capillaries.iscap == 1);
            M = [0 cumsum(cellfun(@(x) length(x), obj.capillaries.indices(cidx))')];
            [X,Z,T] = deal(zeros(1,M(end)));

            for idx = 1:n_segments
                trk = obj.tracks{cidx(idx)}(:,1:4);           % trajectory [x,y,z] 
                loc = obj.capillaries.indices{cidx(idx)};
                X(M(idx)+1:M(idx+1)) = trk(loc,1)'*grid.resBf + grid.x_bounds(1); % x positions
                Z(M(idx)+1:M(idx+1)) = trk(loc,3)'*grid.resBf + grid.z_bounds(1); % z positions
                T(M(idx)+1:M(idx+1)) = ones(1,length(loc)).*range(trk(loc,4))/obj.PRF; % total time in capillary segment
            end

            % Map X/Y values to bin indices
            Xi = round( interp1(grid.xbins, 1:grid.imXsize, X, 'pchip', 'extrap') );
            Zi = round( interp1(grid.zbins, 1:grid.imZsize, Z, 'pchip', 'extrap') );

            % Limit indices to the range [1,numBins]
            id = (Xi>1).*(Zi>1).*(Xi<grid.imXsize).*(Zi<grid.imZsize);

            % accumulate the mean value of the transit time onto grid
            dwellMap = accumarray([Zi(id==1)' Xi(id==1)'], T(id==1)', [grid.imZsize grid.imXsize],@mean);

        end

    end
    
    % =====================================================================
    % STATIC METHODS
    % =====================================================================
    methods(Static)
        % -----------------------------------------------------------------
        % Viterbi Algorithm
        % -----------------------------------------------------------------
        function viterbiPath = runViterbi(observations, numStates, initProbs, ...
                                         emissionProbs, transProbs)
            % Custom implementation of Viterbi algorithm
            %
            % Args:
            %   observations: Sequence of observed states
            %   numStates: Number of hidden states
            %   initProbs: Initial state probabilities
            %   emissionProbs: Emission probability matrix
            %   transProbs: Transition probability matrix
            %
            % Returns:
            %   viterbiPath: Most likely sequence of hidden states
            
            viterbiPath = zeros(1, length(observations));
            delta = initProbs .* emissionProbs(:, observations(1))';
            [~, viterbiPath(1)] = max(delta);
            
            for t = 2:length(observations)
                for s = 1:numStates
                    deltaTemp = delta .* transProbs(:, s)';
                    [delta(s), ~] = max(deltaTemp);
                    delta(s) = delta(s) * emissionProbs(s, observations(t));
                end
                [~, viterbiPath(t)] = max(delta);
            end
        end
        
        % -----------------------------------------------------------------
        % Analyze Viterbi Path
        % -----------------------------------------------------------------
        function [cap_index, cap_loc] = analyzeViterbiPath(viterbiPath, vel, cap_thresh)
            % Classify capillary segment type based on Viterbi path
            %
            % Args:
            %   viterbiPath: Sequence of states from Viterbi algorithm
            %   vel: Velocity measurements
            %   cap_thresh: Minimum length threshold for capillary
            %
            % Returns:
            %   cap_index: 0 = no capillary, 1 = full, 2 = start, 3 = end
            %   cap_loc: Indices of capillary segment
            
            [pks, loc] = findpeaks(abs(diff(viterbiPath)), "MaxPeakWidth", 1);
            cap_loc = [];
            cap_index = 0;
            
            if isempty(pks)
                return
            end
            
            % --- Case 1: Two transitions (full capillary segment) ---
            if numel(pks) == 2
                [~, Uloc] = findpeaks((diff(viterbiPath)));     % Upward transition
                [~, Dloc] = findpeaks(-1*(diff(viterbiPath)));  % Downward transition
                
                % Validate transition pattern
                if numel(Uloc) ~= numel(Dloc) || Dloc < Uloc
                    return
                end
                
                % Calculate mean velocities in each section
                art_sec = mean(vel(1:loc(1)));                   % Arteriole section
                vei_sec = mean(vel(loc(end):end));               % Venule section
                cap_sec = vel(loc(1):loc(end));                  % Capillary section
                thresh = mean(cap_sec) + 1.5 * std(cap_sec);
                
                % Check if capillary section is distinct and long enough
                if (vei_sec > thresh) && (art_sec > thresh) && ...
                   (abs(diff(loc([1 end]))) > cap_thresh)
                    cap_index = 1;
                    cap_loc = loc(1):loc(end);
                end
                
            % --- Case 2: Single transition (partial capillary) ---
            elseif numel(pks) == 1
                sec_1 = vel(1:loc);
                sec_2 = vel(loc:end);
                
                % Check for capillary start (arteriole to capillary)
                if viterbiPath(end) == 1 && loc > cap_thresh && ...
                   (mean(sec_2) > (mean(sec_1) + 2*std(sec_1)))
                    cap_loc = 1:loc;
                    cap_index = 2;
                    
                % Check for capillary end (capillary to venule)
                elseif viterbiPath(1) == 1 && (length(viterbiPath)-loc) > cap_thresh && ...
                       (mean(sec_1) > (mean(sec_2) + 2*std(sec_2)))
                    cap_loc = loc:length(viterbiPath);
                    cap_index = 3;
                end
            end
        end

        % -----------------------------------------------------------------
        % Overlay Dwell Map
        % -----------------------------------------------------------------
        function overlay_dwellmap(dwellMap, denseMap, grid)
            % Overlay dwell time map on dense reconstruction image
            %
            % Args:
            %   dwellMap: 2D dwell time map
            %   denseMap: Dense reconstruction image (background)
            %   grid: Structure containing spatial grid parameters
            
            % Prepare alpha channel from dwell map
            alpha = dwellMap;
            alpha(alpha < 0) = 0;
            alpha = rescale(alpha, 0, 5);
            
            % Create figure with background and foreground layers
            figure(10);
            clf;
            hAx = gca;
            
            % Background layer (dense map)
            hB = imagesc(hAx, grid.xbins, grid.zbins, zeros(size(denseMap)));
            hold on;
            
            % Foreground layer (dwell map)
            hF = imagesc(zeros(size(dwellMap)));
            set(hF, 'XData', get(hB, 'XData'), 'YData', get(hB, 'YData'));
            set(hF, 'AlphaData', alpha);
            
            % Configure axes
            xlabel('x [m]', 'FontName', 'Arial', 'FontSize', 12);
            ylabel('z [m]', 'FontName', 'Arial', 'FontSize', 12);
            axis equal tight
            
            % Prepare background: grayscale dense map
            bg = repmat(mat2gray(mean(denseMap, 3)), [1 1 3]);
            bg = max(bg, 0)*10;
            
            % Prepare foreground: normalized dwell map
            fg = mat2gray(dwellMap);
            
            % Apply images to layers
            set(hB, 'CData', bg);
            set(hF, 'CData', fg);
            
            % Apply custom colormap and show colorbar
            colormap(SingleCapillaryReporter.ghost());
            cb = colorbar;
            ylabel(cb, 'Dwell Time [s]', 'FontName', 'Arial', 'FontSize', 12);
        end

        % -----------------------------------------------------------------
        % Ghost Colormap
        % -----------------------------------------------------------------
        function cmap = ghost()
            % Generate a custom 'ghost' colormap
            %
            % Returns:
            %   cmap: 256x3 RGB colormap matrix
            
            % Define key colors in hex format
            hex_colors = {'#96FC81', '#E0C77B', '#FFE28C'};
            n_colors = length(hex_colors);
            
            % Convert hex to RGB [0, 1]
            rgb_keys = zeros(n_colors, 3);
            for i = 1:n_colors
                hex = hex_colors{i};
                rgb_keys(i, :) = [hex2dec(hex(2:3)), ...
                                  hex2dec(hex(4:5)), ...
                                  hex2dec(hex(6:7))] / 255;
            end
            
            % Interpolate to create 256-color colormap
            numColors = 256;
            cmap = interp1(1:n_colors, rgb_keys, ...
                          linspace(1, n_colors, numColors), 'pchip');
        end
    end
end