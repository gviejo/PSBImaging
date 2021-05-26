%% Welcome to the EXTRACT tutorial! Written by Fatih Dinc, 03/02/2021
%perform cell extraction
clear;
path = '/mnt/DataRAID/MINISCOPE/A0600/A0634/A0634-210130'
[~,mergename] = fileparts(path);

cd(path)

M = read_file(fullfile(pwd, [mergename '.h5']));

M = M(:,:,1:5000);

config=[];
config = get_defaults(config); %calls the defaults

% Essentials, without these EXTRACT will give an error:
config.avg_cell_radius=7; %Average cell radius is 7.


%Optionals, but strongly advised to handpick:
%Movie is small enough that EXTRACT will not automatically partition this,
%but still a good idea to keep these in sight!
config.trace_output_option='raw'; % Choose 'nonneg' for non-negative Ca2+ traces, 'raw' for raw ones!
config.num_partitions_x=1;
config.num_partitions_y=1; 
config.cellfind_filter_type='none'; % The movie is clean enough, no need for lowpass filtering
config.verbose=2; %Keeping verbose=2 gives insight into the EXTRACTion process, always advised to keep 2
config.spatial_highpass_cutoff=inf; % no need for highpass filtering
config.remove_stationary_background=0; %no need for background removal

% Optionals whose defaults exist:
config.use_gpu=1; % This is a small dataset, will be fast on cpu anyways.
config.max_iter = 20; % 10 is a good number for this dataset
config.adaptive_kappa = 1;% Adaptive kappa is on for this movie. For an actual movie, keeping it off
% may be beneficial depending on the noise levels.
config.cellfind_min_snr=0.1;% Default snr is 1, lower this (never less than 0) to increase cell count at the expense of more spurious cells!


% Perform EXTRACTion:
output=extractor(M,config);

cell_check(output, M);
