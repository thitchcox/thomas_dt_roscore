% Config file

%% Selecting datapoints
% Region on the ground plane in which image keypoints can be projected
x_lim_sandbox = [-0.25, 0.25];
y_lim_sandbox = [0, 0.35];

% Region of validity


%% GP settings
% The width of a lane
lane_width = 0.23;  % [m]

% Vertical spacing of prediction points
lin_y = 0 : 0.01 : 0.5;


%% Generating pose estimate
% Fit a line to the local observations to get a pose estimate
fit_line_to_nearest = 0.1;  % [m]

% Look-ahead distance to compute attitude error
d_lookahead = 0.15; % [m]


%% Plotting
line_width = 2;


%% Access stuff
% Access data
addpath(genpath('data'))

% Access colours
matlab_yellow = matlabColors('yellow');
matlab_cyan = matlabColors('cyan');
matlab_grey = matlabColors('grey');