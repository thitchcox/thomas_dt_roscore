% Fit a GP to detected lane points.  
% Assume that points have been detected and passed thru ground projection
% node.  Also, we know which points are white and which are yellow.  

% TODO
% - Result seems to be sensitive to the kernelScale parameter.  Once I
% train this hyperparameter, how to I use it in a function?

close all
clc
config_lane_gp;

% Choose a data file to process
% load('aligned_straight.mat')
% load('offset_straight.mat')
% load('aligned_curve_right.mat')
% load('offset_curve_right.mat')
% load('aligned_curve_left.mat')
% load('offset_curve_left.mat')

% Real data
% [x_white, x_yellow, y_white, y_yellow] = loadRealData('dt_straight_aligned');
[x_white, x_yellow, y_white, y_yellow] = loadRealData('dt_left_curve');


% Choose whether to optimize gp hyperparameters
tf_optimize_hyperparameters = false;


%% Fit GPs
if tf_optimize_hyperparameters
    disp('Optimizing GP hyperparamters...')
    tic
    % Fit a GP to white points: x = f(y)
    gpWhite = fitrgp(y_white, x_white, 'BasisFunction', 'constant', 'Beta', ...
        lane_width / 2, 'KernelFunction', 'matern32', 'OptimizeHyperparameters', ...
        'kernelscale');

    % Fit a GP to yellow points: x = f(y)
    gpYellow = fitrgp(y_yellow, x_yellow, 'BasisFunction', 'constant', 'Beta', ...
        -lane_width / 2, 'KernelFunction', 'matern32', 'OptimizeHyperparameters', ...
        'kernelscale');
    disp('Done.')
    toc
else
    disp('Fitting GPs...')
    tic
    gpWhite = fitrgp(y_white, x_white);
    gpYellow = fitrgp(y_yellow, x_yellow);
    disp('Done.')
    toc
end


%% Predict
% Use gpWhite to predict white line on a regular spacing
lin_x_white = gpWhite.predict(lin_y.');

% Use gpYellow tp predict yellow line on a regular spacing
lin_x_yellow = gpYellow.predict(lin_y.');

% Generate mean path estimate as the average of the two GPs.
x_mid = mean([lin_x_yellow, lin_x_white], 2);


%% Generate pose estimate
% Fit a line to the mean path estimate close to the robot
close_pts = lin_y <= fit_line_to_nearest;
y_nearest = lin_y(close_pts);
x_nearest = x_mid(close_pts);

% Fit a line to these points
near_line = polyfit(x_nearest(:), y_nearest(:) ,1);

% Compute offset estimate (x val at y = 0)
d = - near_line(2) / near_line(1);

% Compute phi (the attitude error).
special_x = interp1(lin_y, x_mid, d_lookahead, 'linear') - d;
L = sqrt(d_lookahead^2 + special_x^2);
phi = - special_x / L;

disp(['Estimated lane position error: ', num2str(d), ' m'])
disp(['Estimated attitude error: ', num2str(rad2deg(phi)), ' deg']);


%% Plots
figure
ha = gca;
hold on

% Show the points
scatter(ha, x_white, y_white, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w')
scatter(ha, x_yellow, y_yellow, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', matlab_yellow)

% Show the GPs
plot(ha, lin_x_white, lin_y, '-', 'LineWidth', line_width, 'Color', matlab_grey)
plot(ha, lin_x_yellow, lin_y, '-', 'LineWidth', line_width, 'Color', matlab_yellow);
plot(ha, x_mid, lin_y, '--', 'LineWidth', line_width, 'Color', matlab_cyan)

% Set limits and grid.
ha.XLim = x_lim_sandbox;
ha.YLim = y_lim_sandbox - 0.01;
xline(0, '--');
grid on

% Legend and labels
% Get handle here, before error lines are drawn
hp = get(gca, 'Children');

% Show the errors
line([0; d], [0; 0], 'Color', 'r', 'LineWidth', line_width / 2);
line([d; d + special_x], [0; d_lookahead], 'Color', 'k', 'LineStyle', '--', ...
    'LineWidth', line_width / 2);
line([d; d], [0; d_lookahead], 'Color', 'k', 'LineStyle', '--', ...
    'LineWidth', line_width / 2);
line([d; d + special_x], [d_lookahead; d_lookahead], 'Color', 'k', 'LineStyle', ...
    '--', 'LineWidth', line_width / 2);
text(d / 3, 0.01, 'd', 'Color', 'r', 'FontSize', 12)
text(5 * d / 4, 0.02, '\phi', 'Color', 'k', 'FontSize', 12)

% Legend and labels (cont.)
legend([hp(6), hp(5), hp(4) hp(3), hp(2)], 'Detected white point', ...
    'Detected yellow point', 'White line estimate', 'Yellow line estimate', ...
    'Trajectory estimate')
xlabel('X [m]')
ylabel('Y [m]')

