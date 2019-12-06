% Make a few fake lanes and make a selection of points to use as training
% points.  
config_lane_gp;

% Draw the sandbox
figure
ha = gca;
ha.XLim = x_lim_sandbox;
ha.YLim = y_lim_sandbox;
xline(0, '--')
grid on

% User selects white points (right line)
disp('Select white points...')
[x_white, y_white] = getpts;
% Show the points
scatter(ha, x_white, y_white, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w')
ha.XLim = x_lim_sandbox;
ha.YLim = y_lim_sandbox;
xline(0, '--')
grid on
hold on

% Select yellow points (left line)
disp('Select yellow points...')
[x_yellow, y_yellow] = getpts;
scatter(ha, x_yellow, y_yellow, 'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', matlab_yellow)
ha.XLim = x_lim_sandbox;
ha.YLim = y_lim_sandbox;
xline(0, '--')
grid on

tf_save = input('Save result? [y/n] ', 's');
switch tf_save
    case 'y'
        name = input('Give me a name: ', 's');
        save(fullfile('data', name), 'x_white', 'y_white', 'x_yellow', 'y_yellow')
end

