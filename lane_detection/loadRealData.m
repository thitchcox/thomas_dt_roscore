% Load the text files 'white.txt' and 'yellow.txt', which contain real
% observations from the Duckietown gym.
function [x_white, x_yellow, y_white, y_yellow] = loadRealData(dir_name)

    % Read matrices
    white_pts = dlmread(fullfile(dir_name, 'white.txt'));
    yellow_pts = dlmread(fullfile(dir_name, 'yellow.txt'));
    
    % Parse
    x_white = -white_pts(:, 2);
    y_white = white_pts(:, 1);
    
    x_yellow = -yellow_pts(:, 2);
    y_yellow = yellow_pts(:, 1);  
end

