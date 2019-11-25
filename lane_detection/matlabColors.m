function rgb = matlabColors(color_name)
%MATLABCOLORS Return an rgb value corresponding to standard MALTAB colours
%
% PARAMETERS
% ----------
% color_name [string]
%     The name of the colour.  See below for options.
% rgb [1 x 3 double array]
%     The rgb values.  Note rgb(i) \in [0, 1].
% -------------------------------------------------------------------------
    if strcmpi(color_name, 'blue')
        rgb = [0    0.4470    0.7410];
    elseif strcmpi(color_name, 'orange')
        rgb = [0.8500, 0.3250, 0.0980];
    elseif strcmpi(color_name, 'yellow')
        rgb = [0.9290, 0.6940, 0.1250];
    elseif strcmpi(color_name, 'purple')
        rgb = [0.4940, 0.1840, 0.5560];
    elseif strcmpi(color_name, 'green')
        rgb = [0.4660, 0.6740, 0.1880];
    elseif strcmpi(color_name, 'cyan')
        rgb = [0.3010, 0.7450, 0.9330];
    elseif strcmpi(color_name, 'maroon')
        rgb = [0.6350, 0.0780, 0.1840];
    elseif strcmpi(color_name, 'grey')
        rgb = [0.5, 0.5, 0.5];
    elseif strcmpi(color_name, 'black')
        rgb = [0, 0 ,0];
    else
        error('Unrecognized colour choice')
    end
end

