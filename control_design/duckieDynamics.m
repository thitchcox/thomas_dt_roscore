function x_dot = duckieDynamics(t,x,u,k_trim)
    v_bar = 0.5;
    span = 0.2;

    % Nominal Plant
    A = [0,v_bar;0,0];
    B =[0;1];
    L =[0;v_bar/span];
    x_dot = A*x + B*u + L*k_trim;
end
    
    