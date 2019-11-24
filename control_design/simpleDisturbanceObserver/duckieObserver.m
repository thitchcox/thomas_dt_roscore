function x_hat_dot = duckieObserver(t,x_hat,u,K_obs, y)
v_bar = 0.5;
span = 0.2;

% Nominal Plant
A = [0,v_bar;0,0];
B =[0;1];
L =[0;v_bar/span];
C = eye(2);

% Augmented system (with disturbance as state)
A_aug = [A, L; 0,0,0];
B_aug = [B;0];
C_aug = [C,[0;0]];

% Observer ODE
x_hat_dot = A_aug*x_hat + B_aug*u + K_obs*(y - C_aug*x_hat);

end
