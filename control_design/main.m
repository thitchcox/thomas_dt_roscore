v_bar = 0.5;
k_trim_actual = 0.1;
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

% Observer design
K_obs = place(A_aug.', C_aug.', [-2;-2;-4]).';

% Control design
k_theta = 5;
%K_cont = place(A,B,[-1;-2]);
K_cont = [k_theta^2/(4*v_bar), k_theta];

% Store parameters
params.k_trim_actual = k_trim_actual;
params.K_cont = K_cont;
params.K_obs = K_obs;

%% Simulate
x_duckie_0 = [-1;0.4];
x_obs_0 = [1;0;0];
x_0 = [x_obs_0;x_duckie_0];
tSpan = linspace(0,10,500);

[t,x] = ode45(@(t,x) master(t,x,params), tSpan, x_0);

%% Plot
% state estimation error
figure(1)
plot(t, x(:,1) - x(:,4), t, x(:,2) - x(:,5),t,x(:,3) - k_trim_actual,'LineWidth',2)
grid on
title('Observer errors')
xlabel('Time (s)')
ylabel('Error')
legend('d estimation error','theta estimation error','bias estimation error')

figure(2)
plot([tSpan(1), tSpan(end)],[k_trim_actual k_trim_actual],t,x(:,3),'LineWidth',2)
title('Trim Estimate')
grid on
legend('Actual trim', 'Estimated trim')
xlabel('Time (s)')
ylabel('Trim value')

figure(3)
plot(t,x(:,4:5),'LineWidth',2)
grid on
title('Lane Pose Error')
xlabel('Time (s)')
ylabel('d, theta')
legend('d', 'theta')
%% Master
function x_dot = master(t,x, params)
    x_obs = x(1:3); % Estimated states
    x_duckie = x(4:end); % Actual states
    
    y = x_duckie(1:2); % Measured lane pose (d, theta)
    bias = x_obs(3); % k_trim from disturbance observer
    
    % Control
    u = duckieController(x_obs(1:2), params.K_cont, bias);
    
    % Update observer states
    x_dot_obs = duckieObserver(t,x_obs,u,params.K_obs,y);
    
    % Update actual states
    x_dot_duckie = duckieDynamics(t,x_duckie,u,params.k_trim_actual);
    
    x_dot = [x_dot_obs;x_dot_duckie];
end