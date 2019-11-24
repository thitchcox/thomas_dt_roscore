v_bar = 1;

% Reference model
omega_r = 5/v_bar;
zeta_r = 1;
k_d = -10;
k_p = -k_d^2/(4*v_bar);
%params.A_r = v_bar*[0, 1; -omega_r^2, -2*zeta_r*omega_r];
params.A_r = [0, v_bar; k_p, k_d];
params.B_r = [0; 1];
params.C_r = eye(2);
params.D_r = 0;
params.P = lyap(params.A_r, eye(2));

%

% Actual dynamics
params.A = [0, v_bar; 0 , 0];
params.B = [0;1];
params.C = eye(2);
params.D = 0;



%% Simulate
x_0 = [1;0;1;0;-5;-0.2;1];
t_span = linspace(0,100, 500);
[u,t] = gensig('square',2,100,0.01);
u_table = [t,u*1];
%u_table = [0,0; 10000,0];

params.gamma = 0;
[t1,x1] = ode45(@(t,x) master(t,x,u_table,params),t_span, x_0);

params.gamma = 8;
[t2,x2] = ode45(@(t,x) master(t,x,u_table,params),t_span, x_0);

%%
figure(1)
plot(t1,x1(:,3),'LineWidth',2,'Color','black','LineStyle','--')
hold on
plot(t2,x2(:,1),'LineWidth',2,'color','red')
plot(t1, x1(:,1),'LineWidth',2,'color','blue')
hold off
grid on
ylabel('$d$','interpreter','latex','FontSize',14)
xlabel('Time (s)','interpreter','latex','FontSize',14)
title('Adaptive Control')
legend('Adaptive control','Reference Model Response')
%axis([-inf inf -0.5 1.5 ])
figure(2)
plot(t1, x2(:,5:7))


%%
function [x_dot] = master(t,x, u_table, params)
    x_actual = x(1:2);
    x_reference = x(3:4);
    theta = x(5:7);
    
    e = x_actual - x_reference;
    u_command = interp1(u_table(:,1), u_table(:,2), t);
    
    % Update theta
    psi = [0,0,0;x_actual(1), x_actual(2), u_command];
    theta_dot = -params.gamma*psi.'*params.P*e;
    
    e_tracking = x_actual - [u_command;0];
    e_tracking_r = x_reference - [u_command;0];
    % Update state
    u_actual = theta(3)*u_command + [theta(1), theta(2)]*e_tracking;
    x_dot_actual = params.A*x_actual + params.B*u_actual;
    
    % Update reference
    x_dot_ref = params.A_r*e_tracking_r + params.B_r*u_command;
    
    x_dot = [x_dot_actual; x_dot_ref; theta_dot];
end
    
    