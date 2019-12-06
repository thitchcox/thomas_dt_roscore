%% Generate Data
m = 0.5;
c = 0.1;
L = 1;

x = linspace(0,1,50).';
yW = m*x + c + normrnd(0,0.05,size(x));
yY = m*x + c + L*sqrt(m^2 + 1)/(2) + normrnd(0,0.05,size(x));
dataW = [x,yW];
dataY = [x,yY];

scatter(x,yY)
hold on
scatter(x,yW)
hold off
grid on
axis equal
axis([0 1 0 1])
pause(0.01)

%% Simultaneous Constrained LS
theta0 = [0.4;0];
dtheta = 10;
iter =0;
theta_star = theta0;
while norm(dtheta) > 0.001 && iter < 20
[e,jac] = errorFunction(theta_star,dataW,dataY);
dtheta = (jac.'*jac)\(-jac.'*e);
theta_star = theta_star + dtheta;
iter = iter + 1
end

options = optimoptions('lsqnonlin','CheckGradient',true);
theta = lsqnonlin(@(x) errorFunction(x, dataW,dataY), theta0,[],[],options)
m = theta(1);
c = theta(2);
L = 1;
theta_star
yW = m*x + c;
yY = m*x + c + L*sqrt(m^2 + 1)/(2);

hold on
plot(x,yY,x,yW,'LineWidth',2)
hold off

function [e,jac] = errorFunction(x, dataW, dataY)
m = x(1);
c = x(2);
L = 1;
e = [dataW(:,2) - (m*dataW(:,1) + c);
     dataY(:,2) - (m*dataY(:,1) + c + L*sqrt(m^2 + 1)/2)];
 
jac =[[-dataW(:,1), -ones(size(dataW(:,1)))];
      [-dataY(:,1) + L*m/(2*sqrt(m^2 +1)), -ones(size(dataY(:,1)))]];
 
end