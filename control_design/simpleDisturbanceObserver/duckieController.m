function u = duckieController(x, K_cont, bias)
v_bar = 0.5;
span = 0.2;
   

u = -K_cont*x - (v_bar/span)*bias;
end