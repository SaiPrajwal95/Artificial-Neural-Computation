function [pre_state,cur_state,pre_action,cur_action,x,v_x,theta,v_theta] = reset_cart(BETA)  % reset the cart pole to initial state
pre_state=1;
cur_state=1;
pre_action=-1;  % -1 means no action been taken
cur_action=-1;
rng(42)
x=rand*BETA;     % the location of cart
rng(42)
v_x=rand*BETA;   % the velocity of cart
rng(42)
theta=rand*BETA;   %the angle of pole
rng(42)
v_theta=rand*BETA;    %the velocity of pole angle
