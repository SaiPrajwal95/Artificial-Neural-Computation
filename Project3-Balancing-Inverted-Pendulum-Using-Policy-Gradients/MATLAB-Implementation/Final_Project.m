clear all
ALPHA = 0.3;    % learning rate
BETA = 0.02;    % magnitude of noise added 
GAMMA = 0.9;    % discount factor 

total_mass = 1.1;  %mass of cart + pole 
mass_pole = 0.1; %mass of the pole
g = 9.8;  % acceleration due to gravity
length = 0.5;  %half length of pole
Force = 10;   %force =10N
time_step = 0.02;  % Update time interval

NUM_BOX = 162;    % Number of states sets to 162
for runs = 1:5
[pre_state,cur_state,pre_action,cur_action,x,v_x,theta,v_theta] = reset_cart(BETA);  % reset the cart pole to initial state
q_val = zeros(NUM_BOX,2);
h1 = figure;    % this figure is cart-pole
axis([-3 3 0 1.5]);
set(gca, 'DataAspectRatio',[1 1 1]);
set(h1, 'Position',[10 100 500 200]);
set(h1,'DoubleBuffer','on')
set(gca, 'PlotBoxAspectRatio',[1 1 1]);
success=0;   % succesee 0 times
reinf=0;
trial=0;
% prepare grid for
[GX,GY]=meshgrid(1:18,1:9);
GZ=zeros(9,18);
best=0;


    while success<=60000
        [q_val,pre_state,pre_action,cur_state,cur_action] = action_value(x,v_x,theta,v_theta,reinf,q_val,pre_state,cur_state,pre_action,cur_action,ALPHA,BETA,GAMMA);
        if (cur_action==1)   % push left
            F=-1*Force;
        else
            F=Force;    % push right
        end %if
        % Update the cart-pole state
        a_theta=(total_mass*g*sin(theta)-cos(theta)*(F+mass_pole*length*v_theta^2*sin(theta)))/(4/3*total_mass*length-mass_pole*length*cos(theta)^2);
        a_x=(F+mass_pole*length*(v_theta^2*sin(theta)-a_theta*cos(theta)))/total_mass;
        v_theta=v_theta+a_theta*time_step;
        theta=theta+v_theta*time_step;
        v_x=v_x+a_x*time_step;
        x=x+v_x*time_step;
        % draw new state
        figure(h1);
        X=[x   x+cos(pi/2-theta)];
        Y=[0.2  0.2+sin(pi/2-theta)];
        obj=rectangle('Position',[x-0.4,0.1,0.8,0.1],...
              'facecolor','b');
        obj2=line(X,Y);  
        hold on
        if (F>0)
           obj3=plot(x-0.4,0.11,'r>');
        else
           obj3=plot(x+0.4,0.11,'r<');
        end %if
        pause(0.00000000001)
        delete(obj)
        delete(obj2)
        delete(obj3)
        % get new box
        [box] = next_trial_flag(x,v_x,theta,v_theta);
        if (box== -1)  % if fail
    %             reinf=get(handles.reinf_sl,'Value');
            reinf = -1;
            predicted_value=0;
            q_val(pre_state,pre_action)= q_val(pre_state,pre_action)+ ALPHA*(reinf+ GAMMA*predicted_value - q_val(pre_state,pre_action));
            [pre_state,cur_state,pre_action,cur_action,x,v_x,theta,v_theta] = reset_cart(BETA);  % reset the cart pole to initial state
            trial=trial+1;
            if (success>best)
                best=success;
            end
            success=0;
            figure(h1);
            title(strcat('Trials  ',num2str(trial),', Best success : ',num2str(best)));
        else
            success=success+1;
            reinf=0;
        end  %if (box

    end %while

    fprintf(strcat('Success at',num2str(trial+1),' trials with',num2str(success),' time steps','\n'));
    successruns(runs,1) = trial+1;
    
end
avg_success = sum(successruns)/5;
fprintf('Average number of trials to success is %4.1f\n\n',avg_success);