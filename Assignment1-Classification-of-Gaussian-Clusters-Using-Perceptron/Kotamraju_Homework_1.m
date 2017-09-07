% Instructions to run:
% 1. Change the var value to change the Variance from 1 to any
% other value.
% 2. Change the net.layers{1}.transferFcn to either logsig or
% tansig or hardlim. This is the activation function.
% 3. Make sure to comment out the lines corresponding to other
% activation functions.
% 4. I took the target values as -1 and +1. So, to plot the
% confusion matrix, I changed -1s to zeros. 
clear; 
%% Data Generation
clc; var = 1;
mu1 = [0 0];       % Mean
sigma1 = [var 0; 0 var]; %Co-Variance vector
m1 = 1000;          % Number of data points.
mu2 = [2.5 0];
sigma2 = [var 0; 0 var];
m2 = 1000;

% Generate sample points with the specified means and covariance matrices.
rng(42);
R1 = chol(sigma1);
X1 = randn(m1, 2) * R1;
X1 = X1 + repmat(mu1, size(X1, 1), 1);

rng(42);
R2 = chol(sigma2);
X2 = randn(m2, 2) * R2;
X2 = X2 + repmat(mu2, size(X2, 1), 1);
X = [X1; X2];

figure(1);
% Display a scatter plot of the two distributions.
hold off;
plot(X1(:, 1), X1(:, 2), 'g*');
hold on;
plot(X2(:, 1), X2(:, 2), 'r*');
set(gcf,'color','white') % White background for the figure.
title('Input clusters');

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-6, 6, gridSize);
[A, B] = meshgrid(u, u);
gridX = [A(:), B(:)];

x = transpose(X);
t = [zeros(1,1000)-1 ones(1,1000)];

%% Training Perceptron
net = perceptron;
net.layers{1}.transferFcn = 'hardlim';
net.performFcn = 'mse';
net = train(net,x,t);
%view(net)
y = net(x);
y(y<=0.5)=-1; y(y>0.5)=1; %For logsig
%y(y<=0) =-1; y(y>0) = 1; %For tansig and elliotsig
%y(y==0)=-1; %For hardlim
%% Training MLNN 
net1 = feedforwardnet;
net1.layers{1}.transferFcn = 'tansig';
net1.layers{2}.transferFcn = 'logsig';
net1 = train(net1,x,t);
%view(net1);
y1 = net1(x);
y1(y1<=0.5)=-1; y1(y1>0.5)=1;

%% Calculation of Training Accuracies
c = (y==t);
c1 = (y1==t);

train_acc = sum(c)/2000;
train_acc_1 = sum(c1)/2000;

%% Testing with Random data
clc;

rng(55);
X1_test = randn(m1, 2) * R1;
X1_test = X1_test + repmat(mu1, size(X1_test, 1), 1);

rng(55);
X2_test = randn(m2, 2) * R2;
X2_test = X2_test + repmat(mu2, size(X2_test, 1), 1);
x_test = transpose([X1_test; X2_test]);

y_test = net(x_test);
y_test(y_test<=0.5)=-1; y_test(y_test>0.5)=1; %For logsig
%y_test(y_test<=0.5)=-1; y_test(y_test>0.5)=1; %For tansig and elliotsig
%y_test(y_test==0)=-1; %For hardlim

y1_test = net1(x_test);
y1_test(y1_test<=0.5)=-1; y1_test(y1_test>0.5)=1;

c_test = (y_test == t);
c1_test = (y1_test == t);

test_acc = sum(c_test)/2000;
test_acc_1 = sum(c1_test)/2000;

%% Results
disp('Perceptron');
fprintf("Training accuracy - %f\n Test accuracy - %f\n",train_acc*100,test_acc*100);
disp('MLN');
fprintf("Training accuracy - %f\n Test accuracy - %f\n",train_acc_1*100,test_acc_1*100);
%%%%%%%% Confusion Matrices %%%%%%%%
t(t==-1) = 0; y(y==-1) = 0; y1(y1==-1)=0; y_test(y_test==-1)=0; y1_test(y1_test==-1)=0;
figure;
plotconfusion(t,y,'Perceptron Train',t,y1,'MLN Train');
figure;
plotconfusion(t,y_test,'Perceptron Test',t,y1_test,'MLN Test');
figure;
plotpv(x,t); plotpc(net.iw{1,1},net.b{1}); title('Perceptron Classification');