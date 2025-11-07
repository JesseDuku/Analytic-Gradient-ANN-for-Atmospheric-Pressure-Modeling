%% ========================================================================
%  Neural Network From Scratch — Atmospheric Pressure vs Altitude
%  ------------------------------------------------------------------------
%  This script builds and trains a simple 1–4–1 feedforward neural network
%  (1 input, 4 hidden neurons, 1 output) to model the barometric formula:
%       P(z) = P0 * exp(-M*g*z / (R*T))
%
%  Features:
%   - Implements ANN manually (no toolboxes)
%   - Computes analytic Jacobian for gradient updates
%   - Adaptive learning rate using fminsearch
%   - Tracks training loss and plots train/test predictions
%
%  Author: [Your Name]
%  ------------------------------------------------------------------------
clear; clc; close all;

%% ------------------------------------------------------------------------
%  1. Generate synthetic data (barometric pressure)
% -------------------------------------------------------------------------
P_atm = 1000;          % pressure at sea level (Pa)
T_a = 293;             % air temperature (K)
g = 9.8;               % gravitational acceleration (m/s^2)
R = 8.314;             % universal gas constant (J/mol*K)
M_a = 0.0289;          % molar mass of air (kg/mol)

% Training and testing altitude grids
z_train = (0:20:10000)';      
z_test  = (10:20:9990)';     

% True barometric pressures
P_train = P_atm * exp(-(M_a * g * z_train) / (R * T_a)); 
P_test  = P_atm * exp(-(M_a * g * z_test) / (R * T_a)); 

% Normalize data
z_mu = mean(z_train); z_std = std(z_train);
P_mu = mean(P_train); P_std = std(P_train);

z_train_norm = (z_train - z_mu) / z_std; 
P_train_norm = (P_train - P_mu) / P_std; 
z_test_norm  = (z_test - z_mu) / z_std; 

%% ------------------------------------------------------------------------
%  2. Initialize neural network parameters
% -------------------------------------------------------------------------
theta = rand(13, 1);    % 13 weights/biases
alpha = 0.01;           % initial learning rate
n_iter = 1000;          % number of iterations
ssr = zeros(n_iter, 1); % store sum of squared residuals

%% ------------------------------------------------------------------------
%  3. Train ANN using gradient descent with adaptive step size
% -------------------------------------------------------------------------
fprintf('Training neural network...\n');
for i = 1:n_iter
    [P_pred, J] = ANNN_model(theta, z_train_norm); % forward + Jacobian
    residual = P_train_norm - P_pred;
    grad = -2 * J' * residual;

    % Define function for step-size optimization
    SSR_2 = @(alpha) sum((P_train_norm - ...
        ANNN_model(theta - alpha * grad, z_train_norm)).^2);

    alpha_opt = fminsearch(SSR_2, alpha, optimset('Display','off'));
    theta = theta - alpha_opt * grad;
    ssr(i) = sum(residual.^2);

    if mod(i,100)==0
        fprintf('Iteration %d / %d | SSR = %.4f\n', i, n_iter, ssr(i));
    end
end

%% ------------------------------------------------------------------------
%  4. Predictions (denormalized)
% -------------------------------------------------------------------------
P_pred_train_norm = ANNN_model(theta, z_train_norm);
P_pred_train = P_pred_train_norm * P_std + P_mu;

P_pred_test_norm = ANNN_model(theta, z_test_norm);
P_pred_test = P_pred_test_norm * P_std + P_mu;

res_test = (P_test - P_mu) / P_std - P_pred_test_norm;
SSR_test = sum(res_test.^2);

fprintf('\nFinal Training SSR (normalized): %.6f\n', ssr(end));
fprintf('Final Test SSR (normalized): %.6f\n', SSR_test);

%% ------------------------------------------------------------------------
%  5. Plots
% -------------------------------------------------------------------------
figure; 
plot(1:n_iter, ssr, 'b-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('SSR'); grid on;
title('Training Loss (Sum of Squared Residuals)');

figure;
plot(z_train, P_train, 'k--', 'LineWidth', 1.5); hold on;
plot(z_train, P_pred_train, 'r-', 'LineWidth', 1.5);
xlabel('Altitude z (m)'); ylabel('Pressure (Pa)');
legend('Measured', 'Predicted'); title('Training Set');

figure;
plot(z_test, P_test, 'k--', 'LineWidth', 1.5); hold on;
plot(z_test, P_pred_test, 'r-', 'LineWidth', 1.5);
xlabel('Altitude z (m)'); ylabel('Pressure (Pa)');
legend('Measured', 'Predicted'); title('Test Set');

%% ------------------------------------------------------------------------
%  ANN Model Definition (1-4-1 Network with Analytic Jacobian)
% -------------------------------------------------------------------------
function [output, Jacobian] = ANNN_model(theta, input_t)
    n = length(input_t);
    output = zeros(n, 1);
    Jacobian = zeros(n, length(theta));

    % Unpack parameters
    w1 = theta(1);  w2 = theta(2);  w3 = theta(3);  w4 = theta(4);
    b1 = theta(5);  b2 = theta(6);  b3 = theta(7);  b4 = theta(8);
    v1 = theta(9);  v2 = theta(10); v3 = theta(11); v4 = theta(12);
    b5 = theta(13);

    for t = 1:n
        x = input_t(t); 
        z1 = w1*x + b1; z2 = w2*x + b2; z3 = w3*x + b3; z4 = w4*x + b4;
        a1 = 1/(1+exp(-z1)); a2 = 1/(1+exp(-z2));
        a3 = 1/(1+exp(-z3)); a4 = 1/(1+exp(-z4));
        y = v1*a1 + v2*a2 + v3*a3 + v4*a4 + b5;
        output(t) = y;

        % Derivatives for Jacobian
        d1 = exp(-z1) / (1 + exp(-z1))^2;
        d2 = exp(-z2) / (1 + exp(-z2))^2;
        d3 = exp(-z3) / (1 + exp(-z3))^2;
        d4 = exp(-z4) / (1 + exp(-z4))^2;

        % w1–w4
        Jacobian(t,1) = v1 * d1 * x;
        Jacobian(t,2) = v2 * d2 * x;
        Jacobian(t,3) = v3 * d3 * x;
        Jacobian(t,4) = v4 * d4 * x;

        % b1–b4
        Jacobian(t,5) = v1 * d1;
        Jacobian(t,6) = v2 * d2;
        Jacobian(t,7) = v3 * d3;
        Jacobian(t,8) = v4 * d4;

        % v1–v4
        Jacobian(t,9)  = a1;
        Jacobian(t,10) = a2;
        Jacobian(t,11) = a3;
        Jacobian(t,12) = a4;

        % b5
        Jacobian(t,13) = 1;
    end
end
