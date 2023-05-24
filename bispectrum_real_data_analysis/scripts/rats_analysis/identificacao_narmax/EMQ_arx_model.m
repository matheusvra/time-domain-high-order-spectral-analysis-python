close all
clear
clc

data = load('boxjenk.dat');

U = data(1:148,1);
Y = data(1:148,2);

U2 = data(148:end,1);
Y2 = data(148:end,2);

[Theta_star, Xi_EMQ, var_Xi_EMQ, ny_regs, nu_regs, nv_regs] = EMQ_arx_modelf(Y, U);

Y_valid = arx(U2, Theta_star);

figure()
hold on
plot(Y2, 'k')
plot(Y_valid, '--r')
legend('boxjenk','estimado')


function Y = arx(u, theta, ny_regs, nu_regs)
    Y = zeros(length(u));
    for k=2:length(u)
       for i=1:ny_regs
           Y(k) = Y(k) + theta(i)*y(k-i);
       end
       for i=1:nu_regs
           Y(k) = Y(k) + theta(i+ny_regs)*u(k-i);
       end
    end
end

function [Theta_star, Xi_EMQ, var_Xi_EMQ, ny_regs, nu_regs, nv_regs] = EMQ_arx_modelf(Y,U)
    %ARX_MODEL_1622 Summary of this function goes here
    %   Y is the model output from the given data.
    %   U is the model input from the given data.

    % Output
    Yout = [];

    % Regressors
    ny_regs = 4;
    nu_regs = 4;
    nv_regs = 4;

    % Initial iteration : the number of init conditions plus 1
    k_start = max([ny_regs nu_regs])+1;

    % Initial conditions
    for k = 1:k_start-1
        Yout = [Yout Y(k)];
    end

    % Generate regressors matrix
    Psi = [];
    for k = k_start:length(U)
        psi = [];
        % Regressors of y
        for ny = 1:ny_regs
            psi = [psi Y(k-ny)];
        end
        % Regressors of u
        for nu = 1:nu_regs
            psi = [psi U(k-nu)];
        end
        Psi = [Psi; psi];
    end

    % Generate Y for estimator
    Y_estimator = Y(k_start:length(U));
    U_estimator = U;

    % Estimate parameters
    Theta = ((Psi'*Psi)\Psi')*Y_estimator;

    % Residuals
    Xi = Y_estimator - Psi*Theta;
    % Xi_EMQ = [zeros(k_start-1,1); Xi];
    Xi_EMQ = Xi;

    % Iteration counter for EMQ: starts always at 2
    iteration = 1;

    % Parameter derivative: while the parameter derivative is greater than a
    % value, it has still not converged
    param_derivative = Inf;
    converged_value = 0.1;
    iterations_to_converge = 5;

    % Variance of residuals
    var_Xi_EMQ = [];

    % Estimated parameters of EMQ
    Theta_star = [];
    % Subtract 1 because the noise in the psi matrix starts at (k-1)
    last_theta_star = zeros(ny_regs+nu_regs+nv_regs,1); 

    % Estimation of theta_EMQ
    while iteration < iterations_to_converge

        % Generate Psi_matrix for EMQ
        Psi_EMQ = [];
        for k = 1:ny_regs
            Psi_EMQ = [Psi_EMQ Y_estimator(k_start-k:end-k)];
        end
        for k = 1:nu_regs
            Psi_EMQ = [Psi_EMQ U_estimator(k_start-k:end-k)];
        end
        for k = 1:nv_regs
            Psi_EMQ = [Psi_EMQ Xi_EMQ(k_start-k:end-k)]; 
        end
    %     Psi_EMQ = Psi_EMQ(k_start:end,:)

        % Estimate parameters
        Theta_star = inv([Psi_EMQ'*Psi_EMQ])*Psi_EMQ'*Y_estimator(k_start:end);
        param_derivative = Theta_star - last_theta_star;
        param_derivative = mean(param_derivative);
        last_theta_star = Theta_star;

        % Estimate residuals EMQ
        Xi_EMQ = [zeros(k_start-1,1); Y_estimator(k_start:end) - Psi_EMQ*Theta_star];
    %     Xi_EMQ = [Y_estimator(k_start:end) - Psi_EMQ*Theta_star];

        var_Xi = (Xi_EMQ'*Xi_EMQ)/(length(Xi_EMQ)-length(Theta_star));
        var_Xi_EMQ = [var_Xi_EMQ var_Xi];

        iteration = iteration + 1;
    end

end

