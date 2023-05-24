close all
clear
clc

% Gera Sinal
N = 1000;

u = 0.3*randn(N,1);

y = zeros(N,1);

v = randn(N,1)*0.03;

for k=2:N
    y(k) = 0.5*y(k-1) + u(k-1) + v(k);
end

% Estima parâmetros
Phi = [y(1:end-1) u(1:end-1)];

theta = ((Phi'*Phi)\Phi')*y(2:end);

y_val = Phi*theta;

% Plota resultados
figure
hold on
plot(y(2:end), 'k', 'LineWidth',2)
plot(y_val, '--r', 'LineWidth',1)