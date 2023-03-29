close all
clear
clc

data_path = '/home/matheus/Documents/repositories/bispectrum_real_data_analysis/bispectrum_real_data_analysis/data/rats_simoes';



file = 'R1_Teste.mat';
disp("loading file " + file + "...")
load('-mat', data_path+"/"+file)
time = linspace(0, (length(data(1, :))-1)*(1/srate), length(data(1, :)));

disp("plotting...")

figure
plot(data(1, :));
title(strrep(erase(file, ".mat"), "_", "-"))
xlabel("time [s]");
ylabel("CS Modulating");

