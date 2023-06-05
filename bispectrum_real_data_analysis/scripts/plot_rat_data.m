close all
clear
clc

data_path = '/home/matheus/Documents/repositories/bispectrum_real_data_analysis/bispectrum_real_data_analysis/data/rats';

% Salina 
experiment = "Salina";

% PreTreino
% experiment = "PreTreino";

% least worst -> most worst g6r6, g7r10, g7r4

rat_number = 10;
group = 7;

file = sprintf("G%d-R%d_%s_events.mat", group, rat_number, experiment);

disp("loading file " + file + "...")
load('-mat', data_path+"/"+file)

%%
time = linspace(0, (length(data(1, :))-1)*(1/srate), length(data(1, :)));

disp("plotting...")

index = 0:size(data, 2)-1;
figure

ic_data = data(1, :);
plot(index, ic_data);
title(strrep(erase(file, ".mat"), "_", "-"))
xlabel("time [s]");
ylabel("CS Modulating");

