close all
clear
clc

data_path = '/home/matheus/Documents/repositories/bispectrum_real_data_analysis/bispectrum_real_data_analysis/data/rats';

paths = dir(data_path);
filenames = string({paths.name});
filenames = filenames(3:end);

for file = filenames
    disp("loading file " + file + "...")
    load('-mat', data_path+"/"+file)
    time = linspace(0, (length(data(1, :))-1)*(1/srate), length(data(1, :)));
    
    index = ["Time";
        "CS_modulating";
        "Inferior_colliculus";];
    
    matrix = [index, [time', data']']';
    
    output_filename = strrep(file, ".mat", ".csv");
    disp("writing " + output_filename + "...")
    writematrix(matrix, set_path(output_filename));
    
    disp("plotting...")
    
    figure
    subplot(211)
    plot(time, data(1, :));
    title(erase(file, ".mat"))
    xlabel("time [s]");
    ylabel("CS Modulating");
    subplot(212)
    plot(time, data(2, :));
    xlabel("time [s]");
    ylabel("IC");

    disp("Done")
end


function [new_path] = set_path(path)
    relative_path = "../data/";
    new_path = GetFullPath(fullfile(cd, relative_path)) + path;
end
