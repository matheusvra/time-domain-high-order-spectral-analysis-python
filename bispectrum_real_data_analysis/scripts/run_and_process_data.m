close all
clear
clc

load_data;

matrix = string(cell2mat(data.data));
time = string(data.timev);

index = ["Time";
    "CS_modulating";
    "mPFC_pre_limbic_1";
    "mPFC_pre_limbic_2";
    "mPFC_infra_limbic_1" ;
    "mPFC_infra_limbic_2" ;
    "Hippocampus_CA1" ;
    "Hippocampus_MOL_layer" ;
    "Hippocampus_GD_1" ;
    "Hippocampus_GD_2" ;
    "Amygdala_lateral_1" ;
    "Amygdala_lateral_2" ;
    "Amygdala_basolateral_1";
    "Amygdala_basolateral_2" ;
    "Inferior_colliculus_1" ;
    "Inferior_colliculus_2" ;
    "Inferior_colliculus_3" ;
    "Inferior_colliculus_4"];

matrix = [index, [time, matrix']']';

writematrix(matrix, set_path('data_matrix.csv'));

header_timedelta_str = ["start"; "end"];

events_index = [header_timedelta_str, string(data.events.idx)']';
writematrix(events_index, set_path('events_index.csv'));

events_index_timestamp = [header_timedelta_str, string(data.events.idx_t)']';
writematrix(events_index_timestamp, set_path('events_index_timestamp.csv'));

events_behavior_TS_LFPindex = [header_timedelta_str, string(data.events.behavior.TS_LFPindex)']';
writematrix(events_behavior_TS_LFPindex, set_path('events_behavior_TS_LFPindex.csv'));

events_behavior_TS_LFPsec = [header_timedelta_str, string(data.events.behavior.TS_LFPsec)']';
writematrix(events_behavior_TS_LFPsec, set_path('events_behavior_TS_LFPsec.csv'));


function [new_path] = set_path(path)
    relative_path = "../data/";
    new_path = GetFullPath(fullfile(cd, relative_path)) + path;
end
