close all
clear
clc

data_path = '/home/matheus/Documents/repositories/bispectrum_real_data_analysis/bispectrum_real_data_analysis/data/rats';

% Salina 
experiment = "Salina";

% PreTreino
% experiment = "PreTreino";


file = sprintf("G5-R1_%s.mat", experiment);

disp("loading file " + file + "...")
load('-mat', data_path+"/"+file)
%%

decimate_ratio = 7;
data = [decimate(data(1, :), decimate_ratio); decimate(data(2, :), decimate_ratio)];
srate = srate/decimate_ratio;

%%

% Time window
short_fft.timewin    = 10400; % in ms

% Convert time window to points
short_fft.timewinpnts  = hamming(round((short_fft.timewin/1000)*srate));

% Number of overlap samples
short_fft.overlap = 90;
short_fft.noverlap = floor(short_fft.overlap*0.01*(round(short_fft.timewin/(1000/srate))));

% nFFT
% short_fft.nFFT = 2^nextpow2(round(short_fft.timewin/(1000/srate)));
short_fft.nFFT = 2^16; 

% 
% Spectrogram
% lines: frequencies / columns: time / third dimension: channels
% 
% If  applied notch and band pass filter according pre_prossing.m define -> 2

for ii = 1:size(data,1)
    if ii == 1
       [short_fft.data(:,:,ii),short_fft.freq,short_fft.time] = spectrogram(data(ii,:),short_fft.timewinpnts,short_fft.noverlap,short_fft.nFFT, srate);
    else
        short_fft.data(:,:,ii) = spectrogram(data(ii,:),short_fft.timewinpnts,short_fft.noverlap,short_fft.nFFT, srate);
    end
end


clear ('ii','jj','not')


steps = diff(short_fft.freq); % according to the fft time window

freq2plot = 51.7:steps(1):55.7;
closestfreq = dsearchn(short_fft.freq,freq2plot');

f2 = figure;
set(gcf,'color','w');
colormap("hot");


% Add titles in subplot introduced in 18b
sgtitle({'Amplitude Spectrum via short-window FFT';['(window = ' num2str(short_fft.timewin./1000) 's' ' - ' 'nFFT = ' num2str(short_fft.nFFT) ' - ' 'overlap = ' num2str(short_fft.overlap) '%)' ]}) 

subplot(2,1,1)
contourf(short_fft.time,short_fft.freq(closestfreq),abs(short_fft.data(closestfreq,:,1)),80,'linecolor','none');
yticks([52, 53, 53.71, 54.5, 55])
xlabel('Time (s)','FontSize',14), ylabel({'CS Modulating';'Frequency (Hz)'},'FontSize',14)
colorbar

subplot(2,1,2)
contourf(short_fft.time,short_fft.freq(closestfreq),abs(short_fft.data(closestfreq,:,2)),80,'linecolor','none');
yticks([52, 53, 53.71, 54.5, 55])
xlabel('Time (s)','FontSize',14), ylabel({'Inferior Colliculus';'Frequency (Hz)'},'FontSize',14)
colorbar

filename_output = sprintf("spectrogram_%s_data.eps", experiment);
print(filename_output, "-depsc")
