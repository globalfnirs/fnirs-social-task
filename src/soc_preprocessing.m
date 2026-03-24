%{
Script for 60 mo social data signal preprocessing
Authors: Borja Blanco and Johann Benerradi
------------------------------------------
%}

%% Clean up
clear; close all; clc

%% Variables
currentFile = which(mfilename);
[pathScripts, ~, ~] = fileparts(currentFile);
pathHomer = fullfile(pathScripts, '..', '..', 'homer2');
pathData = fullfile(pathScripts, '..', '..', 'data', 'raw', '60mo', 'nirs');

sf = 10;  % Hz

sci_th = 0.7;
power_th = 0.1;

tMotion = 1.0;
tMask = 1.0;
STDEVthresh = 15.0;
AMPthresh = 0.5;

p = 0.99;
iqr = 0.80;

tRange = [-2.0 20.0];

hpf = 0.020;
lpf = 0.60;

dpf = [5.44 4.44];

percLook = 60.0;


% Channel reordering and discarding of removed channels
chLab = [34, 23, 35, 14, 24, 36, 11, 15, 25, 3, 12, 16, 1, 4, ...
         13, 2, 5, 29, 26, 30, 20, 27, 31, 17, 21, 28, 8, 18, ...
         22, 6, 9, 19, 7, 10];
chSrc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14];
chDet = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];


%% Processing
addpath(genpath(pathScripts))
addpath(genpath(pathHomer))
cd(pathData)
sub = dir('*.nirs');


%% Iterate over subjects
% nsub = 3;
for nsub = 1:length(sub)
    data = load(sub(nsub).name, '-mat');
    
    % Keep only social task markers and reorder them
    allCond = cell2mat(data.CondNames);
    newS = zeros(size(data.t, 2), 4);
    newS(:, 1) = data.s(:, find(allCond=='C'));
    newS(:, 2) = data.s(:, find(allCond=='S'));
    newS(:, 3) = data.s(:, find(allCond=='V'));
    newS(:, 4) = data.s(:, find(allCond=='N'));
    data.s = newS;
    data.CondNames = [{'C'} {'S'} {'V'} {'N'}];
    
    % Reorder channels
    n_ch_org = length(data.SD.MeasList)/length(data.SD.Lambda);
    data.d = data.d(:, [chLab, chLab+n_ch_org]);
    data.SD.MeasList = data.SD.MeasList([chLab, chLab+n_ch_org], :);
    data.SD.MeasListAct = data.SD.MeasListAct([chLab, chLab+n_ch_org], :);
    data.SD.SpringList = data.SD.SpringList(chLab, :);
    data.SD.SrcPos = data.SD.SrcPos(chSrc, :);
    data.SD.DetPos = data.SD.DetPos(chDet, :);

    % Rotate channel positions
    for nch = 1:length(data.SD.SrcPos)
        x = data.SD.SrcPos(nch, 1);
        y = data.SD.SrcPos(nch, 2);
        new_x = x * cos(-pi/2) + y * sin(-pi/2);
        new_y = -x * sin(-pi/2) + y * cos(-pi/2);
        data.SD.SrcPos(nch, 1) = new_x + 40;
        data.SD.SrcPos(nch, 2) = new_y - 40;
    end
    for nch = 1:length(data.SD.DetPos)
        x = data.SD.DetPos(nch, 1);
        y = data.SD.DetPos(nch, 2);
        new_x = x * cos(-pi/2) + y * sin(-pi/2);
        new_y = -x * sin(-pi/2) + y * cos(-pi/2);
        data.SD.DetPos(nch, 1) = new_x + 40;
        data.SD.DetPos(nch, 2) = new_y - 40;
    end
    
    data.SD.MeasList(find(data.SD.MeasList(:, 1) > 13), 1) = data.SD.MeasList(find(data.SD.MeasList(:, 1) > 13), 1) - 1;
    data.SD.MeasList(find(data.SD.MeasList(:, 1) > 10), 1) = data.SD.MeasList(find(data.SD.MeasList(:, 1) > 10), 1) - 1;

    name_split = split(sub(nsub).name, '.');
    % save(['fixed_', cell2mat(name_split(1)), '.nirs'], '-struct', 'data')

    
    % Add metadata
    data.name = sub(nsub).name;
    data.DPF = dpf;
    data.sf = sf;
    data.nChannels = size(data.d, 2);
    
    %% 1- Mark channels with low attenuation reading (attenuation < 3e-4)
    data.SD.MeasListAct(mean(data.d, 1) < 3e-4, :) = 0;
    
    %% 2- Mark channels based on SCI and cardiac power
    % (SCI < 0.7 & cardiac power < 0.1 uV for more than 70 % of recording with
    % a non-overlapping window of 3 sec)
    [data.bad_links, data.bad_windows] = sci(data, '', sci_th, power_th);
    data.SD.MeasListAct(data.bad_links) = 0;
    
    %% 3- Convert to OD
    data.OD = hmrIntensity2OD(data.d);
    
    %% 4a- Detect motion artefacts by channel
    [~, tIncCh] = hmrMotionArtifactByChannel(data.OD, data.sf, data.SD, ...
                                             [], tMotion, tMask, ...
                                             STDEVthresh, AMPthresh);
    
    %% 4b- Spline correction on channels with motion artefacts (p = 0.99)
    data.spline = hmrMotionCorrectSpline (data.OD, data.t, data.SD, tIncCh, p);
    
    %% 5- Wavelet motion artefact correction (iqr = 0.8)
    data.wav = hmrMotionCorrectWavelet(data.spline, data.SD, iqr);
    
    %% 6- Reject trials with motion artefacts
    % Detect motion artefacts
    [tInc, ~] = hmrMotionArtifactByChannel(data.wav, data.sf, data.SD, ...
                                                [], tMotion, tMask, ...
                                                STDEVthresh, AMPthresh);
    
    % Reject trials with motion artefact
    [data.s, ~] = enStimRejection(data.t, data.s, tInc, [], tRange);
    
    %% 7- Bandpass filter
    data.filt = hmrBandpassFilt(data.wav, data.sf, hpf, lpf);
    
    %% 8- MBLL
    data.conc = hmrOD2Conc(data.filt,data.SD,data.DPF);
    
    %% 9- Reject trials based on looking time
    lt_filename = ['../autocoder/' cell2mat(name_split(1)) '_LT_Autocoder.mat'];
    
    [data.s, data.LTFile] = looking_time(data, lt_filename, percLook);

    %% Save processed data
    save(['../processed/', cell2mat(name_split(1)), '.nirs'], '-struct', 'data')
    
    %% Block averaging
    [dcAvg, dcAvgStd, tHRF, nTrials, dcSum2, dcTrials] = hmrBlockAvg(...
        data.conc, data.s, data.t, tRange);

    % Make sure both wavelengths are bad and replace bad channels by NaNs
    i_wl1 = find(data.SD.MeasList(:,end) == 1);
    i_wl2 = find(data.SD.MeasList(:,end) == 2);
    wl1_bad = find(data.SD.MeasListAct(i_wl1) == 0);
    wl2_bad = find(data.SD.MeasListAct(i_wl2) == 0);
    bad_channels = union(wl1_bad, wl2_bad);
    data.SD.MeasListAct(bad_channels) = 0;
    data.SD.MeasListAct(bad_channels+length(i_wl1)) = 0;

    dcAvg(:, :, bad_channels, :) = nan;
    dcAvgStd(:, :, bad_channels, :) = nan;
    dcSum2(:, :, bad_channels, :) = nan;

    % Save results
    results.s = data.s;
    results.CondNames = data.CondNames;
    results.SD = data.SD;
    results.LTFile = data.LTFile;
    results.dcAvg = dcAvg;
    results.dcAvgStd = dcAvgStd;
    results.tHRF = tHRF;
    results.nTrials = nTrials;
    results.dcSum2 = dcSum2;
    results.dcTrials = dcTrials;
    save(['../../../results/60mo/results_', cell2mat(name_split(1)), '.mat'], 'results')

end
