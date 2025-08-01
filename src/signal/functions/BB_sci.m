%{
Function for channel quality checking using SCI
Authors: Borja Blanco & Luca Pollonini
--------------------------------------
%}

function [bad_links, bad_window_inf] = BB_sci(data, path_figures, sci_th, power_th)

% fcut: 1x2 array [fmin fmax] representing the bandpass of the cardiac pulsation (default [0.5 2.5])
% window: length in seconds of the window to partition the signal with (defaut: 5)
% overlap: fraction overlap (0..0.99) between adjacent windows (default: 0, no overlap)
% lambda_mask: binary array mapping the selected two wavelength to correlate
% (default: [1 1 ...], the first two encountered, no matter how many there are)

close all

% Values for plots
plot_quality_plot = 0;
% plot_bad = 1;
% save_report_table = 0;

% Define parameters
fs = data.sf;
fcut = [1.5 3.5];
window = 3;
overlap = 0;

fcut_min = fcut(1);
fcut_max = fcut(2);
if fcut_max >= (fs)/2
    fcut_max = (fs)/2 - eps;
    uiwait(warndlg('The highpass cutoff has been reduced to Nyquist sampling rate. This setting will be saved for future use.'));
end

% Create filter that will allow passing only the cardiac pulse
[B1,A1]=butter(1,[fcut_min*(2/fs) fcut_max*(2/fs)]);

report_table = table();

% Load data
display (['Working on ' data.name])

% Cut recording to task-related data - BB
%ev_idx = sum(data.s,2);
%ev_idx = find(ev_idx);
%raw_d = data.d(ev_idx(1):ev_idx(end),:);
ev_idx = [1 size(data.d,1)];
raw_d = data.d;
% Create matrices for storing results
lambdas = unique(data.SD.MeasList(:,4));
lambda_mask = zeros(length(lambdas),1);
lambda_mask(1)=1;
lambda_mask(2)=1;
nirs_data = zeros(length(lambdas),size(raw_d,1),size(raw_d,2)/length(lambdas));
cardiac_data = zeros(length(lambdas),size(raw_d,1),size(raw_d,2)/length(lambdas)); %d filtred in the HR band

% Filter everything but the cardiac component --> cardiac_data
% Extract info for each wavelength
for j = 1:length(lambdas)
    idx = find(data.SD.MeasList(:,4) == lambdas(j));
    nirs_data(j,:,:) = raw_d(:,idx);
    filtered_nirs_data = filtfilt(B1,A1,squeeze(nirs_data(j,:,:)));
    cardiac_data(j,:,:) = filtered_nirs_data./repmat(std(filtered_nirs_data,0,1),size(filtered_nirs_data,1),1); % Normalized heartbeat (independ from the machine used or type of signal used)
end
cardiac_data = cardiac_data(find(lambda_mask),:,:);

% Compute Scalp coupling Index
% Divide data in 30 sample windows = 3 seconds windows
% Compute normalized power on each window
overlap_samples = floor(window*fs*overlap);
window_samples = floor(window*fs);
n_windows = floor((size(cardiac_data,2)-overlap_samples)/(window_samples-overlap_samples));
sci_array = zeros(size(cardiac_data,3),n_windows);    % Number of optode is from the user's layout, not the machine
power_array = zeros(size(cardiac_data,3),n_windows);
fpower_array = zeros(size(cardiac_data,3),n_windows);
for j = 1:n_windows % for each window
    interval = (j-1)*window_samples-(j-1)*(overlap_samples)+1 : j*window_samples-(j-1)*(overlap_samples); % extract interval, samples
    cardiac_window = cardiac_data(:,interval,:); % extract cardiac data on the interval
    for k = 1:size(cardiac_window,3) % for each channel
        
        % compute cross correlation between wavelengths
        similarity = xcorr(squeeze(cardiac_window(1,:,k)),squeeze(cardiac_window(2,:,k)),'unbiased');
        
        if any(abs(similarity)>eps)
            
            similarity = length(squeeze(cardiac_window(1,:,k)))*similarity./sqrt(sum(abs(squeeze(cardiac_window(1,:,k))).^2)*sum(abs(squeeze(cardiac_window(2,:,k))).^2));
            
        else
            
            warning('Similarity results close to zero');
            
        end
        
        similarity(isnan(similarity)) = 0; 
        [pxx,f] = periodogram(similarity,hamming(length(similarity)),length(similarity),fs,'power');
        
        %similarity = xcorr(squeeze(cardiac_window(1,:,k)),squeeze(cardiac_window(2,:,k)),'unbiased');  %cross-correlate the two wavelength signals - both should have cardiac pulsations
        %similarity = length(squeeze(cardiac_window(1,:,k)))*similarity./sqrt(sum(abs(squeeze(cardiac_window(1,:,k))).^2)*sum(abs(squeeze(cardiac_window(2,:,k))).^2));  % this makes the SCI=1 at lag zero when x1=x2 AND makes the power estimate independent of signal length, amplitude and Fs
        %[pxx,f] = periodogram(similarity,hamming(length(similarity)),length(similarity),fs,'power'); % Power spectrum estimate of the similarity vector
        [pwrest,idx] = max(pxx(f<fcut_max)); %% check this
        sci = similarity(length(squeeze(cardiac_window(1,:,k)))); % correlation in initial position of the 2 wavelength signals
        power=pwrest;
        fpower=f(idx);
        sci_array(k,j) = sci;    % Adjust not based on machine
        power_array(k,j) = power; % peak of the frequency, the amplitude or power at the specific frequency
        fpower_array(k,j) = fpower; %frequency of the peak, Hz
    end
end

%% Summary analysis

% CB replaced mean with nanmean
mean_sci_link  = nanmean(sci_array,2); % mean sci across channels
std_sci_link  = std(sci_array,0,2); % std sci across channels
good_sci_link = sum(sci_array>sci_th,2)/size(sci_array,2);
mean_sci_window  = nanmean(sci_array,1); % mean sci across windows
std_sci_window  = std(sci_array,0,1); % std sci across windows
good_sci_window = sum(sci_array>sci_th,1)/size(sci_array,1);

mean_power_link  = nanmean(power_array,2); % mean power across channels
std_power_link  = std(power_array,0,2); % std power across channels
good_power_link = sum(power_array>power_th,2)/size(power_array,2);
mean_power_window  = nanmean(power_array,1);
std_power_window  = std(power_array,0,1);
%   good_power_window = sum(power_array>0.1,1)/size(power_array,1);
good_power_window = sum(power_array>power_th,1)/size(sci_array,1);

% SCI changed from 0.80 to 0.70
combo_array = (sci_array >= sci_th) & (power_array >= power_th); % thresholds (temporal and spectral)
mean_combo_link  = nanmean(combo_array,2); 
std_combo_link  = std(combo_array,0,2);
good_combo_link = sum(combo_array>0.1,2)/size(combo_array,2);% combo array is logical
% the 0.1 is not a threshold is just to distinguish 1 from 0
mean_combo_window  = nanmean(combo_array,1);
std_combo_window  = std(combo_array,0,1);
%   good_combo_window = sum(combo_array>0.1,1)/size(combo_array,1);
good_combo_window = sum(combo_array>0.1,1)/size(sci_array,1);
% the 0.1 is not a threshold is just to distinguish 1 from 0

%% Plot summary
%     cd 'HR_LP(0.70)'
if plot_quality_plot
    Fig1 = figure;
    % figure ('Name','SCI and Peak Power Plot')
    colormap(gray)
    subplot(3,1,1)
    imagesc(sci_array,[0 1])
    %axis equal
    axis tight
    h = colorbar;
    h.FontSize = 9;
    set(get(h,'label'),'string','SCI');
    ylabel('Channel #')
    title([data.name 'SCI and Peak Power Plot'], 'interpreter', 'none')
    subplot(3,1,2)
    imagesc(power_array,[0 0.1])
    %axis equal
    axis tight
    h = colorbar;
    h.FontSize = 9;
    set(get(h,'label'),'string','Peak Power');
    ylabel('Channel #')
    subplot(3,1,3)
    imagesc(combo_array,[0 1])
    %axis equal
    axis tight
    h = colorbar;
    h.FontSize = 9;
    set(get(h,'label'),'string','Overall Quality');
    ylabel('Channel #')
    xlabel('Window #')
end
%cd(path_figures)
%saveas(Fig1,[data.name, 'hr_sci.tiff'], 'tiffn');
%close
%% Detect artifacts and bad links
bad_links = find(good_combo_link<0.7); % channels
bad_links = [bad_links; bad_links+size(data.d,2)/2];
bad_windows = find(good_combo_window<0.7); % window

% Store info, adjust index as data has been cut to beginning-end experiment
bad_window_inf = zeros(length(bad_windows), 3);
for j = 1:length(bad_windows) % for each window
    interval = (bad_windows(j)-1)*window_samples-(bad_windows(j)-1)*(overlap_samples)+1 : bad_windows(j)*window_samples-(bad_windows(j)-1)*(overlap_samples); % extract interval, samples
    bad_window_inf(j,1) = bad_windows(j);
    bad_window_inf(j,2) = ev_idx(1) + interval(1); % add what has been cut from beginning
    bad_window_inf(j,3) = ev_idx(1) + interval(end);
end


% save ([data.name,'_HR_channels.mat'],'bad_links','good_combo_link')
%
% % Plot outcome
% new_table_row = table(i, {bad_links'}, {bad_windows});
% report_table = [report_table; new_table_row];
%
% if plot_bad
%     n_col = ceil(length(bad_links)/5);
%     if ~isempty(bad_links)
%         figure('Name', 'Bad Channels')
%         for j = 1:length(bad_links)
%             subplot(5,n_col,j)
%             hold on
%             plot(cardiac_data(1,:,bad_links(j)),'b')
%             plot(cardiac_data(2,:,bad_links(j)),'r')
%             axis tight
%             ylim([-4 4])
%             title (['channel ', num2str(bad_links(j))])
%         end
%     end
%     if ~isempty(bad_links)
%         namefig=[filenm,'_Bad Channels.tif'];
%         print('-dtiff','-zbuffer',namefig)
%         close
%     end
%
%     n_col = ceil(length(bad_windows)/5);
%     if ~isempty(bad_windows)
%         figure('Name','Bad Windows')
%         for j = 1:length(bad_windows)
%             subplot(5,n_col,j)
%             idx = find(~combo_array(:,bad_windows(1)));
%             interval = (bad_windows(j)-1)*window_samples-(bad_windows(j)-1)*(overlap_samples)+1 : bad_windows(j)*window_samples-(bad_windows(j)-1)*(overlap_samples);
%             hold on
%             plot(cardiac_data(1,interval,idx(1)),'b')
%             plot(cardiac_data(2,interval,idx(1)),'r')
%             axis tight
%             ylim([-4 4])
%             title (['window ', num2str(bad_windows(j))])
%         end
%     end
%
% %     if ~isempty(bad_windows)
% %         namefig=[data.name,'_Bad Windows.tif'];
% %         print('-dtiff','-zbuffer',namefig)
% %         close
% %     end
%
% end
%
% %     cd ..
% clear raw
%
% report_table.Properties.VariableNames = {'file_idx','Bad_Links','Bad_Windows'};

%% Save on Excel file
% for i=1:size(report_table,1)
%     a = report_table.Bad_Links{i};
%     b = report_table.Bad_Windows{i};
%     a1 = num2str(a);
%     b1 = num2str(b);
%     report_table.Bad_Links{i} = a1;
%     report_table.Bad_Windows{i} = b1;
% end
%
% if save_report_table
%     writetable(report_table,[data.name 'Quality_Report.xls']);
% end

