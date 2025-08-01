"""
Functions for social task analysis
Author: Johann Benerradi
------------------------
"""

import glob
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from pathlib import Path
from scipy import stats
from statsmodels.stats import multitest


CONFIDENCE = 0.05


def load_results_60mo(path, cond_list):
    """
    Get grand average from processed data results.

    Parameters
    ----------
    path : str
        Path of the results folder.

    cond_list : list of str
        List of condition names to analyse.

    Returns
    -------
    grand_avg : numpy array
        Array of block grand average for all subjects of shape (n_subjects,
        n_conditions, n_channels, n_chromophores, n_timepoints).

    subj_ids : list of str
        List of subject IDs.

    rejected : list of tuples
        List of rejected subjects with reason.
    """
    result_files = glob.glob(path + '/*.mat')

    grand_avg = np.empty((0, len(cond_list), 34, 2, 221))  # block average waveforms: (subjects, conditions, channels, chromophores, timepoints)
    subj_ids = []
    rejected = []

    c_channels = 0
    c_ltfile = 0
    c_trials = 0

    all_n_chs = []

    for result_file in result_files:
        subj_id = result_file.split('sub-')[-1].split('_')[0]

        exclude = False

        # Load block average result file:
        mat_file = scipy.io.loadmat(result_file)
        cn = np.squeeze(mat_file['results']['CondNames'][0, 0])
        cond_names = [np.squeeze(c).tolist() for c in cn]
        cond_indices = []
        for cond in cond_list:
            cond_indices.append(cond_names.index(cond))

        # Exclude subject if no looking time autocoder data:
        reason = None
        if not np.squeeze(mat_file['results']['LTFile'][0, 0]):
            exclude = True
            reason = 'trials'
            c_ltfile += 1
            # print(f"Subject {result_file} rejected (no looking time autocoder data)")

        # Exclude subject if less than 3 trials for any condition ('S', 'V' or 'N'):
        if np.any(np.squeeze(mat_file['results']['nTrials'][0, 0])[cond_indices] < 3):
            exclude = True
            reason = 'trials'
            c_trials += 1
            # print(f"Subject {result_file} rejected (number of trial remaining < 3 for any of the conditions)")

        # Extract data:
        avg = np.squeeze(mat_file['results']['dcAvg'][0, 0])[:, :2, :, cond_indices]
        avg = np.transpose(avg)[np.newaxis, :, :, :, :]

        # Exclude subject if more than 40% of excluded channels:
        if np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum() > 0.4*avg.shape[2]:
            exclude = True
            if not reason:
                reason = 'channels'
            c_channels += 1
            # print(f"Subject {result_file} rejected (more than 40% channels excluded)")
        else:
            n_chs = 34 - np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum()
            all_n_chs.append(n_chs)

        # Exclude subject:
        if exclude:
            rejected.append((subj_id, reason))
            continue

        # Append all subjects:
        grand_avg = np.append(grand_avg, avg, axis=0)
        subj_ids.append(subj_id)

    print(f"N={grand_avg.shape[0]}")
    # print(f"\tNo looking time file: {c_ltfile} subjects")
    # print(f"\tNot enough trials: {c_trials} subjects")
    # print(f"\tBad channel quality: {c_channels} subjects")

    grand_avg *= 1e6

    return grand_avg, subj_ids, rejected, all_n_chs


def load_results_infancy(path, cond_list):
    """
    Get grand average from processed data results.

    Parameters
    ----------
    path : str
        Path of the results folder.

    cond_list : list of str
        List of condition names to analyse.

    Returns
    -------
    grand_avg : numpy array
        Array of block grand average for all subjects of shape (n_subjects,
        n_conditions, n_channels, n_chromophores, n_timepoints).

    subj_ids : list of str
        List of subject IDs.

    rejected : list of tuples
        List of rejected subjects with reason.
    """
    grand_avg = np.empty((0, len(cond_list), 34, 2, 221))  # block average waveforms: (subjects, conditions, channels, chromophores, timepoints)
    subj_ids = []

    # Load group results:
    mat_file_vn = scipy.io.loadmat(f"{path}/V%3EN/groupResults.mat")
    mat_file_sc = scipy.io.loadmat(f"{path}/S%3EC/groupResults.mat")

    # Keep subjects with at least 3 trials in any condition:
    names_vn = [np.squeeze(name).tolist() for name in np.squeeze(mat_file_vn['group']['subjs'][0, 0]['name'][0, :])]
    names_sc = [np.squeeze(name).tolist() for name in np.squeeze(mat_file_sc['group']['subjs'][0, 0]['name'][0, :])]
    subj_list = [(i, name) for i, name in enumerate(names_vn) if name in names_sc]
    rejected = [(name.split('_')[0], 'trials') for name in names_vn if name not in names_sc]
    rejected += [(name.split('_')[0], 'trials') for name in names_sc if name not in names_vn]

    # Prepare condition reordering:
    cn = np.squeeze(mat_file_vn['group']['conditions'][0, 0]['CondNamesAct'][0, 0])
    cond_names = [np.squeeze(c).tolist() for c in cn]
    cond_indices = []
    for cond in cond_list:
        cond_indices.append(cond_names.index(cond))

    # Prepare channel reordering:
    ch_indices = np.array([34, 23, 35, 14, 24, 36, 11, 15, 25, 3, 12, 16, 1, 4, 13, 2, 5, 29, 26, 30, 20, 27, 31, 17, 21, 28, 8, 18, 22, 6, 9, 19, 7, 10]) - 1

    all_n_chs = []

    for i, name in subj_list:
        subj_id = name.split('_')[0]
        # subj_code = '_'.join(name.split('_')[1:]).split('_BRIGHT')[0]

        # Extract data:
        avg = np.squeeze(mat_file_vn['group']['subjs'][0, 0]['procResult'][0, i]['dcAvg'][0, 0])[:, :2, :, cond_indices]
        avg = np.transpose(avg[:, :, ch_indices, :])[np.newaxis, :, :, :, :]

        # Exclude subject if more than 40% of excluded channels:
        if np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum() > 0.4*avg.shape[2]:
            rejected.append((subj_id, 'channels'))
            continue
        else:
            n_chs = 34 - np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum()
            all_n_chs.append(n_chs)

        # Append all subjects:
        grand_avg = np.append(grand_avg, avg, axis=0)
        subj_ids.append(subj_id)

    print(f"N={grand_avg.shape[0]}")

    grand_avg *= 1e6

    return grand_avg, subj_ids, rejected, all_n_chs


def window_average(grand_avg, window=[12, 16]):
    """
    Get average over a set time window.

    Parameters
    ----------
    grand_avg : numpy array
        Array of block grand average for all subjects of shape (n_subjects,
        n_conditions, n_channels, n_chromophores, n_timepoints).

    window: list of float
        Boundaries of the time window to extract the average on, in seconds
        from the trigger onset.

    Returns
    -------
    feature_grand_avg : numpy array
        Array of features for all subjects of shape (n_subjects, n_conditions,
        n_channels, n_chromophores).
    """
    feature_grand_avg = grand_avg[:, :, :, :, int(window[0]*10+20):int(window[1]*10+20)].mean(axis=-1)

    return feature_grand_avg


def analyse_act(feature_grand_avg, condition, fdr=True, dummies=True):
    """
    Get activations for a condition.

    Parameters
    ----------
    feature_grand_avg : numpy array
        Array of features for all subjects of shape (n_subjects, n_conditions,
        n_channels, n_chromophores).

    condition : int
        Condition index from the list provided when loading the results.

    fdr : bool
        Whether to apply false discovery rate correction on the channel level.

    dummies : bool
        Whether the data should include dummy channels (when channels have been
        removed from the preprocessed data).

    Returns
    -------
    p_values : numpy array
        P-values for channel activation of shape (n_channels, n_chromophores).

    t_values : numpy array
        T-values for channel activation of shape (n_channels, n_chromophores).

    trends : numpy array
        Array of trends for channels compared to baseline of shape (n_channels,
        n_chromophores). 1 for significant increase, -1 for significant
        decrease, 0 for neither.

    activations : numpy array
        List of activations for channels compared to baseline of shape
        (n_channels,). Positive for significant activation, negative for
        significant deactivation. 2 for significant increase in HbO and
        decrease in HbR, 1 for significant increase in HbO or decrease in HbR,
        -1 for significant decrease in HbO or increase in HbR, -2 for
        significant decrease in HbO and increase in HbR, 0 for no significant
        activation (HbO and HbR both with same significant trend or both with
        no significant trend).
    """
    p_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromophores)
    t_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromophores)
    trends = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromophores)

    for channel in range(feature_grand_avg.shape[2]):
        for chromophore in range(feature_grand_avg.shape[3]):
            samples = feature_grand_avg[:, condition, channel, chromophore]
            # Get only good channels
            samples = samples[~np.isnan(samples)]
            # Warning if parametric test assumptions not verified
            if len(samples) < 30:
                print(f"Warning, only {len(samples)} sample(s) for channel No {channel+1}")

            # t-test
            if samples.mean() > 0:
                trends[channel, chromophore] = 1
                s_tt, p_tt = stats.ttest_1samp(samples, 0, alternative='greater')
            elif samples.mean() < 0:
                trends[channel, chromophore] = -1
                s_tt, p_tt = stats.ttest_1samp(samples, 0, alternative='less')
            else:
                trends[channel, chromophore] = 0
                s_tt, p_tt = stats.ttest_1samp(samples, 0)
            p_values[channel, chromophore] = p_tt
            t_values[channel, chromophore] = s_tt

    if fdr:
        for chromophore in range(feature_grand_avg.shape[3]):
            # FDR correction at channel level
            _, p_values[:, chromophore] = multitest.fdrcorrection(p_values[:, chromophore], alpha=CONFIDENCE)

    for channel in range(trends.shape[0]):
        for chromophore in range(trends.shape[1]):
            if p_values[channel, chromophore] >= CONFIDENCE:
                trends[channel, chromophore] = 0

    activations = trends[:, 0] - trends[:, 1]  # (channels)

    # Add dummy channels
    if dummies is True:
        dummy_channels = [0, 3, 20, 23]
        dummy_activations = activations.copy()
        for dummy in dummy_channels:
            dummy_activations = np.insert(dummy_activations, dummy, 0, axis=0)
        activations = dummy_activations

    return p_values, t_values, trends, activations


def plot_hrf(grand_avg, condition, activations, baseline=2, dummies=True):
    """
    Plot HRF for each channel for a condition.

    Parameters
    ----------
    grand_avg : numpy array
        Array of block grand average for all subjects of shape (n_subjects,
        n_conditions, n_channels, n_chromophores, n_timepoints).

    condition : int
        Condition index from the list provided when loading the results.

    activations : numpy array
        List of activations for channels compared to baseline of shape
        (n_channels,). Positive for significant activation, negative for
        significant deactivation. 2 for significant increase in HbO and
        decrease in HbR, 1 for significant increase in HbO or decrease in HbR,
        -1 for significant decrease in HbO or increase in HbR, -2 for
        significant decrease in HbO and increase in HbR, 0 for no significant
        activation (HbO and HbR both with same significant trend or both with
        no significant trend).

    baseline: float
        Duration of the baseline prior to trigger onset (in sec).

    dummies : bool
        Whether the data should include dummy channels (when channels have been
        removed from the preprocessed data).     
    """
    # Add dummy channels
    if dummies is True:
        dummy_channels = [0, 3, 20, 23]
        dummy_grand_avg = grand_avg.copy()
        for dummy in dummy_channels:
            dummy_grand_avg = np.insert(dummy_grand_avg, dummy, 0, axis=2)
        grand_avg = dummy_grand_avg

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.tight_layout(h_pad=4, w_pad=4)
    axes[0, 0].set(title='HbO Right', xlabel='Time (sec)', ylabel='Hb concentration (uM)')
    axes[0, 1].set(title='HbO Left', xlabel='Time (sec)')
    axes[1, 0].set(title='HbR Right', xlabel='Time (sec)', ylabel='Hb concentration (uM)')
    axes[1, 1].set(title='HbR Left', xlabel='Time (sec)')

    for channel in range(grand_avg.shape[2]):
        if (dummies is True) and (channel in dummy_channels):
            continue
        if channel < grand_avg.shape[2]/2:
            col = 1
        else:
            col = 0
        for chromophore in range(grand_avg.shape[3]):
            channel_average = np.nanmean(grand_avg[:, condition, channel, chromophore, :], axis=0)
            if activations[channel] > 0:
                axes[chromophore, col].plot(np.linspace(-baseline, 20.0, num=len(channel_average)), channel_average, label=f'Ch {str(channel+1)}', linestyle='solid')
            else:
                axes[chromophore, col].plot(np.linspace(-baseline, 20.0, num=len(channel_average)), channel_average, label=f'Ch {str(channel+1)}', linestyle='dashed')
            axes[chromophore, col].grid(which='major', color='#666666', linestyle='-')
            axes[chromophore, col].minorticks_on()
            axes[chromophore, col].grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
            axes[chromophore, col].margins(x=0)
            axes[chromophore, col].set_ylim(-0.64, 1.2)
            axes[1, col].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    print("Solid line = significant activation")
    print("Dashed line = no significant activation")
    plt.show()


def analyse_contrast(feature_grand_avg, condition_a, condition_b, fdr=True,
                     dummies=True):
    """
    Get activations for condition contrast.

    Parameters
    ----------
    feature_grand_avg : numpy array
        Array of features for all subjects of shape (n_subjects, n_conditions,
        n_channels, n_chromophores).

    condition_a : int
        Condition index to compared to, from the list provided when loading the
        results.

    condition_b : int
        Condition index to subtract, from the list provided when loading the
        results.

    fdr : bool
        Whether to apply false discovery rate correction on the channel level.

    dummies : bool
        Whether the data should include dummy channels (when channels have been
        removed from the preprocessed data).

    Returns
    -------
    p_values : numpy array
        P-values for channel activation of shape (n_channels, n_chromophores).

    t_values : numpy array
        T-values for channel activation of shape (n_channels, n_chromophores).

    trends : numpy array
        Array of trends for channels on the contrast of shape (n_channels,
        n_chromophores). 1 for significant increase, -1 for significant
        decrease, 0 for neither.

    activations : numpy array
        List of activations for channels on the contrast of shape
        (n_channels,). Positive for significant activation, negative for
        significant deactivation. 2 for significant increase in HbO and
        decrease in HbR, 1 for significant increase in HbO or decrease in HbR,
        -1 for significant decrease in HbO or increase in HbR, -2 for
        significant decrease in HbO and increase in HbR, 0 for no significant
        activation (HbO and HbR both with same significant trend or both with
        no significant trend).
    """
    p_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromophores)
    t_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromophores)
    trends = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromophores)

    for channel in range(feature_grand_avg.shape[2]):
        for chromophore in range(feature_grand_avg.shape[3]):
            contrast = feature_grand_avg[:, condition_a, channel, chromophore] - feature_grand_avg[:, condition_b, channel, chromophore]
            # Get only good channels
            samples = contrast[~np.isnan(contrast)]
            # Warning if parametric test assumptions not verified
            if len(samples) < 30:
                print(f"Warning, only {len(samples)} sample(s) for channel No {channel+1}")

            # t-test
            if samples.mean() > 0:
                trends[channel, chromophore] = 1
                s_tt, p_tt = stats.ttest_1samp(samples, 0, alternative='greater')
            elif samples.mean() < 0:
                trends[channel, chromophore] = -1
                s_tt, p_tt = stats.ttest_1samp(samples, 0, alternative='less')
            else:
                trends[channel, chromophore] = 0
                s_tt, p_tt = stats.ttest_1samp(samples, 0)
            p_values[channel, chromophore] = p_tt
            t_values[channel, chromophore] = s_tt

    if fdr:
        for chromophore in range(feature_grand_avg.shape[3]):
            # FDR correction at channel level
            _, p_values[:, chromophore] = multitest.fdrcorrection(p_values[:, chromophore], alpha=CONFIDENCE)

    for channel in range(trends.shape[0]):
        for chromophore in range(trends.shape[1]):
            if p_values[channel, chromophore] >= CONFIDENCE:
                trends[channel, chromophore] = 0

    activations = trends[:, 0] - trends[:, 1]  # (channels)

    # Add dummy channels
    if dummies is True:
        dummy_channels = [0, 3, 20, 23]
        dummy_activations = activations.copy()
        for dummy in dummy_channels:
            dummy_activations = np.insert(dummy_activations, dummy, 0, axis=0)
        activations = dummy_activations

    return p_values, t_values, trends, activations


def get_no_act(activations):
    """
    Get non-activated channels.

    Parameters
    ----------
    activations : numpy array
        List of activations for channels on the contrast of shape
        (n_channels,). Positive for significant activation, negative for
        significant deactivation. 2 for significant increase in HbO and
        decrease in HbR, 1 for significant increase in HbO or decrease in HbR,
        -1 for significant decrease in HbO or increase in HbR, -2 for
        significant decrease in HbO and increase in HbR, 0 for no significant
        activation (HbO and HbR both with same significant trend or both with
        no significant trend).

    Returns
    -------
    no_act_chs : list of int
        List of non-activated channels.
    """
    full_ch = np.linspace(1, len(activations), num=len(activations), dtype=int)
    act_ch = np.where(activations > 0)[0] + 1
    no_act_chs = [x for x in full_ch if x not in act_ch]

    return no_act_chs


def topo_overlay(values, save_path=None):
    """
    Overlay channel results on topo maps.

    Parameters
    ----------
    values: numpy array
        List of values to overlay on the topo, one for each channel.

    save_path : str
        Path of the file to save the topo maps.
    """
    hex_color = '#8b1f2b'  # Mustard: eb9235, Burgundy: 8b1f2b
    rgb_color = mcolors.hex2color(hex_color)
    colors = [(1, 1, 1), rgb_color]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    # cmap = plt.get_cmap('YlOrRd')

    ch_pos = np.array([
        [None, None], [376, 322], [444, 260], [None, None], [412, 292],
        [190, 374], [145, 347], [445, 323], [245, 292], [213, 334],
        [172, 304], [334, 353], [289, 320], [257, 362], [392, 269],
        [361, 308], [320, 279], [431, 296], [407, 339], [217, 322],
        [None, None], [150, 257], [182, 290], [None, None], [410, 373],
        [455, 347], [151, 323], [354, 292], [386, 333], [428, 303],
        [266, 353], [311, 322], [342, 363], [210, 267], [239, 308],
        [280, 279], [169, 298], [192, 339]
        ])

    i_front = np.array([2, 3, 5, 8, 20, 22, 23, 27]) - 1
    i_left = np.array([6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]) - 1
    i_right = np.array([25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]) - 1

    colors = [value*0.5 for value in values]
    c_list = np.array([cmap(color) for color in colors])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left
    img_path = '../../assets/left.png'
    im = plt.imread(img_path)
    axes[2].imshow(im, extent=[0, 600, 0, 600])
    axes[2].scatter(ch_pos[i_left, 0], ch_pos[i_left, 1], c=c_list[i_left],
                    s=200, linewidths=2, edgecolors='k')
    axes[2].axis('off')
    for i in i_left:
        axes[2].annotate(i+1, ch_pos[i], ha='center', va='center', color='w', fontsize=8)

    # Front
    img_path = '../../assets/front.png'
    im = plt.imread(img_path)
    axes[1].imshow(im, extent=[0, 600, 0, 600])
    axes[1].scatter(ch_pos[i_front, 0], ch_pos[i_front, 1], c=c_list[i_front],
                    s=200, linewidths=2, edgecolors='k')
    axes[1].axis('off')
    for i in i_front:
        axes[1].annotate(i+1, ch_pos[i], ha='center', va='center', color='w', fontsize=8)

    # Right
    img_path = '../../assets/right.png'
    im = plt.imread(img_path)
    axes[0].imshow(im, extent=[0, 600, 0, 600])
    axes[0].scatter(ch_pos[i_right, 0], ch_pos[i_right, 1], c=c_list[i_right],
                    s=200, linewidths=2, edgecolors='k')
    axes[0].axis('off')
    for i in i_right:
        axes[0].annotate(i+1, ch_pos[i], ha='center', va='center', color='w', fontsize=8)

    # fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes.ravel().tolist())

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def topo_overlay_roi(values, save_path=None):
    """
    Overlay channel results on topo maps.

    Parameters
    ----------
    values: numpy array
        List of values to overlay on the topo, one for each channel.

    save_path : str
        Path of the file to save the topo maps.
    """
    hex_color = '#8b1f2b'
    rgb_color = mcolors.hex2color(hex_color)
    colors = [(1, 1, 1), rgb_color]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors,
                                                     N=256)
    # cmap = plt.get_cmap('YlOrRd')

    ch_pos = np.array([
        [428, 294], [231, 336], [378, 308],
        [168, 294], [368, 336], [222, 308],
        ])

    i_front = np.array([0, 3])
    i_left = np.array([1, 2])
    i_right = np.array([4, 5])

    colors = [value*0.5 for value in values]
    c_list = np.array([cmap(color) for color in colors])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left
    img_path = '../../assets/left.png'
    im = plt.imread(img_path)
    axes[2].imshow(im, extent=[0, 600, 0, 600])
    axes[2].scatter(ch_pos[i_left, 0], ch_pos[i_left, 1], c=c_list[i_left],
                    s=600, linewidths=2, edgecolors='k', marker=',')
    axes[2].axis('off')

    # Front
    img_path = '../../assets/front.png'
    im = plt.imread(img_path)
    axes[1].imshow(im, extent=[0, 600, 0, 600])
    axes[1].scatter(ch_pos[i_front, 0], ch_pos[i_front, 1], c=c_list[i_front],
                    s=600, linewidths=2, edgecolors='k', marker=',')
    axes[1].axis('off')

    # Right
    img_path = '../../assets/right.png'
    im = plt.imread(img_path)
    axes[0].imshow(im, extent=[0, 600, 0, 600])
    axes[0].scatter(ch_pos[i_right, 0], ch_pos[i_right, 1], c=c_list[i_right],
                    s=600, linewidths=2, edgecolors='k', marker=',')
    axes[0].axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_info_peaks(grand_avg, type="hbo", baseline=2):
    """
    Get peaks info (time to peak and magnitude).

    Parameters
    ----------
    grand_avg : numpy array
        Array of block grand average for all subjects of shape (n_subjects,
        n_conditions, n_channels, n_chromophores, n_timepoints).

    type : string
        Whether to get time to peak on HbO (`"hbo"`), HbR (`"hbr"`) or HbDiff
        (`"hbdiff"`).

    baseline: float
        Duration of the baseline prior to trigger onset (in sec).

    Returns
    -------
    time_to_peaks : numpy array
        Array of time to peaks for all subjects of shape (n_subjects,
        n_conditions).

    magnitudes : numpy array
        Array of magnitudes for all subjects of shape (n_subjects,
        n_conditions).
    """
    start = int(baseline*10)
    grand_avg_mean = np.nanmean(grand_avg, axis=2)
    if type == "hbo":
        time_to_peaks = np.argmax(grand_avg_mean[:, :, 0, start:], axis=-1)
        magnitudes = np.max(grand_avg_mean[:, :, 0, start:], axis=-1)
    elif type == "hbr":
        time_to_peaks = np.argmin(grand_avg_mean[:, :, 1, start:], axis=-1)
        magnitudes = np.min(grand_avg_mean[:, :, 1, start:], axis=-1)
    elif type == "hbdiff":
        hbdiff = grand_avg_mean[:, :, 0, start:] - grand_avg_mean[:, :, 1, start:]
        time_to_peaks = np.argmax(hbdiff, axis=-1)
        magnitudes = np.max(hbdiff, axis=-1)

    return time_to_peaks, magnitudes


def stats_ttp(time_to_peaks, condition_a, condition_b):
    """
    Analyse time to peak differences for condition contrast.

    Parameters
    ----------
    time_to_peaks : numpy array
        Array of time to peaks for all subjects of shape (n_subjects,
        n_conditions).

    condition_a : int
        Condition index to compared to, from the list provided when loading the
        results.

    condition_b : int
        Condition index to subtract, from the list provided when loading the
        results.

    Returns
    -------
    p_value : float
        P-value for the time to peak contrast.

    t_value : float
        T-value for the time to peak contrast.

    trend : int
        Trend for the time to peak contrast of shape. 1 for significant
        positive difference, -1 for significant negative difference, 0 for
        neither.
    """
    # _, p_shap = stats.shapiro(time_to_peaks[:, condition_a])
    # if p_shap < 0.05:
    #     print("Not normal")
    # _, p_shap = stats.shapiro(time_to_peaks[:, condition_b])
    # if p_shap < 0.05:
    #     print("Not normal")

    contrast = time_to_peaks[:, condition_a] - time_to_peaks[:, condition_b]

    # Warning if parametric test assumptions not verified
    if len(contrast) < 30:
        print(f"Warning, only {len(contrast)} sample(s) for the contrast")

    # t-test
    if contrast.mean() > 0:
        trend = 1
        s_tt, p_tt = stats.ttest_1samp(contrast, 0, alternative='greater')
    elif contrast.mean() < 0:
        trend = -1
        s_tt, p_tt = stats.ttest_1samp(contrast, 0, alternative='less')
    else:
        trend = 0
        s_tt, p_tt = stats.ttest_1samp(contrast, 0)

    if p_tt >= CONFIDENCE:
        trend = 0

    return p_tt, s_tt, trend


def paired_ttest(a, b):
    results = scipy.stats.ttest_rel(a, b, nan_policy='omit')
    t_values = results[0]
    return t_values
