"""
Functions for social task analysis
Author: Johann Benerradi
------------------------
"""

import glob
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns

from scipy import stats
from statsmodels.stats import multitest


CONFIDENCE = 0.05


# -----------------------------------------------------------------------------
# Loading functions
# -----------------------------------------------------------------------------
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

    all_n_chs : list of int
        List of number of channels for each participant.

    all_n_trials : list of int
        List of number of trials for each participant.
    """
    result_files = glob.glob(path + '/*.mat')

    # Block average: (subjects, conditions, channels, chromophores, timepoints)
    grand_avg = np.empty((0, len(cond_list), 34, 2, 221))
    subj_ids = []
    rejected = []

    c_channels = 0
    c_ltfile = 0
    c_trials = 0

    all_n_chs = []
    all_n_trials = []

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

        # Exclude if no looking time autocoder data:
        reason = None
        if not np.squeeze(mat_file['results']['LTFile'][0, 0]):
            exclude = True
            reason = 'trials'
            c_ltfile += 1

        # Exclude if less than 3 trials for any condition ('S', 'V' or 'N'):
        if np.any(
            np.squeeze(mat_file['results']['nTrials'][0, 0])[cond_indices] < 3
        ):
            exclude = True
            reason = 'trials'
            c_trials += 1
        else:
            trials = np.squeeze(
                mat_file['results']['nTrials'][0, 0]
            )[cond_indices]
            n_trials = np.mean(trials)

        # Extract data:
        avg = np.squeeze(
            mat_file['results']['dcAvg'][0, 0]
        )[:, :2, :, cond_indices]
        avg = np.transpose(avg)[np.newaxis, :, :, :, :]

        # Exclude if more than 40% of excluded channels:
        if np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum() > 0.4*avg.shape[2]:
            exclude = True
            if not reason:
                reason = 'channels'
            c_channels += 1
        else:
            n_chs = 34 - np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum()

        # Exclude subject:
        if exclude:
            rejected.append((subj_id, reason))
            continue

        # Append all subjects:
        grand_avg = np.append(grand_avg, avg, axis=0)
        subj_ids.append(subj_id)
        all_n_chs.append(n_chs)
        all_n_trials.append(n_trials)

    print(f"N={grand_avg.shape[0]}")

    grand_avg *= 1e6  # µM

    return grand_avg, subj_ids, rejected, all_n_chs, all_n_trials


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

    all_n_chs : list of int
        List of number of channels for each participant.

    all_n_trials : list of int
        List of number of trials for each participant.
    """
    # Block average: (subjects, conditions, channels, chromophores, timepoints)
    grand_avg = np.empty((0, len(cond_list), 34, 2, 221))
    subj_ids = []

    # Load group results
    mat_file_vn = scipy.io.loadmat(f"{path}/V%3EN/groupResults.mat")
    mat_file_sc = scipy.io.loadmat(f"{path}/S%3EC/groupResults.mat")

    # Keep subjects with at least 3 trials in any condition
    names_vn = [np.squeeze(name).tolist() for name in np.squeeze(
        mat_file_vn['group']['subjs'][0, 0]['name'][0, :]
    )]
    names_sc = [np.squeeze(name).tolist() for name in np.squeeze(
        mat_file_sc['group']['subjs'][0, 0]['name'][0, :]
    )]
    subj_list = [(i, name) for i, name in enumerate(names_vn)
                 if name in names_sc]
    rejected = [(name.split('_')[0], 'trials') for name in names_vn
                if name not in names_sc]
    rejected += [(name.split('_')[0], 'trials') for name in names_sc
                 if name not in names_vn]

    # Prepare condition reordering
    cn = np.squeeze(
        mat_file_vn['group']['conditions'][0, 0]['CondNamesAct'][0, 0]
    )
    cond_names = [np.squeeze(c).tolist() for c in cn]
    cond_indices = []
    for cond in cond_list:
        cond_indices.append(cond_names.index(cond))

    # Prepare channel reordering
    ch_indices = np.array([
        34, 23, 35, 14, 24, 36, 11, 15, 25, 3, 12, 16, 1, 4, 13, 2, 5, 29, 26,
        30, 20, 27, 31, 17, 21, 28, 8, 18, 22, 6, 9, 19, 7, 10
    ]) - 1

    all_n_chs = []
    all_n_trials = []

    for i, name in subj_list:
        subj_id = name.split('_')[0]

        # Extract data
        avg = np.squeeze(
            mat_file_vn['group']['subjs'][0, 0]['procResult'][0, i][
                'dcAvg'
            ][0, 0]
        )[:, :2, :, cond_indices]
        avg = np.transpose(avg[:, :, ch_indices, :])[np.newaxis, :, :, :, :]

        # Extract number of trials per condition
        trials = mat_file_vn['group']['subjs'][0, 0]['procResult'][0, i][
            'nTrials'
        ][0, 0]
        trials = trials[~np.all(trials == 0, axis=1)]
        if np.all(trials == trials[0], axis=1).all():
            trials = trials[0, cond_indices]  # select all except baseline
            n_trials = np.mean(trials)
        else:
            raise Exception('number of trials not matching for all channels')

        # Exclude if more than 40% of excluded channels:
        if np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum() > 0.4*avg.shape[2]:
            rejected.append((subj_id, 'channels'))
            continue
        else:
            n_chs = 34 - np.isnan(avg.mean(axis=(0, 1, 3, 4))).sum()

        # Append all subjects:
        grand_avg = np.append(grand_avg, avg, axis=0)
        subj_ids.append(subj_id)
        all_n_chs.append(n_chs)
        all_n_trials.append(n_trials)

    print(f"N={grand_avg.shape[0]}")

    grand_avg *= 1e6  # µM

    return grand_avg, subj_ids, rejected, all_n_chs, all_n_trials


# -----------------------------------------------------------------------------
# HRF functions
# -----------------------------------------------------------------------------
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
    feature_grand_avg = grand_avg[
        :, :, :, :, int(window[0]*10+20):int(window[1]*10+20)
    ].mean(axis=-1)

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
    p_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromas)
    t_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromas)
    trends = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromas)

    for channel in range(feature_grand_avg.shape[2]):
        for chromophore in range(feature_grand_avg.shape[3]):
            samples = feature_grand_avg[:, condition, channel, chromophore]
            # Get only good channels
            samples = samples[~np.isnan(samples)]
            # Warning if parametric test assumptions not verified
            if len(samples) < 30:
                print(f"Warning, only {len(samples)} sample(s) "
                      f"for channel No {channel+1}")

            # t-test
            if samples.mean() > 0:
                trends[channel, chromophore] = 1
                s_tt, p_tt = stats.ttest_1samp(samples, 0,
                                               alternative='greater')
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
            _, p_values[:, chromophore] = multitest.fdrcorrection(
                p_values[:, chromophore], alpha=CONFIDENCE
            )

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
    axes[0, 0].set(title='HbO Right', xlabel='Time (sec)',
                   ylabel='Hb concentration (uM)')
    axes[0, 1].set(title='HbO Left', xlabel='Time (sec)')
    axes[1, 0].set(title='HbR Right', xlabel='Time (sec)',
                   ylabel='Hb concentration (uM)')
    axes[1, 1].set(title='HbR Left', xlabel='Time (sec)')

    for channel in range(grand_avg.shape[2]):
        if (dummies is True) and (channel in dummy_channels):
            continue
        if channel < grand_avg.shape[2]/2:
            col = 1
        else:
            col = 0
        for chromophore in range(grand_avg.shape[3]):
            channel_average = np.nanmean(
                grand_avg[:, condition, channel, chromophore, :], axis=0
            )
            if activations[channel] > 0:
                axes[chromophore, col].plot(
                    np.linspace(-baseline, 20.0, num=len(channel_average)),
                    channel_average, label=f'Ch {str(channel+1)}',
                    linestyle='solid'
                )
            else:
                axes[chromophore, col].plot(
                    np.linspace(-baseline, 20.0, num=len(channel_average)),
                    channel_average, label=f'Ch {str(channel+1)}',
                    linestyle='dashed'
                )
            axes[chromophore, col].grid(which='major', color='#666666',
                                        linestyle='-')
            axes[chromophore, col].minorticks_on()
            axes[chromophore, col].grid(which='minor', color='#999999',
                                        linestyle='-', alpha=0.2)
            axes[chromophore, col].margins(x=0)
            axes[chromophore, col].set_ylim(-0.64, 1.2)
            axes[1, col].legend(loc='upper center',
                                bbox_to_anchor=(0.5, -0.15), ncol=5)
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
    p_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromas)
    t_values = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromas)
    trends = np.empty((feature_grand_avg.shape[2], 2))  # (channels, chromas)

    for channel in range(feature_grand_avg.shape[2]):
        for chromophore in range(feature_grand_avg.shape[3]):
            contrast = (
                feature_grand_avg[:, condition_a, channel, chromophore]
                - feature_grand_avg[:, condition_b, channel, chromophore]
            )
            # Get only good channels
            samples = contrast[~np.isnan(contrast)]
            # Warning if parametric test assumptions not verified
            if len(samples) < 30:
                print(f"Warning, only {len(samples)} sample(s) "
                      f"for channel No {channel+1}")

            # t-test
            if samples.mean() > 0:
                trends[channel, chromophore] = 1
                s_tt, p_tt = stats.ttest_1samp(samples, 0,
                                               alternative='greater')
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
            _, p_values[:, chromophore] = multitest.fdrcorrection(
                p_values[:, chromophore], alpha=CONFIDENCE
            )

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


def topo_overlay(values, color="#8b1f2b"):
    """
    Overlay channel results on topo maps.

    Parameters
    ----------
    values: numpy array
        List of values to overlay on the topo, one for each channel.

    color : str
        Color to use for significant channels.
    """
    hex_color = color
    rgb_color = mcolors.hex2color(hex_color)
    colors = [(1, 1, 1), rgb_color]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors,
                                                     N=256)

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
    i_right = np.array(
        [25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    ) - 1

    colors = [value*0.5 for value in values]
    c_list = np.array([cmap(color) for color in colors])

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left
    img_path = '../../assets/left.png'
    im = plt.imread(img_path)
    axes[2].imshow(im, extent=[0, 600, 0, 600])
    axes[2].scatter(ch_pos[i_left, 0], ch_pos[i_left, 1], c=c_list[i_left],
                    s=200, linewidths=2, edgecolors='k')
    axes[2].axis('off')
    for i in i_left:
        axes[2].annotate(i+1, ch_pos[i], ha='center', va='center', color='w',
                         fontsize=8)

    # Front
    img_path = '../../assets/front.png'
    im = plt.imread(img_path)
    axes[1].imshow(im, extent=[0, 600, 0, 600])
    axes[1].scatter(ch_pos[i_front, 0], ch_pos[i_front, 1], c=c_list[i_front],
                    s=200, linewidths=2, edgecolors='k')
    axes[1].axis('off')
    for i in i_front:
        axes[1].annotate(i+1, ch_pos[i], ha='center', va='center', color='w',
                         fontsize=8)

    # Right
    img_path = '../../assets/right.png'
    im = plt.imread(img_path)
    axes[0].imshow(im, extent=[0, 600, 0, 600])
    axes[0].scatter(ch_pos[i_right, 0], ch_pos[i_right, 1], c=c_list[i_right],
                    s=200, linewidths=2, edgecolors='k')
    axes[0].axis('off')
    for i in i_right:
        axes[0].annotate(i+1, ch_pos[i], ha='center', va='center', color='w',
                         fontsize=8)

    plt.show()


def topo_overlay_roi(values):
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

    ch_pos = np.array([
        [428, 294], [231, 336], [378, 308],
        [168, 294], [368, 336], [222, 308],
        ])

    i_front = np.array([0, 3])
    i_left = np.array([1, 2])
    i_right = np.array([4, 5])

    colors = [value*0.5 for value in values]
    c_list = np.array([cmap(color) for color in colors])

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

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
        hbdiff = (
            grand_avg_mean[:, :, 0, start:]
            - grand_avg_mean[:, :, 1, start:]
        )
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
    contrast = time_to_peaks[:, condition_a] - time_to_peaks[:, condition_b]

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


# -----------------------------------------------------------------------------
# Selectivity functions
# -----------------------------------------------------------------------------
def selective_paired(df, roi_list, ages, subj_list, type, ylim,
                     feature="Window average"):
    """
    Plot paired selectivity data for different ROIs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.

    roi_list : list of str
        List of ROI names to include in the plot.

    ages : list of str
        List of age groups to include in the plot.

    subj_list : list of str
        List of subject IDs to include in the plot.

    type : str
        Type of channel ('hbo' or 'hbr').

    ylim : tuple of float
        Y-axis limits for the plot.

    feature : str
        Name of the feature column to plot. Defaults to ``"Window average"``.
    """
    sub_df = df[df['Condition'].isin(['N', 'V'])]
    sub_df = sub_df[sub_df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    for i_roi, roi in enumerate(roi_list):
        df_roi = sub_df[sub_df['ROI'] == roi]
        axes.flat[i_roi].title.set_text(roi.capitalize())

        # Plot points
        sns.stripplot(data=df_roi, x="Age (months)", y=feature,
                      hue="Condition", dodge=True, alpha=.45, legend=True,
                      order=['5', '8', '12', '18', '24', '60'],
                      hue_order=['N', 'V'], ax=axes.flat[i_roi],
                      palette=['green', 'purple'])

        # Plot links
        for i_age, age in enumerate(ages):
            df_v = df_roi[(df_roi['Age (months)'] == age[:-2])
                          & (df_roi['Condition'] == 'V')]
            df_n = df_roi[(df_roi['Age (months)'] == age[:-2])
                          & (df_roi['Condition'] == 'N')]
            df_v, df_n = df_v.dropna(), df_n.dropna()

            locs1 = axes.flat[i_roi].get_children()[i_age*2].get_offsets()
            locs2 = axes.flat[i_roi].get_children()[i_age*2+1].get_offsets()

            if not list(locs1[:, 1]) == list(df_n[feature]):
                raise Exception('DataFrame and figure not matching')
            if not list(locs2[:, 1]) == list(df_v[feature]):
                raise Exception('DataFrame and figure not matching')

            for i in range(locs1.shape[0]):
                x, y = [locs1[i, 0], locs2[i, 0]], [locs1[i, 1], locs2[i, 1]]
                axes.flat[i_roi].plot(x, y, color="black", alpha=0.25)

        # Plot average marks
        sns.pointplot(data=df_roi, x="Age (months)", y=feature,
                      hue="Condition", dodge=.5 - .5 / 3,
                      errorbar=None, markers="_", markersize=10,
                      linestyle="none",
                      order=['5', '8', '12', '18', '24', '60'],
                      hue_order=['N', 'V'], ax=axes.flat[i_roi], zorder=1000,
                      palette=['black', 'black'], legend=False,)
        if i_roi < 2:
            legend_handles = [
                mlines.Line2D(
                    [], [], color=color, marker='o', linestyle='None',
                    markersize=5, label=label
                )
                for color, label in zip(
                    ['green', 'purple'],
                    ['Auditory non-social', 'Auditory social']
                )
            ]
            axes.flat[i_roi].legend(handles=legend_handles, title='Condition',
                                    loc='upper left')
        else:
            axes.flat[i_roi].legend().set_visible(False)
        axes.flat[i_roi].set_ylabel("HbO response (µM)")
        axes.flat[i_roi].set_ylim(*ylim)
        axes.flat[i_roi].grid()
        axes.flat[i_roi].set_axisbelow(True)
        axes.flat[i_roi].grid(which='major', visible=True, color='silver',
                              linewidth=1.)
    plt.tight_layout(pad=3)
    plt.show()


def selective_trajectories(df, subj_list, type, ylim,
                           feature="Window average"):
    """
    Plot selectivity trajectories for different age groups.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.

    subj_list : list of str
        List of subject IDs to include in the plot.

    type : str
        Type of channel ('hbo' or 'hbr').

    ylim : tuple of float
        Y-axis limits for the plot.

    feature : str, optional
        Name of the feature column to plot. Defaults to ``"Window average"``.
    """
    # Get selectivity groups using HbO anterior temporal
    sub_df = df[df['Channel type'] == 'hbo']
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    spec_map = {'5 mo': [], '8 mo': [], '12 mo': [], 'later': []}
    sub_v = sub_df[sub_df['Condition'] == 'V']
    sub_vn = sub_df[sub_df['Condition'] == 'V-N']
    for subj in subj_list:
        d = sub_vn.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_v = sub_v.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        if (
            d.query("`Age (months)` == '5'")[feature].mean() > 0
            and d_v.query("`Age (months)` == '5'")[feature].mean() > 0
        ):
            spec_map['5 mo'].append(subj)
        elif (
            d.query("`Age (months)` == '8'")[feature].mean() > 0
            and d_v.query("`Age (months)` == '8'")[feature].mean() > 0
        ):
            spec_map['8 mo'].append(subj)
        elif (
            d.query("`Age (months)` == '12'")[feature].mean() > 0
            and d_v.query("`Age (months)` == '12'")[feature].mean() > 0
        ):
            spec_map['12 mo'].append(subj)
        else:
            spec_map['later'].append(subj)

    # Plot data (HbO or HbR)
    sub_df = df[df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    colors = ['#674ea7ff', '#cc4125ff', "#ce9a00ff", '#34a853ff',
              '#ff6d01ff', '#6d9eebff', "#6d9eebff"]

    for i, age_spec in enumerate(spec_map.keys()):
        df_roi = sub_df[sub_df['Condition'].isin(['V', 'N'])]
        df_roi = df_roi.query("ROI.str.contains('anterior')")
        df_roi['Age (months)'] = df_roi['Age (months)'].astype(float)
        df_roi = df_roi[df_roi['Age (months)'] <= 60]
        df_roi = df_roi[df_roi["ID"].isin(spec_map[age_spec])]
        df_roi_kids = df_roi[df_roi['Age (months)'] == 60].copy()
        df_roi_kids['Age (months)'] = df_roi_kids['Age (months)'] - 20
        df_roi = df_roi[df_roi['Age (months)'] != 60].copy()

        if age_spec == 'later':
            axes.flat[i].title.set_text(
                f"No auditory social selectivity by 12 mo "
                f"(N={len(spec_map[age_spec])})"
            )
        else:
            axes.flat[i].title.set_text(
                f"First auditory social selectivity at "
                f"{age_spec} (N={len(spec_map[age_spec])})"
            )

        # Plot points
        sns.lineplot(data=df_roi, x="Age (months)", y=feature,
                     hue='Condition', style='Condition',
                     hue_order=['V', 'N'], style_order=['V', 'N'],
                     palette=[colors[i], colors[i]], markers=["o", "^"],
                     dashes=[(2, 0), (2, 2)], ax=axes.flat[i], errorbar='se',
                     legend=True)
        sns.pointplot(data=df_roi_kids, x="Age (months)", y=feature,
                      hue='Condition', hue_order=['V', 'N'], dodge=0.5,
                      palette=[colors[i], colors[i]], markers=["o", "^"],
                      ax=axes.flat[i], errorbar='se', scale=0.8,
                      native_scale=True, err_kws=dict(alpha=0.5), legend=False)
        axes.flat[i].axhline(y=0, color='black', linewidth=2)
        axes.flat[i].grid()
        axes.flat[i].set_xlim(0, 45)
        axes.flat[i].set_ylim(*ylim)
        axes.flat[i].set_xlabel("Age at the session (months)")
        axes.flat[i].set_ylabel("Anterior temporal HbO response (µM)")
        axes.flat[i].set_xticks(np.arange(0, 45, 10))
        labels = [item.get_text() for item in axes.flat[i].get_xticklabels()]
        labels[-1] = '3-5 y'
        axes.flat[i].set_xticklabels(labels)
        axes.flat[i].axhspan(ylim[0], 0, facecolor='grey', alpha=0.25)
        axes.flat[i].legend(loc='lower right')
        handles, labels = axes.flat[i].get_legend_handles_labels()
        proxy_white = plt.Rectangle((0, 0), 1, 1, fc='white', alpha=0.25,
                                    ec='black', lw=1,
                                    label='Social selectivity')
        proxy_grey = plt.Rectangle((0, 0), 1, 1, fc='grey', alpha=0.25,
                                   ec='black', lw=1,
                                   label='No social selectivity')
        all_handles = handles + [proxy_white, proxy_grey]
        all_labels = labels + ['Activation', 'No activation']
        new_labels = []
        for label in all_labels:
            if label == 'V':
                new_labels.append('Vocal')
            elif label == 'N':
                new_labels.append('Non-vocal')
            else:
                new_labels.append(label)
        axes.flat[i].legend(handles=all_handles, labels=new_labels,
                            loc='lower right')
        axes.flat[i].set_axisbelow(True)
    plt.tight_layout(pad=3)
    plt.show()


def selective_table(df, subj_list, cond, feature="Window average"):
    """
    Generate a table of selectivity data for different age groups.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to process.

    subj_list : list of str
        List of subject IDs to include in the table.

    cond : str
        Condition to filter the data by.

    feature : str
        Name of the feature column to plot. Defaults to ``"Window average"``.

    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame containing the results.
    """
    sub_df = df[df['Channel type'] == 'hbo']
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    sub_v = sub_df[sub_df['Condition'] == 'V']
    sub_n = sub_df[sub_df['Condition'] == 'N']
    sub_df = sub_df[sub_df['Condition'] == cond]
    rows = []
    for subj in subj_list:
        row = {'ID': subj}
        d = sub_df.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_v = sub_v.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_n = sub_n.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        for age in [5, 8, 12, 18, 24, 60]:
            if (
                d.query(f"`Age (months)` == '{age}'")[feature].mean() > 0
                and d_v.query(f"`Age (months)` == '{age}'")[feature].mean() > 0
            ):
                row[f"{age}mo"] = "V"
            elif (
                d.query(f"`Age (months)` == '{age}'")[feature].mean() < 0
                and d_n.query(f"`Age (months)` == '{age}'")[feature].mean() > 0
            ):
                row[f"{age}mo"] = "N"
            elif (
                np.isnan(d.query(f"`Age (months)` == '{age}'")[feature].mean())
            ):
                row[f"{age}mo"] = "Missing"
            else:
                row[f"{age}mo"] = "None"
        rows.append(row)

    df_out = pd.DataFrame(rows)
    return df_out
