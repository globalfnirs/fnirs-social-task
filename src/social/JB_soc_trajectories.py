"""
Script for social task trajectories analysis
Author: Johann Benerradi
------------------------
"""

# %% Imports
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import functions.fig as fig
import functions.soc as soc

import statsmodels.formula.api as smf
from statsmodels.stats import multitest
from scipy.stats import pearsonr


AGES = ['5mo', '8mo', '12mo', '18mo', '24mo', '60mo']
CONDS = ['V', 'N']


# Get ROI dict
with open('../../assets/rois_new.json', 'r') as f:
    roi_json = json.load(f)

# Start channel indices at 0 instead of 1
roi_dict = {}
for age, rois in roi_json.items():
    d = {}
    for roi, channels in rois.items():
        new_channels = list(np.array(channels)-1)
        d[roi] = new_channels
    roi_dict[age] = d

# Define relevant ROIs
roi_list = ['left frontal', 'right frontal',
            'left anterior-temporal', 'right anterior-temporal',
            'left posterior-temporal', 'right posterior-temporal']

# Get all subject IDs
with open('../../assets/ids.json', 'r') as f:
    ids_json = json.load(f)
all_subj_ids = list(ids_json.keys())


# %% Load data
# Load NIRS data
grand_avg_ct_dict = {
    id: np.empty((len(AGES), len(roi_list), 2, 221))*np.nan for id in all_subj_ids}
df_rows = []
for i_age, age in enumerate(AGES):
    print(f'##### {age} #####')

    # Load epochs
    path = f'[path to dataset]/results/{age}/'
    if age == '1mo':
        grand_avg, subj_ids, rejected = soc.load_results_1mo(path, CONDS)
    elif age == '60mo':
        grand_avg, subj_ids, rejected, _ = soc.load_results_60mo(path, CONDS)
    else:
        grand_avg, subj_ids, rejected, _ = soc.load_results_infancy(path, CONDS)

    # Print data rejection info
    n_trials = len([r for r in rejected if r[1] == 'trials'])
    n_chs = len([r for r in rejected if r[1] == 'channels'])
    print(f'{n_trials+n_chs} rejected: trials={n_trials}, chs={n_chs}')

    # Combine channels by ROIs (subjects, conds, rois, types, times)
    if age != '1mo':
        dummy_channels = [0, 3, 20, 23]
        for dummy in dummy_channels:
            grand_avg = np.insert(grand_avg, dummy, np.nan, axis=2)
    roi_grand_avg = np.empty((grand_avg.shape[0], grand_avg.shape[1], 0,
                              grand_avg.shape[3], grand_avg.shape[4]))
    for roi in roi_list:
        roi_avg = np.nanmean(grand_avg[:, :, roi_dict[age][roi], :, :],
                             axis=2, keepdims=True)
        roi_grand_avg = np.append(roi_grand_avg, roi_avg, axis=2)
    grand_avg = roi_grand_avg

    # Extract features
    if age == '1mo':
        grand_avg = grand_avg[:, :, :, :, 20:]
        bl = grand_avg[:, :, :, :, :20].mean(axis=-1, keepdims=True)
        grand_avg -= bl
    ttps_hbo = np.argmax(grand_avg[:, :, :, 0, 20:], axis=-1)
    ttps_hbr = np.argmin(grand_avg[:, :, :, 1, 20:], axis=-1)
    ttps = np.stack((ttps_hbo, ttps_hbr), axis=3)/10.0
    mags = np.max(grand_avg[:, :, :, :, 20:], axis=-1)
    peak_times = soc.get_info_peaks(grand_avg, type="hbo")[0].mean(axis=0)
    peak_time = peak_times[[CONDS.index('V'), CONDS.index('N')]].mean()/10
    print(f'Time window: {peak_time-2}-{peak_time+2} seconds')
    avgs = soc.window_average(grand_avg, window=[peak_time-2, peak_time+2])

    # Calculate contrasts (subjects, rois, types)
    grand_avg_ct = (grand_avg[:, CONDS.index('V'), ...]
                    - grand_avg[:, CONDS.index('N'), ...])
    ttps_ct = ttps[:, CONDS.index('V'), ...] - ttps[:, CONDS.index('N'), ...]
    mags_ct = mags[:, CONDS.index('V'), ...] - mags[:, CONDS.index('N'), ...]
    avgs_ct = avgs[:, CONDS.index('V'), ...] - avgs[:, CONDS.index('N'), ...]

    # Create long format data frame rows
    for i_subj, id in enumerate(subj_ids):
        if id not in grand_avg_ct_dict.keys():
            raise Exception(f"{id} not in the ID dictionary")
        grand_avg_ct_dict[id][i_age] = grand_avg_ct[i_subj]
        for i_roi, roi in enumerate(roi_list):
            # df_rows.append([id, age, 'S', roi, 'hbo',
            #                 avgs[i_subj, CONDS.index('S'), i_roi, 0],
            #                 ttps[i_subj, CONDS.index('S'), i_roi, 0],
            #                 mags[i_subj, CONDS.index('S'), i_roi, 0]])
            # df_rows.append([id, age, 'S', roi, 'hbr',
            #                 avgs[i_subj, CONDS.index('S'), i_roi, 1],
            #                 ttps[i_subj, CONDS.index('S'), i_roi, 1],
            #                 mags[i_subj, CONDS.index('S'), i_roi, 1]])
            df_rows.append([id, age, 'V', roi, 'hbo',
                            avgs[i_subj, CONDS.index('V'), i_roi, 0],
                            ttps[i_subj, CONDS.index('V'), i_roi, 0],
                            mags[i_subj, CONDS.index('V'), i_roi, 0]])
            df_rows.append([id, age, 'V', roi, 'hbr',
                            avgs[i_subj, CONDS.index('V'), i_roi, 1],
                            ttps[i_subj, CONDS.index('V'), i_roi, 1],
                            mags[i_subj, CONDS.index('V'), i_roi, 1]])
            df_rows.append([id, age, 'N', roi, 'hbo',
                            avgs[i_subj, CONDS.index('N'), i_roi, 0],
                            ttps[i_subj, CONDS.index('N'), i_roi, 0],
                            mags[i_subj, CONDS.index('N'), i_roi, 0]])
            df_rows.append([id, age, 'N', roi, 'hbr',
                            avgs[i_subj, CONDS.index('N'), i_roi, 1],
                            ttps[i_subj, CONDS.index('N'), i_roi, 1],
                            mags[i_subj, CONDS.index('N'), i_roi, 1]])
            df_rows.append([id, age, 'V-N', roi, 'hbo',
                            avgs_ct[i_subj, i_roi, 0],
                            ttps_ct[i_subj, i_roi, 0],
                            mags_ct[i_subj, i_roi, 0]])
            df_rows.append([id, age, 'V-N', roi, 'hbr',
                            avgs_ct[i_subj, i_roi, 1],
                            ttps_ct[i_subj, i_roi, 1],
                            mags_ct[i_subj, i_roi, 1]])

# Create data frames
cols = ('ID', 'Age (months)', 'Condition', 'ROI', 'Channel type',
        'Window average', 'Time-to-peak', 'Magnitude')
df = pd.DataFrame(df_rows, columns=cols)
df = df.sort_values(by=['ID', 'Age (months)', 'Condition', 'ROI'],
                    ignore_index=True)
df['Age (months)'] = df['Age (months)'].str[:-2]


# %% Get participants with all sessions until 1 year
# Count how many participants have data for all time points
complete_subj_ids, nz = [], []
roi_contribs = {age: {roi: 0 for roi in roi_list} for age in AGES}
for subj in all_subj_ids:
    nz.append(np.count_nonzero(~np.all(np.isnan(grand_avg_ct_dict[subj]),
                                       axis=(1, 2, 3))))
    if np.count_nonzero(~np.all(np.isnan(grand_avg_ct_dict[subj][:3]),
                                axis=(1, 2, 3))) >= 3:
        complete_subj_ids.append(subj)
        print(~np.all(np.isnan(grand_avg_ct_dict[subj]), axis=(1, 2, 3)))
    # if np.all(~np.all(np.isnan(grand_avg_ct_dict[subj]), axis=(1, 2, 3))):
    #     complete_subj_ids.append(subj)
    for i_age, age in enumerate(AGES):
        for i_roi, roi in enumerate(roi_list):
            if ~np.all(np.isnan(grand_avg_ct_dict[subj][i_age][i_roi])):
                roi_contribs[age][roi] += 1
print('Total:', len(complete_subj_ids))
print(complete_subj_ids)

# Histogram of number of age points
sns.histplot(nz, discrete=True)

# Print number of participants contributing to each ROI
for key, value in roi_contribs.items():
    print(key)
    print(value)


# %% Social selectivity
# Count proportion of selective at each age point
n = [127, 110, 116, 111, 112, 124]
rows = []
for i_age, age in enumerate(AGES):
    sub_df = df[df['Channel type'] == 'hbo']
    sub_df = sub_df[sub_df['Age (months)'] == age[:-2]]
    sub_df = sub_df[sub_df['ID'].isin(all_subj_ids)]
    n_spec_v = 0
    n_spec_n = 0
    n_spec_none = 0
    for subj in all_subj_ids:
        d = sub_df.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        if (d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'V-N'")["Window average"].mean() > 0) and (d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'V'")["Window average"].mean() > 0):
            n_spec_v += 1
        elif (d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'V-N'")["Window average"].mean() < 0) and (d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'N'")["Window average"].mean() > 0):
            n_spec_n += 1
        elif not d.empty:
            # print(d)
            n_spec_none += 1
    rows.append([age, round(n_spec_v/n[i_age]*100), round(n_spec_n/n[i_age]*100), round(n_spec_none/n[i_age]*100)])

df_barplot = pd.DataFrame(np.array(rows), columns=['age', 'V', 'N', 'None'])
df_barplot.set_index('age')
df_barplot['V'] = pd.to_numeric(df_barplot['V'])
df_barplot['N'] = pd.to_numeric(df_barplot['N'])
df_barplot['None'] = pd.to_numeric(df_barplot['None'])

ax = df_barplot.plot(kind='bar', stacked=True, x='age', width=0.95,
                     color=['#8b59bf', '#62bf59', 'darkgrey'])
ax.legend(bbox_to_anchor=(1.01, 0.5), loc=6, title='Selectivity', labels=['Auditory social', 'Auditory non-social', 'Non-selective'])
ax.set_xlabel('Age (months)')
ax.set_ylabel('Percentage of participants')

n_patches = n * 3
# Annotate the bars
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width/2,
            y + height/2,
            f'{str(int(height))} %',
            horizontalalignment='center',
            verticalalignment='center')


# %% Details
roi = "right anterior-temporal"
age = "60"

sub_df = df[df['Channel type'] == 'hbo']
sub_df = sub_df[sub_df['ID'].isin(complete_subj_ids)]

print((sub_df[(sub_df['ROI'] == roi) & (sub_df['Condition'] == "V") & (sub_df["Age (months)"] == age)].set_index("ID")["Window average"].subtract(sub_df[(sub_df['ROI'] == roi) & (sub_df['Condition'] == "N") & (sub_df["Age (months)"] == age)].set_index("ID")["Window average"])>0).value_counts())
print(len((sub_df[(sub_df['ROI'] == roi) & (sub_df['Condition'] == "V") & (sub_df["Age (months)"] == age)].set_index("ID")["Window average"].subtract(sub_df[(sub_df['ROI'] == roi) & (sub_df['Condition'] == "N") & (sub_df["Age (months)"] == age)].set_index("ID")["Window average"])>0)))

print(sub_df[(sub_df['ROI'] == roi) & (sub_df['Condition'] == "V") & (sub_df["Age (months)"] == age)].set_index("ID")["Window average"].mean())
print(sub_df[(sub_df['ROI'] == roi) & (sub_df['Condition'] == "N") & (sub_df["Age (months)"] == age)].set_index("ID")["Window average"].mean())


# %% Plot figures
# # Visual
# fig.violin_hemis(df, roi_list, complete_subj_ids, 'S', 'hbo', ylim=[-1.5, 1.5])
# fig.violin_hemis(df, roi_list, complete_subj_ids, 'S', 'hbr', ylim=[-1, 1])
# fig.paired_scatter_hemis(df, roi_list, AGES, complete_subj_ids, 'S', 'hbo', ylim=[-1.5, 1.5])
# fig.paired_scatter_hemis(df, roi_list, AGES, complete_subj_ids, 'S', 'hbr', ylim=[-2, 2])

# Auditory
fig.paired_scatter_contrast(df, roi_list, AGES, complete_subj_ids, 'hbo', ylim=[-2, 2.8])
fig.paired_scatter_contrast(df, roi_list, AGES, complete_subj_ids, 'hbr', ylim=[-2, 2.5])
fig.violin_hemis(df, roi_list, complete_subj_ids, 'V-N', 'hbo', ylim=[-2, 2.5])
fig.violin_hemis(df, roi_list, complete_subj_ids, 'V-N', 'hbr', ylim=[-1.5, 1.5])
fig.paired_scatter_hemis(df, roi_list, AGES, complete_subj_ids, 'V-N', 'hbo', ylim=[-2, 2.5])
fig.paired_scatter_hemis(df, roi_list, AGES, complete_subj_ids, 'V-N', 'hbr', ylim=[-2, 2])

fig.trajectories(df, roi_list, complete_subj_ids, 'V-N', 'hbo', ylim=[-2, 2.5])
fig.trajectories(df, roi_list, complete_subj_ids, 'V-N', 'hbr', ylim=[-1, 1])

fig.selective_trajectories(df, roi_list, complete_subj_ids, 'V-N', 'hbo', ylim=[-1, 1])
fig.selective_trajectories(df, roi_list, complete_subj_ids, 'V-N', 'hbr', ylim=[-1, 2])

fig.selective_trajectories(df, roi_list, all_subj_ids, 'V-N', 'hbo', ylim=[-1, 1])

fig.selective_sustained(df, all_subj_ids, 'V-N', 'hbo')


# %% Plot individuals
FEATURE = 'Window average'
TYPE = 'hbo'

# Plot waveforms longitudinally
ch_types = ['hbo', 'hbr']
for subj in complete_subj_ids:
    print(subj)
    file_name = f'../../outputs/{subj}.png'
    fig, axes = plt.subplots(3, 2, figsize=(18, 7))
    for i_roi, roi in enumerate(roi_list):
        data = grand_avg_ct_dict[subj][:, i_roi, ch_types.index(TYPE), :]
        plot = sns.heatmap(data, ax=axes.flat[i_roi], vmin=-1, vmax=1,
                           xticklabels=False, yticklabels=AGES, cmap="RdBu_r")
        axes.flat[i_roi].title.set_text(roi)
    plt.tight_layout(pad=3)
    # plt.show()
    plt.savefig(file_name)
    plt.close()

# Plot features longitudinally
for subj in complete_subj_ids:
    print(subj)
    file_name = f'../../outputs/avg_{subj}.png'
    fig, axes = plt.subplots(3, 2, figsize=(18, 7))
    for i_roi, roi in enumerate(roi_list):
        data = df[(df['ID'] == subj) & (df['ROI'] == roi) &
                  (df['Channel type'] == TYPE) & (df['Condition'] == 'V-N')]
        plot = sns.barplot(data=data, x='Age (months)', y=FEATURE,
                           ax=axes.flat[i_roi],
                           order=['5', '8', '12', '18', '24', '60'])
        axes.flat[i_roi].title.set_text(roi)
    plt.tight_layout(pad=3)
    # plt.show()
    plt.savefig(file_name)
    plt.close()


# %% Cross sectional analysis with complete subjects
# Block average with complete subjects
for i_age, age in enumerate(AGES):
    print(f'##### {age} #####')

    # Load epochs
    path = f'[path to dataset]/results/{age}/'
    if age == '1mo':
        grand_avg, subj_ids, rejected = soc.load_results_1mo(path, CONDS)
    elif age == '60mo':
        grand_avg, subj_ids, rejected, _ = soc.load_results_60mo(path, CONDS)
    else:
        grand_avg, subj_ids, rejected, _ = soc.load_results_infancy(path, CONDS)

    # Get indices of subjects in the complete subject set
    keep_indices = [
        i for i, id in enumerate(subj_ids) if id in complete_subj_ids]
    grand_avg = grand_avg[keep_indices]
    subj_ids = np.array(subj_ids)[keep_indices].tolist()
    print(len(subj_ids))

    # Combine channels by ROIs (subjects, conds, rois, types, times)
    if age != '1mo':
        dummy_channels = [0, 3, 20, 23]
        for dummy in dummy_channels:
            grand_avg = np.insert(grand_avg, dummy, np.nan, axis=2)
    roi_grand_avg = np.empty((grand_avg.shape[0], grand_avg.shape[1], 0,
                              grand_avg.shape[3], grand_avg.shape[4]))
    for roi in roi_list:
        roi_avg = np.nanmean(grand_avg[:, :, roi_dict[age][roi], :, :],
                             axis=2, keepdims=True)
        roi_grand_avg = np.append(roi_grand_avg, roi_avg, axis=2)
    grand_avg = roi_grand_avg

    # Extract features
    ttps_hbo = np.argmax(grand_avg[:, :, :, 0, 20:], axis=-1)
    ttps_hbr = np.argmin(grand_avg[:, :, :, 1, 20:], axis=-1)
    ttps = np.stack((ttps_hbo, ttps_hbr), axis=3)/10.0
    mags = np.max(grand_avg[:, :, :, :, 20:], axis=-1)
    peak_times = soc.get_info_peaks(grand_avg, type="hbo")[0].mean(axis=0)
    peak_time = peak_times[[CONDS.index('V'), CONDS.index('N')]].mean()/10
    avgs = soc.window_average(grand_avg, window=[peak_time-2, peak_time+2])

    # Calculate contrasts (subjects, rois, types)
    grand_avg_ct = (grand_avg[:, CONDS.index('V'), ...]
                    - grand_avg[:, CONDS.index('N'), ...])
    ttps_ct = ttps[:, CONDS.index('V'), ...] - ttps[:, CONDS.index('N'), ...]
    mags_ct = mags[:, CONDS.index('V'), ...] - mags[:, CONDS.index('N'), ...]
    avgs_ct = avgs[:, CONDS.index('V'), ...] - avgs[:, CONDS.index('N'), ...]

    # # Plot S activation
    # s_peak_times = soc.get_info_peaks(grand_avg, type="hbo")[0].mean(axis=0)
    # s_peak_time = s_peak_times[CONDS.index('S')].mean()/10
    # s_avgs = soc.window_average(
    #     grand_avg, window=[s_peak_time-2, s_peak_time+2])
    # s_act = soc.analyse_act(s_avgs, CONDS.index('S'), fdr=True)
    # soc.topo_overlay(s_act[-1], None)
    # soc.plot_hrf(grand_avg, CONDS.index('S'), s_act[-1])

    # Plot contrast activation
    vn_grand_ct = grand_avg_ct[:, np.newaxis, :, :, :]
    vn_act = soc.analyse_contrast(
        avgs, CONDS.index('V'), CONDS.index('N'), fdr=True, dummies=False)
    soc.topo_overlay_roi(vn_act[-1], None)
    # soc.plot_hrf(vn_grand_ct, 0, vn_act[-1], baseline=2)


# %% Correlation of contrast waveform window average with age
with open('../../assets/ids.json', 'r') as f:
    participant_ids = json.load(f)
participant_ids = {participant_ids[key]: key for key in participant_ids}
df_bright = pd.read_csv('../../assets/anthrops.csv', index_col=None, delimiter=",")
df_kids = pd.read_csv('../../assets/anthrops_60mo.csv', index_col=None, delimiter=",")
df_anthrop = pd.merge(df_bright, df_kids, on='id', how='outer')
df_anthrop = df_anthrop.dropna(subset=['id', 'famid'])
df_anthrop['testid'] = df_anthrop['id'].map(participant_ids)

agepoints = {
    'id': 'id', 'testid': 'testid',
    'sex': 'sex',
    'agem4': 'agem_birth', 'agem5': 'agem_week', 'agem6': 'agem_1mo', 'agem7': 'agem_5mo', 'agem8': 'agem_8mo', 'agem9': 'agem_12mo', 'agem10': 'agem_18mo', 'agem11': 'agem_24mo', 'agem_60mo': 'agem_60mo',
    'hb4': 'hb_birth', 'hb5': 'hb_week', 'hb6': 'hb_1mo', 'hb7': 'hb_5mo', 'hb8': 'hb_8mo', 'hb9': 'hb_12mo', 'hb10': 'hb_18mo', 'hb11': 'hb_24mo', 'hb_60mo': 'hb_60mo',
    'whz4': 'whz_birth', 'whz5': 'whz_week', 'whz6': 'whz_1mo', 'whz7': 'whz_5mo', 'whz8': 'whz_8mo', 'whz9': 'whz_12mo', 'whz10': 'whz_18mo', 'whz11': 'whz_24mo', 'whz_60mo': 'whz_60mo',
    'haz4': 'haz_birth', 'haz5': 'haz_week', 'haz6': 'haz_1mo', 'haz7': 'haz_5mo', 'haz8': 'haz_8mo', 'haz9': 'haz_12mo', 'haz10': 'haz_18mo', 'haz11': 'haz_24mo', 'haz_60mo': 'haz_60mo'
}
df_anthrop.rename(columns=agepoints, inplace=True)
df_anthrop['hb_60mo'] = np.nan
df_anthrop = df_anthrop[agepoints.values()]
df_anthrop = df_anthrop.replace({'sex': {0.0: 'Male', 1.0: 'Female'}})

p_values = []
sub_df = df[df['Condition'] == 'V-N']
sub_df = sub_df[sub_df['Channel type'] == 'hbo']
sub_df = sub_df[sub_df['ID'].isin(all_subj_ids)]
sub_df = sub_df.query("ROI.str.contains('anterior')")
for i_age, age in enumerate(AGES[:3]):
    subj_means = []
    subj_ages = []
    for subj in complete_subj_ids:
        d = sub_df.query(f"ID == '{subj}'")
        subj_mean = d.query(f"`Age (months)` == '{age[:-2]}'")["Window average"].mean()
        subj_age = df_anthrop[df_anthrop['testid'] == subj][f'agem_{age}'].values
        subj_means.append(subj_mean)
        subj_ages.append(subj_age)
    subj_ages = np.array(subj_ages).squeeze()
    subj_means = np.array(subj_means)
    s, p = pearsonr(subj_ages, subj_means)
    p_values.append(p)
    print(age, s, p)
    plt.figure()
    sns.regplot(x=subj_ages, y=subj_means)
    plt.xlabel('Age (months)')
    plt.ylabel('V-N HbO (µM)')
    plt.show()

_, fdr_p_values = multitest.fdrcorrection(p_values, alpha=0.05)
print(fdr_p_values)


# %% Mixed linear model of contrast waveform window average with age
mem_df = sub_df[sub_df['Age (months)'].isin(['5', '8', '12'])]
mem_df['Age (months)']
mem_df = mem_df.rename(columns={"ID": "testid", "Window average": "win_avg", "Age (months)": "agem"})
result = mem_df.merge(df_anthrop, on='testid', how="inner")
result = result.reset_index()
for i, r in result.iterrows():
    result.iloc[i, result.columns.get_loc('agem')] = r[f"agem_{r['agem']}mo"]

result['agem'] = pd.to_numeric(result['agem'])

# result = result[result['win_avg'].between(-2, 2)]

model = smf.mixedlm("win_avg ~ agem", result,
                    groups=result['testid'], missing='drop')
fitted_model = model.fit()
summary = fitted_model.summary()
params = fitted_model.params
std_errors = fitted_model.bse
p_vals = fitted_model.pvalues
summary_df = pd.DataFrame({
    'Parameter': fitted_model.params.index,
    'Estimate': fitted_model.params.values,
    'Std. Error': fitted_model.bse.values,
    'P>|z|': fitted_model.pvalues.values
})
formatted_summary_df = summary_df.map(lambda x: f"{x:.5f}" if isinstance(x, float) else x)
print(formatted_summary_df)
plot = sns.regplot(result, x='agem', y='win_avg', scatter_kws={"color": "indianred"}, line_kws={"color": "black"})
plt.xlabel("Age (months)")
plt.ylabel("Window average (µM)")
plot.set_axisbelow(True)
plt.grid(which='major', visible=True, color='silver', linewidth=1.)
plt.grid(which='minor', visible=True, color='silver', linewidth=0.5, linestyle='--')
plt.minorticks_on()
plt.show()


# %% Sustained selectivity
df = pd.read_csv('../../outputs/selective_sustained.csv')

sub_df = df[df['5mo'] == 'N']

sub_df['countV'] = df.apply(lambda row: row.eq('V').sum(), axis=1)
sub_df['countN'] = df.apply(lambda row: row.eq('N').sum()-1, axis=1)
sub_df['countNone'] = df.apply(lambda row: row.isna().sum(), axis=1)
sub_df['ratio'] = sub_df['countV'] / (sub_df['countV']+sub_df['countN']+sub_df['countNone'])
len(sub_df[sub_df['ratio'] > 0.5])/len(sub_df)
