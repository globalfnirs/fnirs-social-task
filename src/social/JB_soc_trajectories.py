"""
Script for social task trajectories analysis
Author: Johann Benerradi
------------------------
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import functions.soc as soc

from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


AGES = ['5mo', '8mo', '12mo', '18mo', '24mo', '60mo']
CONDS = ['V', 'N']

# plt.switch_backend('QtAgg')


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# Get ROI dict
with open('../../assets/rois.json', 'r') as f:
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


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
# Load NIRS data
grand_avg_ct_dict = {
    id: np.empty((len(AGES), len(roi_list), 2, 221))*np.nan
    for id in all_subj_ids
}
df_rows = []
sessions_n = []
for i_age, age in enumerate(AGES):
    print(f'===============\n{age} session\n---------------')

    # Load epochs
    path = f'../../../data/results/{age}/'
    if age == '60mo':
        grand_avg, subj_ids, rejected, _, _ = soc.load_60mo(path, CONDS)
    else:
        grand_avg, subj_ids, rejected, _, _ = soc.load_infancy(path, CONDS)

    sessions_n.append(len(grand_avg))

    # Print data rejection info
    n_trials = len([r for r in rejected if r[1] == 'trials'])
    n_chs = len([r for r in rejected if r[1] == 'channels'])
    print(f'{n_trials+n_chs} rejected participants (reasons: '
          f'trials → {n_trials}; channels → {n_chs})')

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
    print(f'Time window: {peak_time-2} → {peak_time+2} seconds')
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
            for i, chroma in enumerate(['hbo', 'hbr']):
                df_rows.append([
                    id, age, 'V', roi, chroma,
                    avgs[i_subj, CONDS.index('V'), i_roi, i],
                    ttps[i_subj, CONDS.index('V'), i_roi, i],
                    mags[i_subj, CONDS.index('V'), i_roi, i]
                ])
                df_rows.append([
                    id, age, 'N', roi, chroma,
                    avgs[i_subj, CONDS.index('N'), i_roi, i],
                    ttps[i_subj, CONDS.index('N'), i_roi, i],
                    mags[i_subj, CONDS.index('N'), i_roi, i]
                ])
                df_rows.append([
                    id, age, 'V-N', roi, chroma,
                    avgs_ct[i_subj, i_roi, i],
                    ttps_ct[i_subj, i_roi, i],
                    mags_ct[i_subj, i_roi, i]
                ])

# Create data frames
cols = ('ID', 'Age (months)', 'Condition', 'ROI', 'Channel type',
        'Window average', 'Time-to-peak', 'Magnitude')
df = pd.DataFrame(df_rows, columns=cols)
df = df.sort_values(by=['ID', 'Age (months)', 'Condition', 'ROI'],
                    ignore_index=True)
df['Age (months)'] = df['Age (months)'].str[:-2]


# -----------------------------------------------------------------------------
# Subeselection of participants with all sessions from 5 to 12 months
# -----------------------------------------------------------------------------
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
    for i_age, age in enumerate(AGES):
        for i_roi, roi in enumerate(roi_list):
            if ~np.all(np.isnan(grand_avg_ct_dict[subj][i_age][i_roi])):
                roi_contribs[age][roi] += 1
print('Total:', len(complete_subj_ids))
print(complete_subj_ids)

# Histogram of number of age points
sns.histplot(nz, discrete=True)
plt.xlabel("Number of valid sessions")
plt.ylabel("Number of participants")
plt.show()

# Print number of participants contributing to each ROI
for key, value in roi_contribs.items():
    print(key)
    print(value)


# -----------------------------------------------------------------------------
# Social selectivity
# -----------------------------------------------------------------------------
# Count proportion of selective at each age point
rows = []
df_selectivity = pd.DataFrame(columns=['ID', 'Age (months)', 'Selectivity'])
for i_age, age in enumerate(AGES):
    sub_df = df[df['Channel type'] == 'hbo']
    sub_df = sub_df[sub_df['Age (months)'] == age[:-2]]
    sub_df = sub_df[sub_df['ID'].isin(all_subj_ids)]
    sub_df_hbr = df[df['Channel type'] == 'hbr']
    sub_df_hbr = sub_df_hbr[sub_df_hbr['Age (months)'] == age[:-2]]
    sub_df_hbr = sub_df_hbr[sub_df_hbr['ID'].isin(all_subj_ids)]
    n_spec_v = 0
    n_spec_n = 0
    n_spec_none = 0
    for subj in all_subj_ids:
        d = sub_df.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_hbr = sub_df_hbr.query(
            f"ID == '{subj}' & ROI.str.contains('anterior')"
        )
        if (
            d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'V-N'")[
                "Window average"].mean() > 0
            and
            d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'V'")[
                "Window average"].mean() > 0
        ):
            n_spec_v += 1
            df_selectivity.loc[len(df_selectivity)] = [subj, age[:-2], 'V']

        elif (
            d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'V-N'")[
                "Window average"].mean() < 0
                and
                d.query(f"`Age (months)` == '{age[:-2]}' & Condition == 'N'")[
                    "Window average"].mean() > 0
        ):
            n_spec_n += 1
            df_selectivity.loc[len(df_selectivity)] = [subj, age[:-2], 'N']

        elif not d.empty:
            n_spec_none += 1
            df_selectivity.loc[len(df_selectivity)] = [subj, age[:-2], None]
    print(n_spec_v, n_spec_n, n_spec_none)
    rows.append([
        age, round(n_spec_v/sessions_n[i_age]*100),
        round(n_spec_n/sessions_n[i_age]*100),
        round(n_spec_none/sessions_n[i_age]*100)
    ])

df_barplot = pd.DataFrame(np.array(rows), columns=['age', 'V', 'N', 'None'])
df_barplot.set_index('age')
df_barplot['V'] = pd.to_numeric(df_barplot['V'])
df_barplot['N'] = pd.to_numeric(df_barplot['N'])
df_barplot['None'] = pd.to_numeric(df_barplot['None'])

ax = df_barplot.plot(kind='bar', stacked=True, x='age', width=0.95,
                     color=['#8b59bf', '#62bf59', 'darkgrey'])
ax.legend(bbox_to_anchor=(1.01, 0.5), loc=6, title='Selectivity',
          labels=['Auditory social', 'Auditory non-social', 'Non-selective'])
ax.set_xlabel('Age (months)')
ax.set_ylabel('Percentage of participants')

n_patches = sessions_n * 3
# Annotate the bars
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x + width/2,
            y + height/2,
            f'{str(int(height))} %',
            horizontalalignment='center',
            verticalalignment='center')


# -----------------------------------------------------------------------------
# Details
# -----------------------------------------------------------------------------
soc.selective_paired(df, roi_list, AGES, complete_subj_ids, 'hbo',
                     ylim=[-2, 2.8])

soc.selective_trajectories(df, complete_subj_ids, 'hbo', ylim=[-0.7, 0.7])
soc.selective_trajectories(df, complete_subj_ids, 'hbr', ylim=[-0.7, 0.7])

selectivity_table = soc.selective_table(df, all_subj_ids)
print(selectivity_table)


# -----------------------------------------------------------------------------
# Mixed GLM of V selectivity with age
# -----------------------------------------------------------------------------
with open('../../assets/ids.json', 'r') as f:
    participant_ids = json.load(f)
participant_ids = {participant_ids[key]: key for key in participant_ids}
df_bright = pd.read_csv('../../assets/anthrops.csv', index_col=None,
                        delimiter=",")
df_kids = pd.read_csv('../../assets/anthrops_60mo.csv', index_col=None,
                      delimiter=",")
df_anthrop = pd.merge(df_bright, df_kids, on='id', how='outer')
df_anthrop = df_anthrop.dropna(subset=['id', 'famid'])
df_anthrop['testid'] = df_anthrop['id'].map(participant_ids)

agepoints = {
    'id': 'id', 'testid': 'testid',
    'sex': 'sex',
    'agem4': 'agem_birth', 'agem5': 'agem_week', 'agem6': 'agem_1mo',
    'agem7': 'agem_5mo', 'agem8': 'agem_8mo', 'agem9': 'agem_12mo',
    'agem10': 'agem_18mo', 'agem11': 'agem_24mo', 'agem_60mo': 'agem_60mo',
    'hb4': 'hb_birth', 'hb5': 'hb_week', 'hb6': 'hb_1mo', 'hb7': 'hb_5mo',
    'hb8': 'hb_8mo', 'hb9': 'hb_12mo', 'hb10': 'hb_18mo', 'hb11': 'hb_24mo',
    'hb_60mo': 'hb_60mo',
    'whz4': 'whz_birth', 'whz5': 'whz_week', 'whz6': 'whz_1mo', 'whz7':
    'whz_5mo', 'whz8': 'whz_8mo', 'whz9': 'whz_12mo', 'whz10': 'whz_18mo',
    'whz11': 'whz_24mo', 'whz_60mo': 'whz_60mo',
    'haz4': 'haz_birth', 'haz5': 'haz_week', 'haz6': 'haz_1mo', 'haz7':
    'haz_5mo', 'haz8': 'haz_8mo', 'haz9': 'haz_12mo', 'haz10': 'haz_18mo',
    'haz11': 'haz_24mo', 'haz_60mo': 'haz_60mo'
}
df_anthrop.rename(columns=agepoints, inplace=True)
df_anthrop['hb_60mo'] = np.nan
df_anthrop = df_anthrop[agepoints.values()]
df_anthrop = df_anthrop.replace({'sex': {0.0: 'Male', 1.0: 'Female'}})

glmm_df = df_selectivity[df_selectivity['Age (months)'].isin(
    ['5', '8', '12', '18', '24', '60']
)]
glmm_df = glmm_df.rename(
    columns={"ID": "testid", "Age (months)": "agem", "Selectivity": "selec"}
)
result = glmm_df.merge(df_anthrop, on='testid', how="inner")
result = result.reset_index()
result['agem'] = result['agem'].astype(float)
for i, r in result.iterrows():
    result.iloc[i, result.columns.get_loc('agem')] = r[
        f"agem_{int(r['agem'])}mo"
    ]

result['selec_binaray'] = (result['selec'] == 'V').astype(int)

result = result.dropna(subset=['agem'])  # remove row with NaN in agem column

random = {"testid_intercept": '0 + C(testid)'}
model = BinomialBayesMixedGLM.from_formula('selec_binaray ~ agem', random,
                                           result)
fitted_model = model.fit_vb()
summary = fitted_model.summary()
print(summary)
