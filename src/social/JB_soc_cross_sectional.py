"""
Script for cross-sectional social task analysis
Author: Johann Benerradi
------------------------
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import functions.soc as soc
import statsmodels.api as sm

from scipy.stats import pearsonr
from statsmodels.stats import multitest
from statsmodels.formula.api import ols


AGES = [5, 8, 12, 18, 24, 60]
CONDS = ['S', 'V', 'N']

# plt.switch_backend('QtAgg')


# -----------------------------------------------------------------------------
# Anthropometrics
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
    'whz4': 'whz_birth', 'whz5': 'whz_week', 'whz6': 'whz_1mo',
    'whz7': 'whz_5mo', 'whz8': 'whz_8mo', 'whz9': 'whz_12mo',
    'whz10': 'whz_18mo', 'whz11': 'whz_24mo', 'whz_60mo': 'whz_60mo',
    'haz4': 'haz_birth', 'haz5': 'haz_week', 'haz6': 'haz_1mo',
    'haz7': 'haz_5mo', 'haz8': 'haz_8mo', 'haz9': 'haz_12mo',
    'haz10': 'haz_18mo', 'haz11': 'haz_24mo', 'haz_60mo': 'haz_60mo',
    'hc4': 'hc_birth', 'hc5': 'hc_week', 'hc6': 'hc_1mo', 'hc7': 'hc_5mo',
    'hc8': 'hc_8mo', 'hc9': 'hc_12mo', 'hc10': 'hc_18mo', 'hc11': 'hc_24mo',
    'hc_mean_60mo': 'hc_60mo',
}
df_anthrop.rename(columns=agepoints, inplace=True)
df_anthrop['hb_60mo'] = np.nan
df_anthrop = df_anthrop[agepoints.values()]
df_anthrop = df_anthrop.replace({'sex': {0.0: 'Male', 1.0: 'Female'}})

with open('../../assets/ids.json', 'r') as f:
    participant_ids = list(json.load(f).keys())


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
all_grand_avg = []
ttps_p_values = []  # (n_agepoints,)
trend_values = []
all_ttps_hbo = []  # (n_agepoint, n_subjects, n_conditions)
all_ttps_hbr = []  # (n_agepoint, n_subjects, n_conditions)
all_mags_hbo = []  # (n_agepoint, n_subjects, n_conditions)
all_mags_hbr = []  # (n_agepoint, n_subjects, n_conditions)
vnctlist = []
sctlist = []
all_vn_feature_avg = []
anthrop_table = pd.DataFrame(columns=[
    'Session', 'N', 'Percent female', 'Age (months)', 'WHZ',
    'Head circumference', 'Number channels', 'Number trials'
])
for i, age in enumerate(AGES):
    print(f'===============\n{age} mo session\n---------------')
    # Load block average data
    path = f'../../../data/results/{age}mo/'
    if age == 60:
        grand_avg, subj_ids, rejected, n_chs, n_trials = soc.load_60mo(
            path, CONDS
        )
    else:
        grand_avg, subj_ids, rejected, n_chs, n_trials = soc.load_infancy(
            path, CONDS
        )

    # Get anthropometric info
    df_age = df_anthrop[df_anthrop['testid'].isin(subj_ids)]
    anthrop_row = [
        age,
        len(subj_ids),
        f"{round((df_age['sex'] == 'Female').sum()*100 / len(df_age), 2)} \\%",
        (
            f"{round(df_age[f'agem_{age}mo'].mean(), 2)} $\\pm$ "
            f"{round(df_age[f'agem_{age}mo'].std(), 2)}"
        ),
        (
            f"{round(df_age[f'whz_{age}mo'].mean(), 2)} $\\pm$ "
            f"{round(df_age[f'whz_{age}mo'].std(), 2)}",
        ),
        (
            f"{round(df_age[f'hc_{age}mo'].mean(), 2)} $\\pm$ "
            f"{round(df_age[f'hc_{age}mo'].std(), 2)}",
        ),
        f"{round(np.mean(n_chs), 2)} $\\pm$ {round(np.std(n_chs), 2)}",
        f"{round(np.mean(n_trials), 2)} $\\pm$ {round(np.std(n_trials), 2)}"
    ]
    anthrop_table.loc[i] = anthrop_row

    # Add dummy channels
    dummy_channels = [0, 3, 20, 23]
    dummy_grand_avg = grand_avg.copy()
    for dummy in dummy_channels:
        dummy_grand_avg = np.insert(dummy_grand_avg, dummy, 0, axis=2)
    all_grand_avg.append(dummy_grand_avg)

    for row in subj_ids:
        if row not in participant_ids:
            print('Not in subject dictionary:', row)
    n_rtrials = len([r for r in rejected if r[1] == 'trials'])
    n_rchs = len([r for r in rejected if r[1] == 'channels'])
    print(f'{n_rtrials+n_rchs} rejected participants (reasons: '
          f'trials → {n_rtrials}; channels → {n_rchs})')

    # Compute peak infos
    ttps_hbo, mags_hbo = soc.get_info_peaks(grand_avg, type="hbo")
    ttps_hbr, mags_hbr = soc.get_info_peaks(grand_avg, type="hbr")
    all_ttps_hbo.append(ttps_hbo)
    all_ttps_hbr.append(ttps_hbr)
    all_mags_hbo.append(mags_hbo)
    all_mags_hbr.append(mags_hbr)

    stats_ttps = soc.stats_ttp(ttps_hbo, CONDS.index('V'), CONDS.index('N'))
    ttps_p_values.append(stats_ttps[0])
    trend_values.append(stats_ttps[-1])

    # S
    print('Visual social (S)')
    s_peak_time = ttps_hbo.mean(axis=0)[CONDS.index('S')].mean()/10
    print(f'Time window S: {s_peak_time-2} → {s_peak_time+2} seconds')
    s_feature_avg = soc.window_average(
        grand_avg, window=[s_peak_time-2, s_peak_time+2]
    )
    s_ct = soc.analyse_act(
        s_feature_avg, CONDS.index('S'), fdr=True)
    soc.topo_overlay(s_ct[-1])
    sctlist.append(s_ct[-1])

    # V - N
    print('Auditory vocal > non-vocal (V-N)')
    vn_peak_time = ttps_hbo.mean(axis=0)[
        [CONDS.index('V'), CONDS.index('N')]
    ].mean()/10
    print(f'Time window VN: {vn_peak_time-2} → {vn_peak_time+2} seconds')
    vn_feature_avg = soc.window_average(
        grand_avg, window=[vn_peak_time-2, vn_peak_time+2]
    )
    all_vn_feature_avg.append(vn_feature_avg)
    vn_ct = soc.analyse_contrast(
        vn_feature_avg, CONDS.index('V'), CONDS.index('N'), fdr=True)
    soc.topo_overlay(vn_ct[-1])
    vnctlist.append(vn_ct[-1])

_, fdr_ttp_p_values = multitest.fdrcorrection(ttps_p_values, alpha=0.05)
print(fdr_ttp_p_values)
print(trend_values)

anthrop_table.to_csv('../../outputs/social_info.csv', index=False)


# -----------------------------------------------------------------------------
# Visualise channel HRF
# -----------------------------------------------------------------------------
I_CHANNEL = 31
cond_legend = {'S': 'Visual social',
               'V': 'Auditory social',
               'N': 'Auditory non-social'}

custom_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700',
                 '#8A2BE2']

scale = [-0.4, 1.2]


for cond in cond_legend:
    i_cond = CONDS.index(cond)

    fig, ax = plt.subplots(figsize=(7, 5))

    full_df = pd.DataFrame()
    for i, age in enumerate(AGES):
        df = pd.DataFrame({
            'Hemodynamic response (µM)': np.nanmean(
                all_grand_avg[i][:, i_cond, I_CHANNEL, 0, :], axis=0
            ),
            'Age': f'{age} mo (HbO)'
        })
        df['Time (sec)'] = (np.array(df.index) - 20) / 10
        full_df = pd.concat([full_df, df])
    p = sns.lineplot(data=full_df, x='Time (sec)',
                     y='Hemodynamic response (µM)', hue='Age',
                     palette=sns.color_palette("flare"), ax=ax)
    p.set(ylim=scale)

    full_df = pd.DataFrame()
    for i, age in enumerate(AGES):
        df = pd.DataFrame({
            'Hemodynamic response (µM)': np.nanmean(
                all_grand_avg[i][:, i_cond, I_CHANNEL, 1, :], axis=0
            ),
            'Age': f'{age} mo (HbR)'
        })
        df['Time (sec)'] = (np.array(df.index) - 20) / 10
        full_df = pd.concat([full_df, df])
    p = sns.lineplot(data=full_df, x='Time (sec)',
                     y='Hemodynamic response (µM)', hue='Age',
                     palette=sns.color_palette("crest"), ax=ax, dashes=(2, 1))
    p.set(ylim=scale)

    plt.legend(title='Age')
    plt.title(f'{cond_legend[cond]}')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


# -----------------------------------------------------------------------------
# HRF magnitude analysis
# -----------------------------------------------------------------------------
big_table = np.empty((0, 3))
for i, agepoint in enumerate(all_mags_hbo):
    table = np.empty((0))
    for column in range(agepoint.shape[1]):
        table = np.concatenate((table, agepoint[:, column]))
    table = table[:, np.newaxis]
    table = np.hstack((
        table,
        np.repeat(
            np.array([
                'Silent (HbO)', 'Vocal (HbO)', 'Non-vocal (HbO)'
            ]),
            len(agepoint)
        )[:, np.newaxis]
    ))
    table = np.hstack((np.full((len(table), 1), AGES[i]), table))
    big_table = np.vstack((big_table, table))
df_hbo = pd.DataFrame(
    data=big_table,
    columns=["Age (months)", "HRF peak amplitude (μM)", "Condition"]
)
big_table_hbr = np.empty((0, 3))
for i, agepoint in enumerate(all_mags_hbr):
    table = np.empty((0))
    for column in range(agepoint.shape[1]):
        table = np.concatenate((table, agepoint[:, column]))
    table = table[:, np.newaxis]
    table = np.hstack((
        table,
        np.repeat(
            np.array([
                'Silent (HbR)', 'Vocal (HbR)', 'Non-vocal (HbR)'
            ]),
            len(agepoint)
        )[:, np.newaxis]
    ))
    table = np.hstack((np.full((len(table), 1), AGES[i]), table))
    big_table_hbr = np.vstack((big_table_hbr, table))
df_hbr = pd.DataFrame(
    data=big_table_hbr,
    columns=["Age (months)", "HRF peak amplitude (μM)", "Condition"]
)
df = pd.concat([df_hbo, df_hbr])
df["Age (months)"] = pd.to_numeric(df["Age (months)"])
df["HRF peak amplitude (μM)"] = pd.to_numeric(df["HRF peak amplitude (μM)"])
df = df[df["Condition"] != "Control"]

df_kids = df[df['Age (months)'] == 60].copy()
df_kids['Age (months)'] = df_kids['Age (months)'] - 20
df = df[df['Age (months)'] != 60].copy()

fig, axes = plt.subplots(1, 1, figsize=(8, 5))
sns.lineplot(
    df, x="Age (months)", y="HRF peak amplitude (μM)", hue="Condition",
    style="Condition", markers=["o", "^", "s", "o", "^", "s"],
    dashes=[(2, 0), (2, 2), (2, 4), (2, 0), (2, 2), (2, 4)],
    palette=['indianred', 'indianred', 'indianred',
             'royalblue', 'royalblue', 'royalblue'],
    legend=True, errorbar='se'
)
sns.pointplot(
    df_kids, x="Age (months)", y="HRF peak amplitude (μM)", hue="Condition",
    palette=['indianred', 'indianred', 'indianred',
             'royalblue', 'royalblue', 'royalblue'],
    scale=0.8, markers=["o", "^", "s", "o", "^", "s"], native_scale=True,
    dodge=0.5, err_kws=dict(alpha=0.5), legend=False, errorbar='se'
)
plt.grid()
plt.xlim(0, 45)
plt.ylim(-0.25, 0.6)
plt.ylabel("HRF peak amplitude (μM)", fontsize=12)
plt.xlabel("Age (months)", fontsize=12)
plt.xticks(np.arange(0, 45, 10))
labels = [item.get_text() for item in axes.get_xticklabels()]
labels[-1] = '3-5 y'
axes.set_xticklabels(labels)
plt.legend(bbox_to_anchor=(1.01, 0.5), loc=6)
plt.tight_layout()
plt.show()

df_mag = df.copy()


# -----------------------------------------------------------------------------
# HRF time-to-peak analysis
# -----------------------------------------------------------------------------
big_table = np.empty((0, 3))
for i, agepoint in enumerate(all_ttps_hbo):
    table = np.empty((0))
    for column in range(agepoint.shape[1]):
        table = np.concatenate((table, agepoint[:, column]))
    table = table[:, np.newaxis]
    table = np.hstack((
        table,
        np.repeat(
            np.array(
                ['Silent (HbO)', 'Vocal (HbO)', 'Non-vocal (HbO)']
            ),
            len(agepoint)
        )[:, np.newaxis]
    ))
    table = np.hstack((np.full((len(table), 1), AGES[i]), table))
    big_table = np.vstack((big_table, table))
df_hbo = pd.DataFrame(
    data=big_table,
    columns=["Age (months)", "HRF time-to-peak (sec)", "Condition"]
)
big_table_hbr = np.empty((0, 3))
for i, agepoint in enumerate(all_ttps_hbr):
    table = np.empty((0))
    for column in range(agepoint.shape[1]):
        table = np.concatenate((table, agepoint[:, column]))
    table = table[:, np.newaxis]
    table = np.hstack((
        table,
        np.repeat(
            np.array(
                ['Silent (HbR)', 'Vocal (HbR)', 'Non-vocal (HbR)']
            ),
            len(agepoint)
        )[:, np.newaxis]
    ))
    table = np.hstack((np.full((len(table), 1), AGES[i]), table))
    big_table_hbr = np.vstack((big_table_hbr, table))
df_hbr = pd.DataFrame(
    data=big_table_hbr,
    columns=["Age (months)", "HRF time-to-peak (sec)", "Condition"]
)
df = pd.concat([df_hbo, df_hbr])
df["Age (months)"] = pd.to_numeric(df["Age (months)"])
df["HRF time-to-peak (sec)"] = pd.to_numeric(df["HRF time-to-peak (sec)"])/10
df = df[df["Condition"] != "Control"]

df_kids = df[df['Age (months)'] == 60].copy()
df_kids['Age (months)'] = df_kids['Age (months)'] - 20
df = df[df['Age (months)'] != 60].copy()

fig, axes = plt.subplots(1, 1, figsize=(8, 5))
sns.lineplot(
    df, x="Age (months)", y="HRF time-to-peak (sec)", hue="Condition",
    style="Condition", markers=["o", "^", "s", "o", "^", "s"],
    palette=['indianred', 'indianred', 'indianred',
             'royalblue', 'royalblue', 'royalblue'],
    dashes=[(2, 0), (2, 2), (2, 4), (2, 0), (2, 2), (2, 4)],
    legend=True, errorbar='se'
)
sns.pointplot(
    df_kids, x="Age (months)", y="HRF time-to-peak (sec)", hue="Condition",
    palette=['indianred', 'indianred', 'indianred',
             'royalblue', 'royalblue', 'royalblue'],
    scale=0.8, markers=["o", "^", "s", "o", "^", "s"], native_scale=True,
    dodge=0.5, err_kws=dict(alpha=0.5), legend=False, errorbar='se'
)
plt.grid()
plt.xlim(0, 45)
plt.ylim(8, 16)
plt.ylabel("HRF time-to-peak (sec)", fontsize=12)
plt.xlabel("Age (months)", fontsize=12)
plt.xticks(np.arange(0, 45, 10))
labels = [item.get_text() for item in axes.get_xticklabels()]
labels[-1] = '3-5 y'
axes.set_xticklabels(labels)
plt.legend(bbox_to_anchor=(1.01, 0.5), loc=6)
plt.tight_layout()
plt.show()

df_ttp = df.copy()


# -----------------------------------------------------------------------------
# Magnitude
# -----------------------------------------------------------------------------
METRIC = 'HRF peak amplitude (μM)'

sub_df = df_mag.rename(
    columns={METRIC: 'mag', 'Age (months)': 'age', 'Condition': 'cond'}
)
sub_df['age'] = sub_df['age'].astype(str)
sub_df = sub_df[sub_df['cond'].str.contains('HbO')]

model = ols("mag ~ C(cond) + C(age) + C(cond):C(age)", data=sub_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


sub_df = df_mag.rename(
    columns={METRIC: 'mag', 'Age (months)': 'age', 'Condition': 'cond'}
)
sub_df['age'] = sub_df['age'].astype(str)
sub_df = sub_df[sub_df['cond'].str.contains('HbR')]

model = ols("mag ~ C(cond) + C(age) + C(cond):C(age)", data=sub_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# -----------------------------------------------------------------------------
# Time-to-peak stats
# -----------------------------------------------------------------------------
METRIC = 'HRF time-to-peak (sec)'

sub_df = df_ttp.rename(
    columns={METRIC: 'ttp', 'Age (months)': 'age', 'Condition': 'cond'}
)
sub_df['age'] = sub_df['age'].astype(str)
sub_df = sub_df[sub_df['cond'].str.contains('HbO')]

model = ols("ttp ~ C(cond) + C(age) + C(cond):C(age)", data=sub_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


sub_df = df_ttp.rename(
    columns={METRIC: 'ttp', 'Age (months)': 'age', 'Condition': 'cond'}
)
sub_df['age'] = sub_df['age'].astype(str)
sub_df = sub_df[sub_df['cond'].str.contains('HbR')]

model = ols("ttp ~ C(cond) + C(age) + C(cond):C(age)", data=sub_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# -----------------------------------------------------------------------------
# Correlation of magnitude with time-to-peak
# -----------------------------------------------------------------------------
METRIC_A = 'HRF peak amplitude (μM)'
METRIC_B = 'HRF time-to-peak (sec)'
conditions = ['Vocal (HbO)', 'Vocal (HbR)',
              'Non-vocal (HbO)', 'Non-vocal (HbR)']
p_values = []
for condition in conditions:
    sub_df_mag = df_mag[df_mag['Condition'] == condition]
    sub_df_ttp = df_ttp[df_ttp['Condition'] == condition]
    s, p = pearsonr(sub_df_mag[METRIC_A], sub_df_ttp[METRIC_B])
    p_values.append(p)
    print(f"Correlation coefficient statistic: {s}")
_, fdr_p_values = multitest.fdrcorrection(p_values, alpha=0.05)
print(f"p-values: {fdr_p_values}")
