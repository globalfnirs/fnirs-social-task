"""
Functions for figure plotting
Author: Johann Benerradi
------------------------
"""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats


def violin_hemis(df, roi_list, subj_list, cond, type, ylim,
                 feature="Window average"):
    sub_df = df[df['Condition'] == cond]
    sub_df = sub_df[sub_df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    sub_df[['Hemisphere', 'Location']] = sub_df['ROI'].str.split(' ',
                                                                 expand=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 30))
    for i_roi, roi in enumerate(roi_list[::2]):
        df_roi = sub_df[sub_df['Location'] == roi.split(' ')[-1]]
        axes.flat[i_roi].title.set_text(roi.split(' ')[-1])
        sns.violinplot(data=df_roi, x="Age (months)", y=feature,
                       hue='Hemisphere',
                       dodge=True, alpha=.45, split=True,
                       order=['1', '5', '8', '12', '18', '24', '60'],
                       hue_order=['left', 'right'], ax=axes.flat[i_roi])
        axes.flat[i_roi].legend(bbox_to_anchor=(1.01, 0.5), loc=6)
        axes.flat[i_roi].set_ylim(*ylim)
        axes.flat[i_roi].grid()
    plt.tight_layout(pad=3)
    plt.show()


def paired_scatter_hemis(df, roi_list, ages, subj_list, cond, type, ylim,
                         feature="Window average"):
    sub_df = df[df['Condition'] == cond]
    sub_df = sub_df[sub_df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    sub_df[['Hemisphere', 'Location']] = sub_df['ROI'].str.split(' ',
                                                                 expand=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 30))
    for i_roi, roi in enumerate(roi_list[::2]):
        df_roi = sub_df[sub_df['Location'] == roi.split(' ')[-1]]
        print(roi.split(' ')[-1])
        axes.flat[i_roi].title.set_text(roi.split(' ')[-1])

        # Plot points
        sns.stripplot(data=df_roi, x="Age (months)", y=feature,
                      hue="Hemisphere", dodge=True, alpha=.45, legend=False,
                      order=['1', '5', '8', '12', '18', '24', '60'],
                      hue_order=['left', 'right'], ax=axes.flat[i_roi])

        for i_age, age in enumerate(ages):
            df_l = df_roi[(df_roi['Age (months)'] == age[:-2])
                          & (df_roi['Hemisphere'] == 'left')]
            df_r = df_roi[(df_roi['Age (months)'] == age[:-2])
                          & (df_roi['Hemisphere'] == 'right')]

            # Stats
            right, left = [], []
            for subj in subj_list:
                right.append(df_r[df_r['ID'] == subj][feature].values)
                left.append(df_l[(df_l['ID'] == subj)][feature].values)
            try:
                pval = stats.ttest_rel(right, left, nan_policy='omit').pvalue
                print(age, pval, len(right), len(left), end='|')
            except ValueError:
                pass

            # Plot links
            df_l, df_r = df_l.dropna(), df_r.dropna()
            if len(df_l) != len(df_r):
                continue

            locs1 = axes.flat[i_roi].get_children()[i_age*2].get_offsets()
            locs2 = axes.flat[i_roi].get_children()[i_age*2+1].get_offsets()

            if not list(locs1[:, 1]) == list(df_l[feature]):
                raise Exception('DataFrame and figure not matching')
            if not list(locs2[:, 1]) == list(df_r[feature]):
                raise Exception('DataFrame and figure not matching')

            for i in range(locs1.shape[0]):
                x, y = [locs1[i, 0], locs2[i, 0]], [locs1[i, 1], locs2[i, 1]]
                axes.flat[i_roi].plot(x, y, color="black", alpha=0.25)
        print()

        # Plot average marks
        sns.pointplot(data=df_roi, x="Age (months)", y=feature,
                      hue="Hemisphere", dodge=.5 - .5 / 3, palette="dark",
                      errorbar=None, markers="_", markersize=10,
                      linestyle="none",
                      order=['1', '5', '8', '12', '18', '24', '60'],
                      hue_order=['left', 'right'], ax=axes.flat[i_roi])
        axes.flat[i_roi].legend(bbox_to_anchor=(1.01, 0.5), loc=6)
        axes.flat[i_roi].set_ylim(*ylim)
        axes.flat[i_roi].grid()
    plt.tight_layout(pad=3)
    plt.show()


def paired_scatter_contrast(df, roi_list, ages, subj_list, type, ylim, feature="Window average"):
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
            legend_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=5, label=label)
                              for color, label in zip(['green', 'purple'], ['Auditory non-social', 'Auditory social'])]
            axes.flat[i_roi].legend(handles=legend_handles, title='Condition', loc='upper left')
        else:
            axes.flat[i_roi].legend().set_visible(False)
        axes.flat[i_roi].set_ylabel("Window average (µM)")
        axes.flat[i_roi].set_ylim(*ylim)
        axes.flat[i_roi].grid()
        axes.flat[i_roi].set_axisbelow(True)
        axes.flat[i_roi].grid(which='major', visible=True, color='silver', linewidth=1.)
        axes.flat[i_roi].grid(which='minor', visible=True, color='silver', linewidth=0.5, linestyle='--')
        axes.flat[i_roi].minorticks_on()
    plt.tight_layout(pad=3)
    plt.show()


def trajectories(df, roi_list, subj_list, cond, type, ylim, feature="Window average"):
    if type == 'hbo':
        sign = 1
    else:
        sign = -1
    sub_df = df[df['Condition'] == cond]
    sub_df = sub_df[sub_df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    for i_roi, roi in enumerate(roi_list):
        df_roi = sub_df[sub_df['ROI'] == roi]
        df_roi['Age (months)'] = df_roi['Age (months)'].astype(float)
        axes.flat[i_roi].title.set_text(roi)

        n_1mo = 0
        n_5mo = 0
        n_8mo = 0
        n_late = 0
        for subj in subj_list:
            subj_df = df_roi[df_roi['ID'] == subj].copy().sort_values(
                by=['Age (months)'])
            # slope = stats.linregress(subj_df['Age (months)'].values,
            #                          subj_df[feature].values).slope
            val_1mo = subj_df[subj_df['Age (months)'] == 1][feature].values
            val_5mo = subj_df[subj_df['Age (months)'] == 5][feature].values
            val_8mo = subj_df[subj_df['Age (months)'] == 8][feature].values
            if sign*val_1mo > 0:
                start = '1mo_spec'
                n_1mo += 1
            elif sign*val_5mo > 0:
                start = '5mo_spec'
                n_5mo += 1
            elif sign*val_8mo > 0:
                start = '8mo_spec'
                n_8mo += 1
            else:
                start = 'late_spec'
                n_late += 1

            df_roi.loc[df_roi['ID'] == subj, 'Start'] = start
        df_roi.replace('1mo_spec', f'1mo_spec ({n_1mo})', inplace=True)
        df_roi.replace('5mo_spec', f'5mo_spec ({n_5mo})', inplace=True)
        df_roi.replace('8mo_spec', f'8mo_spec ({n_8mo})', inplace=True)
        df_roi.replace('late_spec', f'late_spec ({n_late})', inplace=True)

        # Plot points
        sns.lineplot(data=df_roi, x="Age (months)", y=feature,
                     hue="Start",  # units='ID', estimator=None,
                     ax=axes.flat[i_roi], errorbar='se',
                     hue_order=[f'1mo_spec ({n_1mo})', f'5mo_spec ({n_5mo})',
                                f'8mo_spec ({n_8mo})', f'late_spec ({n_late})'])
        axes.flat[i_roi].legend(bbox_to_anchor=(1.01, 0.5), loc=6)
        axes.flat[i_roi].set_ylim(*ylim)
        axes.flat[i_roi].grid()
    plt.tight_layout(pad=3)
    plt.show()


def selective_trajectories(df, roi_list, subj_list, cond, type, ylim,
                           feature="Window average"):
    if type == 'hbo':
        sign = 1
    else:
        sign = -1
    sub_df = df[df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    spec_map = {}
    n_5mo = 0
    n_8mo = 0
    n_12mo = 0
    n_late = 0
    sub_v = sub_df[sub_df['Condition'] == 'V']
    sub_df = sub_df[sub_df['Condition'] == cond]
    for subj in subj_list:
        d = sub_df.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_v = sub_v.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        if sign*d.query("`Age (months)` == '5'")[feature].mean() > 0 and sign*d_v.query("`Age (months)` == '5'")[feature].mean() > 0:
            spec_map[subj] = '5mo_spec'
            n_5mo += 1
        elif sign*d.query("`Age (months)` == '8'")[feature].mean() > 0 and sign*d_v.query("`Age (months)` == '8'")[feature].mean() > 0:
            spec_map[subj] = '8mo_spec'
            n_8mo += 1
        elif sign*d.query("`Age (months)` == '12'")[feature].mean() > 0 and sign*d_v.query("`Age (months)` == '12'")[feature].mean() > 0:
            spec_map[subj] = '12mo_spec'
            n_12mo += 1
        else:
            spec_map[subj] = 'late_spec'
            n_late += 1
    sub_df['Specialisation'] = sub_df['ID'].map(spec_map)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for i_roi, roi in enumerate(roi_list[2:4]):
        df_roi = sub_df[sub_df['ROI'] == roi]
        df_roi['Age (months)'] = df_roi['Age (months)'].astype(float)
        df_roi = df_roi[df_roi['Age (months)'] <= 60]
        axes.flat[i_roi].title.set_text(roi.capitalize())

        df_roi.replace('5mo_spec', f'5 mo ({n_5mo})', inplace=True)
        df_roi.replace('8mo_spec', f'8 mo ({n_8mo})', inplace=True)
        df_roi.replace('12mo_spec', f'12 mo ({n_12mo})', inplace=True)
        df_roi.replace('late_spec', f'Remaining ({n_late})', inplace=True)

        # Plot points
        sns.lineplot(data=df_roi, x="Age (months)", y=feature,
                     hue="Specialisation",  # units='ID', estimator=None,
                     ax=axes.flat[i_roi], errorbar='se',
                     hue_order=[f'5 mo ({n_5mo})', f'8 mo ({n_8mo})',
                                f'12 mo ({n_12mo})', f'Remaining ({n_late})'])
        axes.flat[i_roi].legend(loc='lower right')
        axes.flat[i_roi].set_ylabel("Window average difference (µM)")
        axes.flat[i_roi].set_ylim(*ylim)
        axes.flat[i_roi].set_xlim(5, 60)
        axes.flat[i_roi].grid()
        axes.flat[i_roi].set_axisbelow(True)
        axes.flat[i_roi].grid(which='major', visible=True, color='silver', linewidth=1.)
        axes.flat[i_roi].grid(which='minor', visible=True, color='silver', linewidth=0.5, linestyle='--')
        axes.flat[i_roi].minorticks_on()
    plt.tight_layout(pad=3)
    plt.show()


def selective_sustained(df, subj_list, cond, type, feature="Window average"):
    if type == 'hbo':
        sign = 1
    else:
        sign = -1
    sub_df = df[df['Channel type'] == type]
    sub_df = sub_df[sub_df['ID'].isin(subj_list)]
    sub_v = sub_df[sub_df['Condition'] == 'V']
    sub_n = sub_df[sub_df['Condition'] == 'N']
    sub_df = sub_df[sub_df['Condition'] == cond]
    fw = open('../../outputs/selective_sustained.csv', 'w')
    fw.write("ID,5mo,8mo,12mo,18mo,24mo,60mo\n")
    for subj in subj_list:
        fw.write(f"{subj}")
        d = sub_df.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_v = sub_v.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        d_n = sub_n.query(f"ID == '{subj}' & ROI.str.contains('anterior')")
        for age in [5, 8, 12, 18, 24, 60]:
            fw.write(",")
            print(d.query(f"`Age (months)` == '{age}'")[feature].mean())
            if sign*d.query(f"`Age (months)` == '{age}'")[feature].mean() > 0 and sign*d_v.query(f"`Age (months)` == '{age}'")[feature].mean() > 0:
                fw.write("V")
            elif sign*d.query(f"`Age (months)` == '{age}'")[feature].mean() < 0 and sign*d_n.query(f"`Age (months)` == '{age}'")[feature].mean() > 0:
                fw.write("N")
            elif np.isnan(d.query(f"`Age (months)` == '{age}'")[feature].mean()):
                fw.write("Missing")
            else:
                fw.write("None")
        fw.write("\n")
