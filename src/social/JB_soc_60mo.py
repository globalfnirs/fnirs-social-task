"""
Script for 60 mo social task analysis
Author: Johann Benerradi
------------------------
"""

import functions.soc as soc
import matplotlib.pyplot as plt


CONDS = ['S', 'V', 'N']

plt.switch_backend('QtAgg')


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
grand_avg, subj_ids, rejected, _, _ = soc.load_60mo(
    '../../../data/dataset_bright/soc/results/60mo/', ['S', 'V', 'N']
)
feature_grand_avg = soc.window_average(grand_avg)


# -----------------------------------------------------------------------------
# Visualise HRFs
# -----------------------------------------------------------------------------
# Visual social silent
print("=========================")
print("Visual social (S)")
print("-------------------------")
condition = 0  # S
p_values, t_values, trends, activations = soc.analyse_act(feature_grand_avg,
                                                          condition, fdr=True)
soc.plot_hrf(grand_avg, condition, activations)
soc.topo_overlay(activations)
# plt.show()

# Auditory vocal
print("=========================")
print("Auditory vocal (V)")
print("-------------------------")
condition = 1  # V
p_values, t_values, trends, activations = soc.analyse_act(feature_grand_avg,
                                                          condition, fdr=True)
soc.plot_hrf(grand_avg, condition, activations)
soc.topo_overlay(activations)

# Auditory non-vocal
print("=========================")
print("Auditory non-vocal (N)")
print("-------------------------")
condition = 2  # N
p_values, t_values, trends, activations = soc.analyse_act(feature_grand_avg,
                                                          condition, fdr=True)
soc.plot_hrf(grand_avg, condition, activations)
soc.topo_overlay(activations)
