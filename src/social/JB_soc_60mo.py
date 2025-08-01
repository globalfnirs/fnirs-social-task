"""
Script for 60 mo social task analysis
Author: Johann Benerradi
------------------------
"""

# %% Imports
import functions.soc as soc

# %% Load data
grand_avg, _, _ = soc.load_results_60mo('[path to dataset]/results/60mo/', ['S', 'V', 'N'])
feature_grand_avg = soc.window_average(grand_avg)


# %% Visualise HRFs
# Visual social silent
condition = 0  # S
p_values, t_values, trends, activations = soc.analyse_act(feature_grand_avg, condition, fdr=True)
soc.plot_hrf(grand_avg, condition, activations)
soc.topo_overlay(activations)

# Auditory vocal
condition = 1  # V
p_values, t_values, trends, activations = soc.analyse_act(feature_grand_avg, condition, fdr=True)
soc.plot_hrf(grand_avg, condition, activations)
soc.topo_overlay(activations)

# Auditory non-vocal
condition = 2  # N
p_values, t_values, trends, activations = soc.analyse_act(feature_grand_avg, condition, fdr=True)
soc.plot_hrf(grand_avg, condition, activations)
soc.topo_overlay(activations)
