# BRIGHT social task fNIRS analysis

**MATLAB requirements:**
1. [Install MATLAB](https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html)
2. Download the [Homer2](https://www.nitrc.org/projects/homer2) toolbox.

**Python requirements:**
1. [Install Python 3 with Miniconda](https://repo.anaconda.com/miniconda/)
2. Install dependencies:
```python
pip install uv
uv pip install matplotlib numpy pandas scipy seaborn statsmodels PyQt6
```

**Scripts:**
1. `fnirs-social-task/src/signal/JB_soc_preprocessing.m` → social task fNIRS preprocessing
2. `fnirs-social-task/src/social/JB_soc_60mo.py` → detailed social task fNIRS analysis at the 3-5 years age point
3. `fnirs-social-task/src/social/JB_soc_cross_sectional.py` → social task fNIRS analysis at all the age points from 5 months to 3-5 years old
4. `fnirs-social-task/src/social/JB_soc_trajectories.py` → study of trajectories with fNIRS on the social task from 5 months to 3-5 years old
