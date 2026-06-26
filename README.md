# MC-testing-by-betting (Python Version)

This repository contains the **Python Jupyter Notebook** implementations of the experiments from the paper "Sequential Monte-Carlo testing by betting" (and its follow-up works on Multiple Testing).

The original codebase was written in R. This repository provides a clean, self-contained port of the algorithms, statistical simulations, and multiple-testing frameworks into Python, using `numpy`, `pandas`, `scipy`, and `statsmodels`.

## Contents

- **`Power_nperm_generate_data.ipynb`**: Simulates statistical power and average permutation requirements across various betting strategies.
- **`Real_data_CRT.ipynb`**: Conditional Randomization Test (CRT) using the 2011 Capital Bikeshare dataset (the dataset is automatically downloaded and processed within the notebook).
- **`Real_data_Fisher_sharp.ipynb`**: Evaluation of the Fisher Sharp null hypothesis.
- **`Appendix_simulations.ipynb`**: Supplemental simulations for varying distributional assumptions and P-value empirical CDF comparisons.

### Multiple Testing Extension (`Multiple testing/`)
- `sequential_BH.py`: Core Python implementation of the sequential Benjamini-Hochberg (BH) procedure algorithms (`bm`, `bc`, `AMT`).
- `Multiple_testing_data_generator.ipynb`: Large-scale data generation for evaluating False Discovery Rate (FDR) and Power under multiple testing scenarios.
- `Multiple_testing_sequential_BH.ipynb`: Plotting pipelines and fMRI real-data analysis.

## Requirements

To run these notebooks, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Generated Figures (LaTeX Ready)

All generated plots have been extracted in **vector format (PDF)** to prevent quality loss when embedding into a LaTeX report. You can find them inside the `images/` directory:

- `Figure1_wealth_lower.pdf`: Wealth lower bounds
- `Figure2_wealth_upper.pdf`: Wealth upper bounds
- `Figure3_power_alpha005.pdf`: Statistical power (alpha = 0.05)
- `Figure4_power_alpha001.pdf`: Statistical power (alpha = 0.01)
- `Figure5_power_randomized.pdf`: Power comparison (Randomized Binomial)
- `Figure6_power_randomized_comb.pdf`: Power comparison (Combined)
- `Figure7_worst_bounds.pdf`: Worst-case bound analysis
- `Figure8_power_pis.pdf`: Power distribution under various pi values
- `Figure9_power_alphas.pdf`: Power distribution under various alpha values
- `Figure10_power_mus.pdf`: Power distribution under various mu values
- `Figure11_power_Ms.pdf`: Power distribution by hypotheses count (M)
- `Figure12_power_rhos.pdf`: FDR distribution under varying PRDS rhos
- `Figure13_power_all.pdf`: Combined grid of all multiple testing power plots
- `Figure14_nperm_dist.pdf`: Distribution of number of permutations until rejection
