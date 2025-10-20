import pandas as pd
import numpy as np
import os

# --- Helper functions ---
def safe_divide(x, y):
    """Performs safe division to avoid divide-by-zero or NaN issues."""
    if x == 0.0 and y == 0.0:
        return 1.0
    elif y == 0.0:
        return np.inf
    else:
        return x / y

def calc_ratio(x, y):
    """Calculates element-wise ratio using safe_divide."""
    return np.array([safe_divide(a, b) for a, b in zip(x, y)])

# --- Folder path ---
folder_path = "../simulation/output/adapt_none/"

# --- Read CSV files ---
o_base_df = pd.read_csv(os.path.join(folder_path, "ssp585/ssp585_output_base.csv"))
o_ssp126_df = pd.read_csv(os.path.join(folder_path, "ssp126/ssp126_output_change.csv"))
o_ssp585_df = pd.read_csv(os.path.join(folder_path, "ssp585/ssp585_output_change.csv"))

# --- 2045-2050 mean ---
o_base_mean_ = o_base_df.iloc[:, -6:].mean(axis=1)
o_ssp126_mean_ = o_ssp126_df.iloc[:, -6:].mean(axis=1)
o_ssp585_mean_ = o_ssp585_df.iloc[:, -6:].mean(axis=1)

# --- Derived calculations ---
o_base_mean = o_base_mean_
o_ssp126_mean = o_ssp126_mean_
o_ssp585_mean = o_ssp585_mean_

o_ssp126_diff = o_ssp126_mean - o_base_mean
o_ssp585_diff = o_ssp585_mean - o_base_mean

o_ssp126_ratio = (calc_ratio(o_ssp126_mean_, o_base_mean_) - 1.0) * 100.0
o_ssp585_ratio = (calc_ratio(o_ssp585_mean_, o_base_mean_) - 1.0) * 100.0

# --- Check for NaN or Inf values ---
if np.any(np.isnan(o_ssp126_ratio)) or np.any(np.isinf(o_ssp126_ratio)):
    raise ValueError("ssp126_ratio contains NaN or Inf values.")
if np.any(np.isnan(o_ssp585_ratio)) or np.any(np.isinf(o_ssp585_ratio)):
    raise ValueError("ssp585_ratio contains NaN or Inf values.")

# --- Create output DataFrame ---
output_df = pd.DataFrame({
    "area": o_base_df["area"],
    "code": o_base_df["code"],
    "region": o_base_df["region"],
    "item": o_base_df["item"],
    "base": o_base_mean,
    "ssp126": o_ssp126_mean,
    "ssp585": o_ssp585_mean,
    "ssp126_diff": o_ssp126_diff,
    "ssp585_diff": o_ssp585_diff,
    "ssp126_ratio": o_ssp126_ratio,
    "ssp585_ratio": o_ssp585_ratio
})

# --- Save as CSV ---
os.makedirs("output", exist_ok=True)
output_df.to_csv("output/prod_output_data.csv", index=False)