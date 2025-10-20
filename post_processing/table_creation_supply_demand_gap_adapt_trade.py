import os
import numpy as np
import pandas as pd

# --- Helper functions ---
def safe_divide_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Elementwise 'safe divide':
      - if x == 0 and y == 0  -> 1
      - if isinf(x) or isinf(y) -> 1
      - elif y == 0           -> +inf
      - else                  -> x / y
    """
    a = a.astype(float, copy=False)
    b = b.astype(float, copy=False)
    out = np.empty_like(a, dtype=float)

    both_zero = (a == 0) & (b == 0)
    any_inf   = np.isinf(a) | np.isinf(b)
    denom_zero = (b == 0)

    # default case
    out = np.divide(a, b, out=np.full_like(a, np.nan), where=~denom_zero)

    # apply special rules
    out[both_zero] = 1.0
    out[any_inf] = 1.0
    out[denom_zero & ~both_zero & ~any_inf] = np.inf
    return out

def calc_mean_df(df: pd.DataFrame) -> pd.Series:
    """Row-wise mean."""
    return df.mean(axis=1)

def safe_divide_mean(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    """Elementwise safe divide, then row-wise mean."""
    arr = safe_divide_array(df1.to_numpy(), df2.to_numpy())
    return pd.Series(arr.mean(axis=1), index=df1.index)

# ---  Paths ---
folder_path = "../simulation/output/adapt_trade/"

# --- Read inputs ---
ssp126_import_act_df = pd.read_csv(os.path.join(folder_path, "ssp126/ssp126_import_act.csv"))
ssp585_import_act_df = pd.read_csv(os.path.join(folder_path, "ssp585/ssp585_import_act.csv"))

ssp126_import_req_df = pd.read_csv(os.path.join(folder_path, "ssp126/ssp126_import_req.csv"))
ssp585_import_req_df = pd.read_csv(os.path.join(folder_path, "ssp585/ssp585_import_req.csv"))

ssp126_stock_req_df  = pd.read_csv(os.path.join(folder_path, "ssp126/ssp126_stock_req.csv"))
ssp585_stock_req_df  = pd.read_csv(os.path.join(folder_path, "ssp585/ssp585_stock_req.csv"))

ssp126_demand_req_df = pd.read_csv(os.path.join(folder_path, "ssp126/ssp126_demand_req.csv"))
ssp585_demand_req_df = pd.read_csv(os.path.join(folder_path, "ssp585/ssp585_demand_req.csv"))


# --- Population data ---
pop_df = pd.read_csv("input/WPP_population.csv")

# 2020 population (thousands -> persons)
pop_2020 = pop_df["2020"] * 1e3
pop_dict = dict(zip(pop_df["ISO3"], pop_2020))

# 2045–2050 mean population
pop_mean = pop_df.iloc[:, 98:104].mean(axis=1) * 1e3
pop_mean_dict = dict(zip(pop_df["ISO3"], pop_mean))

pop_mean_ratio = pop_mean / pop_mean.sum()
pop_mean_ratio_dict = dict(zip(pop_df["ISO3"], pop_mean_ratio))

# Create column with population data
n_items = ssp585_import_act_df["item"].nunique()
country_codes = ssp585_import_act_df["code"].drop_duplicates().tolist()

pop_2020_mapped       = [pop_dict.get(c, 0.0)           for c in country_codes]
pop_mean_mapped       = [pop_mean_dict.get(c, 0.0)      for c in country_codes]
pop_mean_ratio_mapped = [pop_mean_ratio_dict.get(c, 0.0) for c in country_codes]

pop_2020_repeated       = np.repeat(pop_2020_mapped, n_items)
pop_mean_repeated       = np.repeat(pop_mean_mapped, n_items)
pop_mean_ratio_repeated = np.repeat(pop_mean_ratio_mapped, n_items)

pop_cols = pop_df.columns[98:104]
pop_sel  = pop_df[["ISO3"] + pop_cols.tolist()]

result = ssp585_import_act_df.iloc[:, 0:3].merge(pop_sel, left_on="code", right_on="ISO3", how="left")
result = result.sort_values(["area", "item"])
pop_4550_df = result.iloc[:, -6:].fillna(0.0) * 1e3

# --- Food fraction data ---
path_ = "input/FABIO_country_info_2020.csv"
fabio_df = pd.read_csv(path_)

food_ratio_vec = safe_divide_array(
    fabio_df["food_frac"].to_numpy(),
    (fabio_df["food_frac"] + fabio_df["other_frac"]).to_numpy()
)

# --- Extract 2045-2050 windows ---
ssp126_demand_req = ssp126_demand_req_df.iloc[:, -6:]
ssp585_demand_req = ssp585_demand_req_df.iloc[:, -6:]

ssp126_import_act = ssp126_import_act_df.iloc[:, -6:]
ssp585_import_act = ssp585_import_act_df.iloc[:, -6:]

ssp126_import_req = ssp126_import_req_df.iloc[:, -6:]
ssp585_import_req = ssp585_import_req_df.iloc[:, -6:]

ssp126_stock_req  = ssp126_stock_req_df.iloc[:, -6:]
ssp585_stock_req  = ssp585_stock_req_df.iloc[:, -6:]

ssp126_stock_lack = ssp126_import_req - ssp126_import_act
ssp585_stock_lack = ssp585_import_req - ssp585_import_act

ssp126_stock_act = (ssp126_stock_req - ssp126_stock_lack).clip(lower=0)
ssp585_stock_act = (ssp585_stock_req - ssp585_stock_lack).clip(lower=0)

# --- Demand ratio ---
demand_ratio = safe_divide_array(
    ssp585_demand_req.to_numpy(),
    ssp585_stock_req.to_numpy()
)

# --- Demand actual ---
ssp126_demand_act = pd.DataFrame(ssp126_stock_act.to_numpy() * demand_ratio,
                                 columns=ssp126_stock_act.columns, index=ssp126_stock_act.index)
ssp585_demand_act = pd.DataFrame(ssp585_stock_act.to_numpy() * demand_ratio,
                                 columns=ssp585_stock_act.columns, index=ssp585_stock_act.index)

# Means across the 6 years
ssp126_import_act_mean = calc_mean_df(ssp126_import_act)
ssp585_import_act_mean = calc_mean_df(ssp585_import_act)

ssp126_import_req_mean = calc_mean_df(ssp126_import_req)
ssp585_import_req_mean = calc_mean_df(ssp585_import_req)

ssp126_stock_req_mean  = calc_mean_df(ssp126_stock_req)
ssp585_stock_req_mean  = calc_mean_df(ssp585_stock_req)

ssp126_stock_lack_mean = ssp126_import_req_mean - ssp126_import_act_mean
ssp585_stock_lack_mean = ssp585_import_req_mean - ssp585_import_act_mean

ssp126_stock_act_mean  = ssp126_stock_req_mean - ssp126_stock_lack_mean
ssp585_stock_act_mean  = ssp585_stock_req_mean - ssp585_stock_lack_mean

ssp126_demand_req_mean = calc_mean_df(ssp126_demand_req)
ssp585_demand_req_mean = calc_mean_df(ssp585_demand_req)

# Food demand (apply country ratios row-wise)
ssp126_food_demand_req = ssp126_demand_req.mul(food_ratio_vec, axis=0)
ssp585_food_demand_req = ssp585_demand_req.mul(food_ratio_vec, axis=0)
ssp126_food_demand_act = ssp126_demand_act.mul(food_ratio_vec, axis=0)
ssp585_food_demand_act = ssp585_demand_act.mul(food_ratio_vec, axis=0)

ssp126_food_demand_ratio = (safe_divide_mean(ssp126_food_demand_act, ssp126_food_demand_req) - 1.0) * 100.0
ssp585_food_demand_ratio = (safe_divide_mean(ssp585_food_demand_act, ssp585_food_demand_req) - 1.0) * 100.0

ssp126_food_demand_req_mean = calc_mean_df(ssp126_food_demand_req)
ssp585_food_demand_req_mean = calc_mean_df(ssp585_food_demand_req)
ssp126_food_demand_act_mean = calc_mean_df(ssp126_food_demand_act)
ssp585_food_demand_act_mean = calc_mean_df(ssp585_food_demand_act)

ssp126_food_demand_diff_mean = ssp126_food_demand_act_mean - ssp126_food_demand_req_mean
ssp585_food_demand_diff_mean = ssp585_food_demand_act_mean - ssp585_food_demand_req_mean

# Per-capita (×1000 to convert tonne -> kg)
ssp126_demand_req_mean_cap = safe_divide_mean(ssp126_demand_req, pop_4550_df) * 1e3
ssp585_demand_req_mean_cap = safe_divide_mean(ssp585_demand_req, pop_4550_df) * 1e3
ssp126_demand_act_mean_cap = safe_divide_mean(ssp126_demand_act, pop_4550_df) * 1e3
ssp585_demand_act_mean_cap = safe_divide_mean(ssp585_demand_act, pop_4550_df) * 1e3
ssp126_demand_diff_mean_cap = safe_divide_mean(ssp126_demand_act - ssp126_demand_req, pop_4550_df) * 1e3
ssp585_demand_diff_mean_cap = safe_divide_mean(ssp585_demand_act - ssp585_demand_req, pop_4550_df) * 1e3

ssp126_food_demand_req_mean_cap = ssp126_demand_req_mean_cap * food_ratio_vec
ssp585_food_demand_req_mean_cap = ssp585_demand_req_mean_cap * food_ratio_vec
ssp126_food_demand_act_mean_cap = ssp126_demand_act_mean_cap * food_ratio_vec
ssp585_food_demand_act_mean_cap = ssp585_demand_act_mean_cap * food_ratio_vec
ssp126_food_demand_diff_mean_cap = ssp126_food_demand_act_mean_cap - ssp126_food_demand_req_mean_cap
ssp585_food_demand_diff_mean_cap = ssp585_food_demand_act_mean_cap - ssp585_food_demand_req_mean_cap

# Absolute difference (tonnes) between 2020 and 2045–50
ssp126_2020_demand_req = ssp126_demand_req_df.iloc[:, -31]
ssp585_2020_demand_req = ssp585_demand_req_df.iloc[:, -31]

ssp126_demand_req_diff = ssp126_demand_req_mean - ssp126_2020_demand_req
ssp585_demand_req_diff = ssp585_demand_req_mean - ssp585_2020_demand_req

ssp126_food_demand_req_diff = ssp126_demand_req_diff * food_ratio_vec
ssp585_food_demand_req_diff = ssp585_demand_req_diff * food_ratio_vec


# --- Create output DataFrame ---
ssp126_summary_df = pd.DataFrame({
    "area":  ssp585_import_act_df["area"],
    "code":  ssp585_import_act_df["code"],
    "region":ssp585_import_act_df["region"],
    "item":  ssp585_import_act_df["item"],
    "food_req_cap":     ssp126_food_demand_req_mean_cap,
    "food_gap_cap":     ssp126_food_demand_diff_mean_cap,
    "food_gap_ratio":   ssp126_food_demand_ratio,
    "food_req_total":   ssp126_demand_req_df.iloc[:, -31] * food_ratio_vec,
    "food_gap_total":   ssp126_food_demand_req_diff,
    "pop":              pop_mean_repeated,
    "pop_ratio":        pop_mean_ratio_repeated
})

ssp585_summary_df = pd.DataFrame({
    "area":  ssp585_import_act_df["area"],
    "code":  ssp585_import_act_df["code"],
    "region":ssp585_import_act_df["region"],
    "item":  ssp585_import_act_df["item"],
    "food_req_cap":     ssp585_food_demand_req_mean_cap,
    "food_gap_cap":     ssp585_food_demand_diff_mean_cap,
    "food_gap_ratio":   ssp585_food_demand_ratio,
    "food_req_total":   ssp585_demand_req_df.iloc[:, -31] * food_ratio_vec,
    "food_gap_total":   ssp585_food_demand_req_diff,
    "pop":              pop_mean_repeated,
    "pop_ratio":        pop_mean_ratio_repeated
})

# --- Save as CSV ---
os.makedirs("output", exist_ok=True)
ssp126_summary_df.to_csv("output/ssp126_summary_data_adapt_trade.csv", index=False)
ssp585_summary_df.to_csv("output/ssp585_summary_data_adapt_trade.csv", index=False)