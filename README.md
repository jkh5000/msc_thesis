# MSc Thesis Repository

This repository contains source code, together with input and output data, for my MSc thesis:
**“Modelling Climate and Demographic Impacts on Global Food Supply Networks.”**

---

## Repository Structure

### `simulation/`
This folder includes the three adaptation scenarios described in the thesis:

- **No adaptation:** `simulation_msc_final.py`  
- **Trade adaptation:** `simulation_msc_final_adapt_trade.py`  
- **Trade and production adaptation:** `simulation_msc_final_adapt_trade_prod.py`

---

### `post_processing/`
This folder contains the data used in the thesis, as well as the scripts to derive them.

#### **Production Output Data**

| Column | Description |
|:--------|:-------------|
| `base` | 2020 output (tonnes) |
| `ssp126`, `ssp585` | 2045–2050 average output (tonnes) under SSP1–2.6 and SSP5–8.5 |
| `ssp126_diff`, `ssp585_diff` | Change from 2020 to 2045–2050 (tonnes) |
| `ssp126_ratio`, `ssp585_ratio` | Change from 2020 to 2045–2050 (%) |

#### **Supply–Demand Data**

| Column | Description |
|:--------|:-------------|
| `food_req_cap` | 2020 food demand (kg/capita) |
| `food_gap_cap` | 2045–2050 supply–demand gap (kg/capita) |
| `food_gap_ratio` | 2045–2050 supply–demand gap (%) |
| `food_req_total` | 2020 total food demand (tonnes) |
| `food_gap_total` | Change in total food demand, 2020–2050 (tonnes) |
| `pop` | 2045–2050 population (capita) |
| `pop_ratio` | Share of total 2045–2050 population (%) |