### IMPORT ###

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs

import numpy as np


### FUNCTIONS ###

def calc_stock_ratio(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
    mask = (x == 0) & (y == 0)
    result[mask] = 1.
    result[np.isinf(result)] = 0.  # Replace Inf with 0 (means no stock requested)

    if np.any(np.isnan(result)):
        print("Warning: NaN values encountered in calc_stock_ratio")
    return result

def calc_export_ratio(x, y):
    # do the division (suppress warnings for 0/0 and div-by-zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
    # wherever both x and y are zero, force result to 1
    mask = (x == 0) & (y == 0)
    result[mask] = 1.
    result[result > 1.] = 1.

    if np.any(np.isnan(result)):
        print("Warning: NaN values encountered in calc_export_ratio")
    return result


### PARAMETERS ###

scenario = "ssp585"  # scenario name (ssp126, ssp585)

pop_flag = True      # Population changes on/off
crop_flag = True     # Crop yield changes on/off

input_folder = './input/'           # folder with parameters and input data
output_folder = f'./output/adapt_trade/{scenario}/'     # folder to write results to

start_yr = 2020
stop_yr = 2050
prerun = 100

tau = stop_yr - start_yr + prerun            # number of iterations


### LOADING DATA ###

# Load list of country
countries = pd.read_csv(input_folder+'country-index.csv',header=None,names=['c'],sep='\t')
countries = countries['c'].values

# Load list of items
items = pd.read_csv(input_folder+'item-index.csv', header=None, names=['i'],sep='\t')
items = items['i'].values

# Load list of processes
processes = pd.read_csv(input_folder+'process-index.csv',header=None, names=['p'],sep='\t')
processes = processes['p'].values

# Load information on countries
c_frame = pd.read_csv(input_folder+'country-information.csv',index_col=[2])
c_frame.drop(['Unnamed: 0'], inplace=True, axis=1)

# Build (country,item)-index
ci_index = pd.MultiIndex.from_product([countries,items])

Ni = len(items)         # number of items
Nc = len(countries)     # number of countries
Np = len(processes)     # number of processes

matrix_nu = io.mmread(input_folder+'/sparse_nu.mtx')
matrix_alpha = io.mmread(input_folder+'/sparse_alpha.mtx')
matrix_beta = io.mmread(input_folder+'/sparse_beta.mtx')
matrix_trade = io.mmread(input_folder+'/sparse_trade.mtx')

vec_stock = io.mmread(input_folder+'/sparse_stock.mtx')
vec_prod_input = io.mmread(input_folder+'/sparse_prod_input.mtx')
vec_prod_output = io.mmread(input_folder+'/sparse_prod_output.mtx')
vec_demand = io.mmread(input_folder+'/sparse_demand.mtx')
vec_import = io.mmread(input_folder+'/sparse_import.mtx')

if scenario == 'ssp126':
    yield_ratio_matrix = io.mmread(input_folder+'/sparse_yield_ratio_ssp126.mtx')
elif scenario == 'ssp585':
    yield_ratio_matrix = io.mmread(input_folder+'/sparse_yield_ratio_ssp585.mtx')

pop_ratio_matrix = io.mmread(input_folder+'/sparse_pop_ratio.mtx')


# Turn data into sparse csr-format
alpha = sprs.csr_matrix(matrix_alpha)                        # conversion from input to output
beta = sprs.csr_matrix(matrix_beta)                          # output for non-converting processes
T = sprs.csr_matrix(matrix_trade)                            # fraction sent to each trading partner
nu = sprs.csr_matrix(matrix_nu)                              # fraction allocated to a specific production process
stock_base = sprs.csr_matrix(vec_stock)                      # initial stock levels
prod_input_base = sprs.csr_matrix(vec_prod_input)            # production input levels
prod_output_base = sprs.csr_matrix(vec_prod_output)          # production output levels
demand_base = sprs.csr_matrix(vec_demand)                    # initial demand levels
import_base = sprs.csr_matrix(vec_import)                    # initial import levels
crop_ratio_matrix = sprs.csr_matrix(yield_ratio_matrix)      # yield ratios for each crop and country
pop_ratio_matrix = sprs.csr_matrix(pop_ratio_matrix)         # population ratios for each country

# eliminate zeros from sparse matrices
alpha.eliminate_zeros()
beta.eliminate_zeros()
T.eliminate_zeros()
nu.eliminate_zeros()
stock_base.eliminate_zeros()
prod_input_base.eliminate_zeros()
demand_base.eliminate_zeros()
import_base.eliminate_zeros()
crop_ratio_matrix.eliminate_zeros()
pop_ratio_matrix.eliminate_zeros()

one_vec = sprs.csr_matrix(np.ones(Nc*Np))
one_vec = one_vec.transpose()

# Calculate balancing vector
x_bal = stock_base - alpha @ (nu @ (prod_input_base)) - (beta @ one_vec) - T @ (stock_base - prod_input_base - demand_base)


### FIND CROP INDICES ###

crop_names = ['Maize and products', 'Rice (Milled Equivalent)', 'Soyabeans', 'Wheat and products']

# Create a vector for crop identification (all zeros by default)
is_crop = sprs.lil_matrix((Nc*Ni, 1))
is_maize = sprs.lil_matrix((Nc*Ni, 1))
is_rice = sprs.lil_matrix((Nc*Ni, 1))
is_soy = sprs.lil_matrix((Nc*Ni, 1))
is_wheat = sprs.lil_matrix((Nc*Ni, 1))

# iteration over all crops
for idx, crop_name in enumerate(crop_names):
    # iteration over all countries
    for country in countries:

        # find the index for the country and crop
        crop_index = ci_index.get_loc((country, crop_name))

        # set value of is_crop to 1 at the found index
        is_crop[crop_index, 0] = 1
        if crop_name == 'Maize and products':
            is_maize[crop_index, 0] = 1
        if crop_name == 'Rice (Milled Equivalent)':
            is_rice[crop_index, 0] = 1
        if crop_name == 'Soyabeans':
            is_soy[crop_index, 0] = 1
        if crop_name == 'Wheat and products':
            is_wheat[crop_index, 0] = 1

# Find indices where is_crop is 1
crop_indices = is_crop.nonzero()[0]


### INITIALIZATION ###

# Initialize stock
x = stock_base.copy()

# Initialize demand
demand_new = demand_base.copy()

# Create Dataframes to store data
Ochanged = pd.DataFrame(np.array(prod_output_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
Ochanged.index.names = ['area','item']
Obase = pd.DataFrame(np.array(prod_output_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
Obase.index.names = ['area','item']
demand_req_df = pd.DataFrame(np.array(demand_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
demand_req_df.index.names = ['area','item']
demand_act_df = pd.DataFrame(np.array(demand_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
demand_act_df.index.names = ['area','item']
import_act_df = pd.DataFrame(np.array(import_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
import_act_df.index.names = ['area','item']
import_req_df = pd.DataFrame(np.array(import_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
import_req_df.index.names = ['area','item']
stock_req_df = pd.DataFrame(np.array(prod_input_base.toarray()[:,0]) + np.array(demand_base.toarray()[:,0]),index=ci_index,columns=[f'{start_yr}'])
stock_req_df.index.names = ['area','item']


### SIMULATION ###

for t in range(0,tau):

    ## ALLOCATION BLOCK START ##

    if t >= prerun:
        # Multiply demand by population ratio
        if pop_flag:
            pop_ratio = pop_ratio_matrix[:,t-prerun]
            demand_new = demand_base.multiply(pop_ratio)
        else:
            demand_new = demand_base.copy()

    if t>0: x = x + x_bal #add balancing term
    stock_act = x.toarray()
    stock_req = prod_input_base.toarray() + demand_new.toarray()
    stock_ratio = calc_stock_ratio(stock_act, stock_req)
    
    # Create masks for where the condition is true vs false
    sufficient_supply = stock_act >= stock_req

    # Initialize arrays for the results
    prod_input_array = np.zeros_like(stock_act)
    demand_array = np.zeros_like(stock_act)

    # Where supply is sufficient, allocate as requested
    prod_input_array[sufficient_supply] = prod_input_base.toarray()[sufficient_supply]
    demand_array[sufficient_supply] = demand_new.toarray()[sufficient_supply]

    # Where supply is insufficient, split 50/50
    prod_input_array[~sufficient_supply] = prod_input_base.toarray()[~sufficient_supply] * stock_ratio[~sufficient_supply]
    demand_array[~sufficient_supply] = demand_new.toarray()[~sufficient_supply] * stock_ratio[~sufficient_supply]

    # Make sure values are non-negative
    prod_input_array[prod_input_array < 0.] = 0.
    demand_array[demand_array < 0.] = 0.

    # Convert back to sparse matrices
    prod_input = sprs.csr_matrix(prod_input_array)
    demand = sprs.csr_matrix(demand_array)

    ## ALLOCATION BLOCK END ##


    ## PRODUCTION BLOCK START ##

    output = alpha @ (nu @ (prod_input)) + (beta @ one_vec)

    if t >= prerun and crop_flag:
        output = output.tolil()
        # For those indices, replace values in output with values from prod_output_base
        for idx in crop_indices:
            crop_ratio = crop_ratio_matrix[idx, t-prerun]
            output[idx, 0] = prod_output_base[idx, 0] * crop_ratio
        output = output.tocsr()

    ## PRODUCTION BLOCK END ##


    ## TRADE BLOCK START ##

    trade_input = x - prod_input - demand
    trade_input.data[trade_input.data < 0.] = 0.

    import_max = T @ (trade_input)
    import_max.data[import_max.data < 0.] = 0.

    import_req = stock_req - stock_act
    import_req[import_req < 0.] = 0.
    import_req = np.maximum(import_req, import_base.toarray())
    import_agg = import_max.toarray().reshape(Nc, Ni).sum(axis=0)
    import_req_agg = import_req.reshape(Nc, Ni).sum(axis=0)
    export_ratio_agg = calc_export_ratio(import_agg, import_req_agg)
    export_ratio = np.tile(export_ratio_agg, Nc)
    
    import_act = np.zeros_like(import_req)
    import_act[export_ratio == 1.] = import_req[export_ratio == 1.]
    import_act[export_ratio < 1.] = (import_req.ravel()[export_ratio < 1.] * export_ratio[export_ratio < 1.]).reshape(-1, 1)
    imports = sprs.csr_matrix(import_act)

    ## TRADE BLOCK END ##


    ## UPDATE BLOCK START ##

    x = output + imports

    ## UPDATE BLOCK END ##

    # Store data
    if t >= prerun - 1:
        # Add results to the dataframe
        Ochanged[f'{start_yr+t+1-prerun}'] = np.array(output.toarray()[:,0])
        Obase[f'{start_yr+t+1-prerun}'] = np.array(prod_output_base.toarray()[:,0])
        demand_req_df[f'{start_yr+t+1-prerun}'] = np.array(demand_new.toarray()[:,0])
        demand_act_df[f'{start_yr+t+1-prerun}'] = np.array(demand.toarray()[:,0])
        import_act_df[f'{start_yr+t+1-prerun}'] = np.array(import_act)
        import_req_df[f'{start_yr+t+1-prerun}'] = np.array(import_req)
        stock_req_df[f'{start_yr+t+1-prerun}'] = np.array(stock_req)

# Output
print('Simulation done.')


### SAVE RESULTS ###

# Merge region information into each dataframe
Ochanged_region = (
    Ochanged
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area','code','item', 'region'])
)
Obase_region = (
    Obase
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area','code','item', 'region'])
)
demand_req_df_region = (
    demand_req_df
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area', 'code', 'item', 'region'])
)
demand_act_df_region = (
    demand_act_df
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area', 'code', 'item', 'region'])
)
import_act_df_region = (
    import_act_df
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area', 'code', 'item', 'region'])
)
import_req_df_region = (
    import_req_df
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area', 'code', 'item', 'region'])
)
stock_req_df_region = (
    stock_req_df
    .reset_index()
    .merge(c_frame[['region']], how='left', left_on='area', right_index=True)
    .merge(c_frame[['code']], how='left', left_on='area', right_index=True)
    .set_index(['area', 'code', 'item', 'region'])
)

# Save to CSV
Ochanged_region.to_csv(output_folder+f'{scenario}_output_change.csv')
Obase_region.to_csv(output_folder+f'{scenario}_output_base.csv')
demand_req_df_region.to_csv(output_folder+f'{scenario}_demand_req.csv')
demand_act_df_region.to_csv(output_folder+f'{scenario}_demand_act.csv')
import_act_df_region.to_csv(output_folder+f'{scenario}_import_act.csv')
import_req_df_region.to_csv(output_folder+f'{scenario}_import_req.csv')
stock_req_df_region.to_csv(output_folder+f'{scenario}_stock_req.csv')