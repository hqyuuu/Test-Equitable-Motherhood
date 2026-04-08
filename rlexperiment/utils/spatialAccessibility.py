# @author: Shiqi Wang qtec@outlook.com
# Function: calculate accessibility function
import pandas as pd
import numpy as np
import time

def cal_time_logger(label, input_time, cal_time_log,print_out = False):
    temp_time = time.time()
    cal_time_log.append((label, temp_time - input_time))
    if print_out:
        print(f'time of {label} is {temp_time - input_time}')
    return temp_time, cal_time_log

def test_import():
   print('you have imported spatialAccessibility.py')
def calculate_accessibility_np(od_matrix, origin_df, dest_df, AccModel='Gravity', beta=1, Threshold=5000, Expon=0.8, set_ddof=1, print_out=True, time_recorder=False):
    """
    Calculate Accessibility Function

    Parameters:
    od_matrix (numpy.ndarray): A 991x991 matrix representing the travel cost between demand and supply points.
    origin_df (DataFrame): A dataset of demand points, containing OriginID and O_Demand information.
    dest_df (DataFrame): A dataset of supply points, containing DestinationID and D_Supply information.
    AccModel (str): Accessibility model, default is 'Gravity'.
    beta (float): Gravity model parameter, default is 1.
    Threshold (int): Threshold for the 2SFCA model, default is 5000.
    Expon (float): Exponential model parameter, default is 0.8.
    print_out (bool): Whether to print output, default is True.

    Returns:
    CurrentAcc (DataFrame): Current accessibility results.
    summary_Acc (DataFrame): Descriptive statistics of accessibility results.
    """

    # record the start time
    cal_start_time = time.time()
    cal_time_log = []

    # get the demand and supply data
    OriginPopu = origin_df['O_Demand'].values
    DestinationSupply = dest_df['D_Supply'].values

    temp_time, cal_time_log = cal_time_logger("get the demand and supply data", cal_start_time, cal_time_log)

    # calculate the total supply and demand
    TotalSupply = DestinationSupply.sum()
    demandpt = len(origin_df)
    demand_popu = OriginPopu.sum()
    supplypt = len(dest_df)
    ave_accessibility = TotalSupply / demand_popu

    if print_out:
        print(f'{"Demand Locations":<20} {demandpt:<10} {"with total population of":<40}{demand_popu:<10}')
        print(f'{"Facilities":<20} {supplypt:<10} {"with total capacity of ":<40}{round(TotalSupply, 3):<10}')
        print(f'{"Average Accessibility Score":<40} {ave_accessibility:<10}')

    if AccModel == '2SFCA':
        fdij = (od_matrix <= Threshold).astype(int)
    elif AccModel == 'Gravity':
        if print_out:
            print(f'{"Gravity method applied, with beta =":<40}{beta:<10}')
        fdij = od_matrix ** (-1 * beta)
    elif AccModel == 'Exponential':
        fdij = np.exp(-1 * od_matrix * Expon)
    else:
        raise ValueError(f"Unknown accessibility model: {AccModel}. Supported models are '2SFCA', 'Gravity', and 'Exponential'.")

    temp_time, cal_time_log = cal_time_logger("计算fdij", temp_time, cal_time_log)

    # calculate the total demand weighted by the distance to each facility
    Sum_Dfdki = fdij.T @ OriginPopu
    temp_time, cal_time_log = cal_time_logger("计算Sum_Dfdki", temp_time, cal_time_log)

    # calculate the Fij matrix
    Fij = fdij / Sum_Dfdki
    temp_time, cal_time_log = cal_time_logger("计算Fij", temp_time, cal_time_log)

    # calculate the current accessibility
    CurrentAcc = Fij @ DestinationSupply
    temp_time, cal_time_log = cal_time_logger("计算CurrentAcc", temp_time, cal_time_log)

    CurrentAcc_df = pd.DataFrame(CurrentAcc, columns=['CurrentAcc'])

    if print_out:
      print(f'{"Current Standard deviation:":<40} {round(CurrentAcc_df["CurrentAcc"].std(ddof=set_ddof), 6):<10}')
      print(f'{"Current Variance:":<40} {round(CurrentAcc_df["CurrentAcc"].var(ddof=set_ddof), 6):<10}')

    summary_CurrentAcc = CurrentAcc_df['CurrentAcc'].describe().round(6)
    summary_Acc = pd.DataFrame({'Current_Acc': summary_CurrentAcc}).T
    temp_time, cal_time_log = cal_time_logger("生成summary_Acc", temp_time, cal_time_log)

    cal_end_time = time.time()
    elapsed_time = cal_end_time - cal_start_time
    temp_time, cal_time_log = cal_time_logger("其余的计算", temp_time, cal_time_log)

    if print_out:
        print(summary_Acc)
        print(f"\nFunction 'calculate_accessibility' took {elapsed_time:.4f} seconds to execute.")
        print(f'the time on calculate accessibility is {cal_time_log}')

    return CurrentAcc_df, summary_Acc




# caculate_accessibility_use_np
import time
import pandas as pd
import numpy as np

def calculate_accessibility_use_np(df_destinations, df_origins, od_matrix, beta=0.6, print_out=True):

    # record start time
    start_time = time.time()

    # get the demand and supply data
    origin_ids = df_origins['OriginID'].values
    destination_ids = df_destinations['DestinationID'].values
    origin_demands = df_origins['O_Demand'].values
    destination_supplies = df_destinations['D_Supply'].values

    # make sure od_matrix is a numpy array and the shape is correct (assuming it's len(origin_ids) x len(destination_ids) matrix)
    od_matrix_np = od_matrix[['TravelCost']].values.reshape((len(origin_ids), len(destination_ids)), order='C')

    if print_out:
        print(f'OD Matrix shape: {od_matrix_np.shape}')
        print(f'Number of origins: {len(origin_ids)}, Number of destinations: {len(destination_ids)}')

    # use the new function to calculate accessibility
    CurrentAcc_df, summary_Acc = calculate_accessibility_np(
        od_matrix_np,
        pd.DataFrame({'OriginID': origin_ids, 'O_Demand': origin_demands}),
        pd.DataFrame({'DestinationID': destination_ids, 'D_Supply': destination_supplies}),
        beta=beta,
        print_out=print_out
    )

    # merge the results, and return the results to the original demand point DataFrame
    gdf_origins_temp = df_origins.copy()
    gdf_origins_temp = gdf_origins_temp.merge(
        CurrentAcc_df,
        left_on='OriginID',
        right_index=True,
        how='left'
    )

    # fill nan with 0
    gdf_origins_temp['accessibility'] = gdf_origins_temp['CurrentAcc'].fillna(0)

    # 选择需要的列并返回处理后的 DataFrame
    # choose the columns you need and return the processed DataFrame
    df_origin_temp = gdf_origins_temp[['OriginID', 'lng', 'lat', 'O_Demand', 'accessibility']].sort_values(by='OriginID').reset_index(drop=True)

    # calculate the time used
    end_time = time.time()
    elapsed_time = end_time - start_time
    if print_out:
        print(f"Function 'calculate_accessibility_use_np' took {elapsed_time:.4f} seconds to execute.")

    return df_origin_temp


# Add calculate_gini
#gini index 
def calculate_gini(df_destinations, df_origins, od_matrix):
    
    start_time = time.time()

    # Ensure consistent data type for 'OriginID' and 'DestinationID'
    df_origins['OriginID'] = df_origins['OriginID'].astype('int64')
    df_destinations['DestinationID'] = df_destinations['DestinationID'].astype('int64')
    od_matrix['OriginID'] = od_matrix['OriginID'].astype('int64')
    od_matrix['DestinationID'] = od_matrix['DestinationID'].astype('int64')

    # Merge travel costs with supply and demand data
    merged_df = od_matrix.merge(df_destinations[['DestinationID', 'D_Supply']], 
                               on='DestinationID', how='left')
    merged_df = merged_df.merge(df_origins[['OriginID', 'O_Demand']], 
                               on='OriginID', how='left')
    
    # Calculate true supply with distance decay (supply / travel_cost^2)
    # Replace TravelCost <= 0 with a small value to avoid division by zero
    merged_df['TravelCost'] = merged_df['TravelCost'].replace(0, 1e-6)
    merged_df['True_Supply'] = merged_df['D_Supply'] / (merged_df['TravelCost'] ** 2)
    
    # Group by OriginID to calculate total true supply available to each origin
    total_true_supply_per_origin = merged_df.groupby('OriginID')['True_Supply'].sum().reset_index()
    total_true_supply_per_origin.rename(columns={'True_Supply': 'Total_True_Supply'}, inplace=True)
    
    # Merge total true supply back to the main dataframe
    merged_df = merged_df.merge(total_true_supply_per_origin, on='OriginID', how='left')
    
    # Allocate demand proportionally to true supply for each hospital-origin pair
    merged_df['Allocated_Demand'] = (merged_df['O_Demand'] * merged_df['True_Supply']) / merged_df['Total_True_Supply']
    
    # Calculate supply-to-demand ratio for each hospital-origin pair
    merged_df['Ratio'] = np.where(merged_df['Allocated_Demand'] > 0, 
                                 merged_df['True_Supply'] / merged_df['Allocated_Demand'], 
                                 0)
    
    # Function to calculate Gini index for a list of ratios
    def gini_index(ratios):
        if len(ratios) == 0 or np.all(ratios == 0):
            return np.nan  # Return NaN for invalid cases
        n = len(ratios)
        mean_ratio = np.mean(ratios)
        if mean_ratio == 0:
            return np.nan  # Avoid division by zero
        diffs = np.abs(np.subtract.outer(ratios, ratios))
        gini = np.sum(diffs) / (2 * n**2 * mean_ratio)
        return gini
    
    # Calculate Gini index for each origin
    gini_per_origin = merged_df.groupby('OriginID').apply(
        lambda x: gini_index(x['Ratio'].values)
    ).reset_index(name='accessibility')  # Name the column 'accessibility'
    
    # Copy the original DataFrame and merge with Gini values
    df_origins_temp = df_origins.copy()
    df_origins_temp = df_origins_temp.merge(
        gini_per_origin,
        on='OriginID',
        how='left'
    )
    
    # Fill NaN with 0 in the 'accessibility' column
    df_origins_temp['accessibility'] = df_origins_temp['accessibility'].fillna(0)
    
    # Select required columns and sort by OriginID
    df_origins_updated = df_origins_temp[['OriginID', 'lng', 'lat', 'O_Demand', 'accessibility']].sort_values(by='OriginID').reset_index(drop=True)
    
    # Calculate execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function 'calculate_gini' took {elapsed_time:.4f} seconds to execute.")
    print(df_origins_updated)
    
    return df_origins_updated
