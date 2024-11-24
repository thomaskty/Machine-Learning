import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.float_format',lambda x : '%.3f' % x)
# pd.set_option('display.max_columns',10000)
# pd.set_option('display.max_colwidth',10000)

import numpy as np
# np.set_printoptions(suppress=True,precision=5,floatmode='fixed')

import sklearn
import sklearn as sk
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt 
import seaborn as sns
from functools import * 

import os
import dateutil
import datetime
import itertools
import pytz

from scipy import stats
from prettytable import PrettyTable

def get_current_time_ist():
    """
    Get the current time in IST (Indian Standard Time) in a compact string format.
    The function retrieves the current UTC time, converts it to IST, and formats it into
    a compact string by removing special characters like `:` and `-`.

    Returns:
    --------
    str:The current IST time formatted as `YYYYMMDDhhmmssAM/PM`.
    """
    try:
        # Define the IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        
        # Get the current UTC time
        now_utc = datetime.datetime.now(pytz.utc)
        
        # Convert UTC to IST
        now_ist = now_utc.astimezone(ist)
        
        # Format IST time as a string
        formatted_time = now_ist.strftime('%Y-%m-%d %I:%M:%S %p')
        
        # Remove unwanted characters for compact output
        replacement_dict = {':': '', '-': '', ' ': ''}
        output = formatted_time
        for old, new in replacement_dict.items():
            output = output.replace(old, new)
        
        return output

    except Exception as e:
        print(f"Error in getting current IST time: {e}")
        return None


def last_day(date):
    """
    Get the last day of the month for a given date.

    Parameters:
    -----------
    date : datetime
        The input date.

    Returns:
    --------
    datetime
        The last day of the month.
    """
    next_month = date.replace(day=28) + datetime.timedelta(days=4)  # Ensures crossing into the next month
    return next_month - datetime.timedelta(days=next_month.day)  # Subtract days to move back to the last day


def get_yyyymm(dt):
    """
    Get the month and year in 'MMYYYY' and 'YYYYMM' formats for the last day of the month.

    Parameters:
    -----------
    dt : datetime : The input date.

    Returns:
    --------
    tuple
        A tuple containing:
        - `mmyyyy` (str): Month and year in 'MMYYYY' format.
        - `yyyymm` (str): Year and month in 'YYYYMM' format.
    """
    end_of_month = last_day(dt)
    year = str(end_of_month.year)
    month = f"{end_of_month.month:02d}"  # Ensure two digits for the month
    mmyyyy = f"{month}{year}"
    yyyymm = f"{year}{month}"
    return mmyyyy, yyyymm


def get_previous_months(date_str, n):
    """
    Get the end dates of the last `n` months from a given date string.

    Parameters:
    -----------
    date_str : str : The input date in 'YYYY-MM-DD' format.
    n : int : Number of previous months to compute.

    Returns:
    --------
    tuple : A tuple of strings representing the last days of the previous `n` months in 'YYYY-MM-DD' format.
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')  # Parse the input string into a datetime object
    end_dates = []
    for _ in range(n):
        date = date - dateutil.relativedelta.relativedelta(months=1)  # Move to the previous month
        end_of_month = last_day(date)  # Get the last day of the month
        end_dates.append(end_of_month.strftime('%Y-%m-%d'))  # Format as 'YYYY-MM-DD'
    return tuple(end_dates)


def lag_n(input_date, n=-1):
    """
    Get the last day of the nth lagged month relative to a given date.

    Parameters:
    -----------
    input_date : datetime : The reference date.
    n : int, optional : The number of months to lag. Negative values move backward. (Default: -1)

    Returns:
    --------
    datetime : The last day of the lagged month.
    """
    lagged_date = input_date + dateutil.relativedelta.relativedelta(months=n)  # Compute the lagged date
    return last_day(lagged_date)  # Get the last day of that month


def show_dateinfo():
    """
    Display current and previous month's date-related information.
    This function calculates and prints the following details:
    - Current date and the date of the previous month (-1 month lag).
    - Current and previous month's names.
    - Current and previous year's values.
    - Current and previous month's representations in "mmyyyy" and "yyyymm" formats.
    Assumptions:
    - The helper functions `lag_n(date, n)` and `get_yyyymm(date)` are pre-defined:
      - `lag_n(date, n)`: Adjusts the given `date` by `n` months.
      - `get_yyyymm(date)`: Returns two formatted strings: "mmyyyy" and "yyyymm" for the given `date`.
    Returns:
    --------
    None: Outputs the date-related information directly to the console.
    """
    try:
        # Current date and previous month's date
        date_current = datetime.date.today()
        date_lag = lag_n(date_current, n=-1)
        
        # Extracting month names and years
        month_name_current = date_current.strftime('%B')
        month_name_lag = date_lag.strftime('%B')
        year_current = date_current.year
        year_lag = date_lag.year
        
        # Format current and lagged dates into "mmyyyy" and "yyyymm"
        mmyyyy_current, yyyymm_current = get_yyyymm(date_current)
        mmyyyy_lag, yyyymm_lag = get_yyyymm(date_lag)

        dd = pd.DataFrame(columns = ['Value'])

        # Print formatted results
        dd.loc["date current"] = date_current
        dd.loc["date lag (-1)"] = date_lag
        dd.loc["month name current"] = month_name_current
        dd.loc["month name lag"] = month_name_lag
        dd.loc["year current"] = year_current
        dd.loc["year lag"] = year_lag
        dd.loc["mmyyyy_current"] = mmyyyy_current
        dd.loc["yyyymm_current"] = yyyymm_current
        dd.loc["mmyyyy_lag"] = mmyyyy_lag
        dd.loc["yyyymm_lag"] = yyyymm_lag
        dd.style.set_properties(**{'text-align': 'left'})
        table(dd)

    except NameError as e:
        print(f"Error: Missing required function or variable: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def trend(x):
    """
    Compute the normalized trend score of a sequence.
    The trend score measures the relative ordering of elements in a sequence.
    A positive score indicates an increasing trend, 
    while a negative score indicates a decreasing trend.

    Parameters:
    -----------
    x : array-like : A sequence of numeric values.

    Returns:float
        The normalized trend score, ranging from -1 (strictly decreasing)
        to 1 (strictly increasing). Returns 0 for a single-element sequence.
    Notes:
    ------
    - The computation involves counting the number of elements greater than, 
      less than, or equal to each element in the sequence.
    - The normalization factor ensures the score is bounded between -1 and 1.

    Example:
    --------
    >>> row_trend([1, 2, 3, 4])
    1.0
    >>> row_trend([4, 3, 2, 1])
    -1.0
    >>> row_trend([1, 3, 2, 4])
    0.3333333333333333
    """
    dd = np.asarray(x)
    
    if len(dd) <= 1:
        return 0  # Single element or empty sequence has no trend
    
    # Normalization factor for total comparisons
    denum = (len(dd) - 1) * len(dd) / 2

    suball = 0
    addall = 0
    
    # Iterate over each element to compute comparisons
    for i in range(len(dd) - 1):
        element = dd[i]
        remaining = dd[i + 1:]
        
        suball += np.sum(remaining < element)
        addall += np.sum(remaining > element)

    # Normalize the trend score
    return (addall - suball) / denum

def table(input_dataframe):
    """
    Display a Pandas DataFrame or Series as a PrettyTable.
    
    This function takes a Pandas DataFrame or Series, processes it into a format suitable 
    for PrettyTable, and displays the output in a clean tabular format. It handles:
    - Multi-index DataFrames by resetting the index.
    - Multi-level column names by flattening them.
    - Pandas Series by converting them into a two-column DataFrame.

    Parameters:
    -----------
    input_dataframe : pd.DataFrame or pd.Series
        The Pandas object to be displayed as a PrettyTable.

    Returns: None 
    --------
    Prints the PrettyTable representation of the input DataFrame or Series.
    """

    # Validate input type
    if not isinstance(input_dataframe, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a Pandas DataFrame or Series.")

    # Make a copy of the input
    df_or_series = input_dataframe.copy()

    # If input is a Series, convert to DataFrame with default column names
    if isinstance(df_or_series, pd.Series):
        index_name = df_or_series.index.name or 'index'
        series_name = df_or_series.name or 'value'
        df_or_series = df_or_series.reset_index()
        df_or_series.columns = [index_name, series_name]

    # If DataFrame has a MultiIndex, reset the index
    if isinstance(df_or_series.index, pd.MultiIndex):
        df_or_series.reset_index(inplace=True)
    else:
        # Convert the index to a column if not already part of the DataFrame
        df_or_series.reset_index(inplace=True)

    # Handle multi-level column names
    if isinstance(df_or_series.columns, pd.MultiIndex):
        # Flatten multi-level column names into single-level
        df_or_series.columns = [
            '_'.join(map(str, col)).strip('_') for col in df_or_series.columns
        ]

    # Create a PrettyTable
    table = PrettyTable()

    # Set column headers
    table.field_names = df_or_series.columns.tolist()

    # Add rows to the PrettyTable
    for row in df_or_series.itertuples(index=False):
        table.add_row(row)

    # Print the table
    print(table)
    

def iteratively_remove_correlated_features(df, threshold=0.6):
   """
   Iteratively remove correlated features, keeping only the feature with the highest variance
   until no two features have a correlation above the threshold.

   Parameters:
   df (pd.DataFrame): Input DataFrame with features to evaluate.
   threshold (float): Correlation threshold to define high correlation (default=0.6).

   Returns:
   pd.DataFrame: DataFrame containing only the selected features.
   list: List of features to keep.
   list: List of features to drop.
   pd.DataFrame: A log dataframe with detailed information for each feature pair processed.
   """
   features_to_keep = list(df.columns)
   initial_feature_count = len(features_to_keep)
   # print("Starting with {} features.".format(initial_feature_count))
   iteration = 0  # To track iterations
   log_data = []
   # To track correlated features
   correlated_features_dict = {f: set() for f in features_to_keep}

   all_features_dropped = set()

   while True:
       iteration += 1
       # print("\n--- Iteration {} ---".format(iteration))
       # print("Number of features being processed:{}".format(len(features_to_keep)))

       # Compute the correlation matrix for the remaining features
       corr_matrix = df[features_to_keep].corr().abs()

       # Identify pairs of highly correlated features
       upper_triangle =corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
       correlated_pairs = [
           (col, upper_triangle.index[idx])
           for idx, col in enumerate(upper_triangle.columns)
           for col in upper_triangle.index[upper_triangle[col] > threshold]
       ]

       pair_count = len(correlated_pairs)
       # print("Found {} correlated pairs above {}.".format(pair_count,threshold))

       if not correlated_pairs:
           # print("No pairs with correlation above the threshold.Stopping...\n")
           break

       drop_candidates = set()
       for feature_1, feature_2 in correlated_pairs:
           variance_1 = df[feature_1].var()
           variance_2 = df[feature_2].var()

           # Update correlated feature lists
           correlated_features_dict[feature_1].add(feature_2)
           correlated_features_dict[feature_2].add(feature_1)

           # Determine which feature to drop (the one with lower variance)
           to_drop = feature_1 if variance_1 < variance_2 else feature_2
           drop_candidates.add(to_drop)

           # Log details of each comparison
           log_data.append({
               "Feature1": feature_1,
               "Feature2": feature_2,
               "Feature1_Variance": np.round(variance_1,3),
               "Feature2_Variance": np.round(variance_2,3),
               "Correlation": np.round(corr_matrix.loc[feature_1, feature_2],3),
               "Dropping": to_drop,
               "Iteration_Step": iteration,
               "Feature1_Correlated_With_Count":len(correlated_features_dict[feature_1]),
               "Feature2_Correlated_With_Count":len(correlated_features_dict[feature_2]),
               "Feature1_Correlated_With_List":list(correlated_features_dict[feature_1]),
               "Feature2_Correlated_With_List":list(correlated_features_dict[feature_2]),
           })

           # print("Comparing {} (var={:.4f}) and {} (var={:.4f}) -> Dropping {}".format(
           #    feature_1, variance_1, feature_2, variance_2, to_drop))

       drop_count = len(drop_candidates)
       # print("Number of features to drop in this iteration:{}\n".format(drop_count))

       # Update the list of features to keep
       features_to_keep = [f for f in features_to_keep if f not in drop_candidates]
       all_features_dropped.update(drop_candidates)

   final_feature_count = len(features_to_keep)
   # print("\nFinal Features to Keep:", features_to_keep)
   # print("Reduced from {} features to {} features.".format(initial_feature_count, final_feature_count))

   # Create a DataFrame from the log
   log_df = pd.DataFrame(log_data)
   table(log_df[['Feature1','Feature2','Feature1_Variance','Feature2_Variance','Correlation','Dropping']]) 

   return list(all_features_dropped),log_df

def iteratively_remove_correlated_features_with_target(df, target, threshold=0.6):
    """
    Iteratively remove correlated features, keeping only the feature with the highest
    correlation with the target until no two features have a correlation above the threshold.

    Parameters:
    df (pd.DataFrame): Input DataFrame with features to evaluate.
    target (str): Target column name to calculate correlation.
    threshold (float): Correlation threshold to define high correlation (default=0.6).

    Returns:
    pd.DataFrame: DataFrame containing only the selected features.
    list: List of features to keep.
    list: List of features to drop.
    pd.DataFrame: A log dataframe with detailed information for each feature pair processed.
    """
    features = list(df.columns)
    features.remove(target)  # Remove target from the list of features
    features_to_keep = features.copy()
    log_data = []
    iteration = 0
    correlated_features_dict = {f: set() for f in features_to_keep}
    all_features_dropped = set()

    while True:
        iteration += 1

        # Compute the correlation matrix for the remaining features
        corr_matrix = df[features_to_keep].corr().abs()

        # Identify pairs of highly correlated features
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_pairs = [
            (col, upper_triangle.index[idx])
            for idx, col in enumerate(upper_triangle.columns)
            for col in upper_triangle.index[upper_triangle[col] > threshold]
        ]

        if not correlated_pairs:
            break

        drop_candidates = set()
        for feature_1, feature_2 in correlated_pairs:
            target_corr_1 = df[feature_1].corr(df[target])
            target_corr_2 = df[feature_2].corr(df[target])

            # Update correlated feature lists
            correlated_features_dict[feature_1].add(feature_2)
            correlated_features_dict[feature_2].add(feature_1)

            # Determine which feature to drop (the one with lower correlation to the target)
            to_drop = feature_1 if abs(target_corr_1) < abs(target_corr_2) else feature_2
            drop_candidates.add(to_drop)

            # Log details of each comparison
            log_data.append({
                "Feature1": feature_1,
                "Feature2": feature_2,
                "Feature1_Target": np.round(target_corr_1,3),
                "Feature2_Target": np.round(target_corr_2,3),
                "Correlation": np.round(corr_matrix.loc[feature_1, feature_2],3),
                "Dropping": to_drop,
                "Iteration_Step": iteration,
                "Feature1_Correlated_With_Count": len(correlated_features_dict[feature_1]),
                "Feature2_Correlated_With_Count": len(correlated_features_dict[feature_2]),
                "Feature1_Correlated_With_List": list(correlated_features_dict[feature_1]),
                "Feature2_Correlated_With_List": list(correlated_features_dict[feature_2]),
            })

        # Update the list of features to keep
        features_to_keep = [f for f in features_to_keep if f not in drop_candidates]
        all_features_dropped.update(drop_candidates)

    # Create a DataFrame from the log
    log_df = pd.DataFrame(log_data)
    table(log_df[['Feature1','Feature2','Feature1_Target','Feature2_Target','Correlation','Dropping']]) 

    return list(all_features_dropped), log_df

def scale_data(data,method='standard',feature_range=(0,1)):
   if len(data.shape) ==1:
       data = data.reshape(-1,1)
   if method.lower()=='standard':
       scaler = preprocessing.StandardScaler()
   elif method.lower()=='minmax':
       scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
   else:
       raise ValueError('Method must be either "standard" or "minmax"')
   scaled_data = scaler.fit_transform(data)
   return scaled_data,scaler

def inverse_scale_data(scaled_data,scaler):
   if len(scaled_data.shape)==1:
       scaled_data = scaled_data.reshape(-1,1)
   original_data = scaler.inverse_transform(scaled_data)
   return original_data

def get_combinations(param_dict):
   keys = param_dict.keys()
   values = param_dict.values()
   param_combinations = itertools.product(*values)
   param_dicts = [dict(zip(keys,combination)) for combination in param_combinations]
   return param_dicts


def optimal_nobins(data):
    """
    The Freedman-Diaconis rule is very robust and works well in practice. T
    he bin-width is set to h=2*IQR*n-1/3. 
    So the number of bins is $(max-min)/h$, 
    where n is the number of observations, 
    max is the maximum value 
    and min is the minimum value
    adjustment : instead of dividing max_min we divide iqr(to reduce the number of bins)
    """
    n = len(data)
    min_max = data.max()-data.min()
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    h = 2*(iqr*(n**(-1/3)))
    return {'iqr':int(iqr/h),'min_max':int(min_max/h)}

def analyse_feature(df,feat1,feat2,relation,bins=False):
    """
    Analyze the relationship between two features based on their data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the features to be analyzed.
    feat1 : str
        The name of the first feature.
    feat2 : str
        The name of the second feature.
    relation : str
        Specifies the relationship between features. Supported values:
        - 'cat2cat': Categorical to Categorical
        - 'con2con': Continuous to Continuous
        - 'cat2con': Categorical to Continuous
        - 'con2cat': Continuous to Categorical
    bins : int, str, list, or bool, optional (default=False)
        Determines binning strategy for continuous features:
        - int: Number of bins.
        - str: 'optimal' to use the optimal binning method (`optimal_nobins`).
        - list: List of tuples defining custom bins.
        - False: No binning.

    Returns:
    --------
    dict or pd.DataFrame
        - For 'cat2cat' and 'con2cat': Returns a dictionary with keys:
          - 'counts': Absolute counts of occurrences.
          - 'total_ratio': Percentage of total for each bin.
          - 'bin_ratio': Percentage distribution within each bin.
        - For 'con2con' and 'cat2con': Returns a DataFrame summarizing 
          statistical measures for each group/bin.

    Notes:
    ------
    - Use `optimal_nobins` for automatic bin size calculation.
    - Handles missing values by grouping them appropriately during analysis.
    - Ensure features are correctly categorized before using the function.

    Examples:
    ---------
    # Example 1: Categorical to Continuous
    analyse_feature(df, 'gender', 'income', relation='cat2con')

    # Example 2: Continuous to Categorical
    analyse_feature(df, 'age', 'purchase', relation='con2cat', bins=5)
    """
    if relation.lower() in ['cat2cat','con2cat']:
        print('feature analysis output is a dictionary with keys counts,total_ratio,bin_ratio')
    try:
        temp = df[[feat1,feat2]].copy()
        if relation.lower()=='cat2cat':
            output_dict = dict() 
            feat2_labels = temp[feat2].value_counts().index
            for i in feat2_labels:
                temp[i] = np.where(temp[feat2]==i,1,0)
            output1 = temp.groupby(feat1).agg({i:'sum' for i in feat2_labels})
            output1.loc['total'] = output1.sum()
            
            output2 = (output1/temp.shape[0])*100
            output3 = output1.div(output1.sum(axis=1), axis=0)
            output3 = (output3*100)
            
            output1['total'] = output1.sum(axis=1)
            output_dict['counts'] = output1 
            
            output2['total'] = output2.sum(axis=1)
            for j in output2.columns:
                output2[j] = output2[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['total_ratio'] = output2
            
            output3['total'] = output3.sum(axis=1)
            for j in output3.columns:
                output3[j] = output3[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['bin_ratio'] = output3 

            print('Counts') 
            table(output_dict['counts'])
            print('Total Ratio') 
            table(output_dict['total_ratio'])
            print('Binwise Ratio') 
            table(output_dict['bin_ratio'])
            
        elif relation.lower()=='con2con':
            # creating bins for feature a if not given
            if type(bins)==str:
                nob = optimal_nobins(temp[feat1])['iqr']
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = nob,duplicates = 'drop')
            elif type(bins)==int:
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = bins,duplicates = 'drop')
            elif type(bins)==list:
                bins = pd.IntervalIndex.from_tuples(bins)
                temp[feat1+'_bins'] = pd.cut(temp[feat1],bins=bins,duplicates = 'drop')

            output1 = temp.groupby(feat1+'_bins').describe()[feat2]
            output1['25%'] = np.round(output1['25%'],3)
            output1['50%'] = np.round(output1['50%'],3)
            output1['75%'] = np.round(output1['75%'],3)
            output1['mean'] = np.round(output1['mean'],3)
            output1['std'] = np.round(output1['std'],3)
            output1['ratio'] = output1['count']/output1['count'].sum()
            output1['ratio'] = output1[['ratio']].applymap(lambda x: "{0:.2f}%".format(x*100))
            
            table(output1) 
        
        elif relation.lower()=='cat2con':
            output1 = temp.groupby(feat1).describe()[feat2]
            output1['25%'] = np.round(output1['25%'],3)
            output1['50%'] = np.round(output1['50%'],3)
            output1['75%'] = np.round(output1['75%'],3)
            output1['mean'] = np.round(output1['mean'],3)
            output1['std'] = np.round(output1['std'],3)
            
            output1['ratio'] = output1['count']/output1['count'].sum()
            output1['ratio'] = output1[['ratio']].applymap(lambda x: "{0:.2f}%".format(x*100))
            table(output1) 
            
        elif relation.lower()=='con2cat':
            output_dict = dict()

            if type(bins)==str:
                nob = optimal_nobins(temp[feat1])['iqr']
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = nob,duplicates = 'drop')
            elif type(bins)==int:
                temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = bins,duplicates = 'drop')
            elif type(bins)==list:
                bins = pd.IntervalIndex.from_tuples(bins)
                temp[feat1+'_bins'] = pd.cut(temp[feat1],bins=bins,duplicates = 'drop')
            
            feat2_labels = temp[feat2].value_counts().index
            for i in feat2_labels:
                temp[i] = np.where(temp[feat2]==i,1,0)
            output1 = temp.groupby(feat1+'_bins').agg({i:'sum' for i in feat2_labels})
            output1.loc['total'] = output1.sum()
            
            output2 = (output1/temp.shape[0])*100
            output3 = output1.div(output1.sum(axis=1), axis=0)
            output3 = (output3*100)
            
            output1['total'] = output1.sum(axis=1)
            output_dict['counts'] = output1
            
            output2['total'] = output2.sum(axis=1)
            for j in output2.columns:
                output2[j] = output2[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['total_ratio'] = output2
            
            output3['total'] = output3.sum(axis=1)
            for j in output3.columns:
                output3[j] = output3[[j]].applymap(lambda x: "{0:.2f}%".format(x))
            output_dict['bin_ratio'] = output3

            print('Counts') 
            table(output_dict['counts'])
            print('Total Ratio') 
            table(output_dict['total_ratio'])
            print('Binwise Ratio') 
            table(output_dict['bin_ratio'])
            
    except Exception as e:
        print('Exception  : ',e)

def woe_iv(data,feature,target,events,bins,continuous):
    """
    target may contain more than two labels
    events,non events in the target columns 
    woe = log2(%events/%nonevents)
    iv = woe * (% events-% non events)
    """
    temp = data[[feature,target]].copy()
    if continuous:
        if type(bins)==str:
            temp[feature+'_bins'] = pd.qcut(
                temp[feature],
                q = optimal_nobins(temp[feature])['iqr'],
                duplicates='drop'
            )
        elif type(bins)==list:
            bins = pd.IntervalIndex.from_tuples(bins)
            temp[feature+'_bins'] = pd.cut(temp[feature],bins)
        elif type(bins)==int:
            temp[feature+'_bins'] = pd.qcut(temp[feature],q = bins,duplicates='drop')

        feature = feature+'_bins'

    target_labels = temp[target].value_counts().index 
    temp[events] = np.where(temp[target]==events,1,0)
    temp = temp.groupby(feature).agg({events:['sum','count']})
    temp['%events'] = np.round(temp[events]['sum']/temp[events]['sum'].sum(),3)
    temp['#nonevents'] = temp[events]['count']-temp[events]['sum']
    temp['%nonevents'] = np.round(temp['#nonevents']/temp['#nonevents'].sum(),3)
    temp['woe'] = np.round(np.log((temp['%events']) / (temp['%nonevents'])),3)
    temp['IV'] = np.round(temp['woe']* (temp['%events'] - temp['%nonevents']),3)
    table(temp) 
    return temp

def thresh_analysis(model,x,y):
    cols = ['thresh','precision','recall','f1','accuracy','tp','fp','tn','fn']
    output = pd.DataFrame(columns=cols)
    thresh_list = [i/100 for i in list(range(30,80,1))]
    probs = model.predict_proba(x)[:,1]
    for i in range(len(thresh_list)):
        y_pred = np.where(probs>=thresh_list[i],1,0)
        cf = confusion_matrix(y,y_pred)
        tn,fp,fn,tp = cf[0,0],cf[0,1],cf[1,0],cf[1,1]
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        row = [thresh_list[i],p,r,f1,accuracy,tp,fp,tn,fn]
        output.loc[i] = row
    return output

def deciles(x,y,model):
    output = pd.DataFrame({'target':y,'proba':model.predict_proba(x)[:,1]})
    output['decile'],probability_cuts = pd.qcut(
        output['proba'],q = 10,
        duplicates='drop',retbins=True
    )
    return output 

def aggregate(data):
    temp = data.copy()
    temp = temp.groupby('decile').agg({'target':['count',sum],'proba':['min','max']})
    temp.sort_index(ascending = False,inplace = True)
    temp['cust%'] = temp[('target','count')]/temp[('target','count')].sum()
    temp['cum_cust%'] = temp['cust%'].cumsum()
    temp['capture'] = temp[('target','sum')]/temp['target']['count'].sum()
    temp['gain'] = temp[('target','count')]/temp[('target','sum')].sum()
    temp['cum_gain'] = temp['gain'].cumsum()
    temp['lift'] = temp['cum_gain']/temp['cum_cust%'].cumsum() 
    return temp.sort_index(ascending = False)

def prediction_frame(x,y,model,threshold):
    pred_df = pd.concat([x,y],axis=1)
    pred_df['proba'] = model.predict_proba(x)[:,1]
    pred_df['prediction'] = np.where(pred_df['proba']>threshold,1,0)
    pred_df['tp'] = np.where(((pred_df[y.name]==1)&(pred_df['prediction']==1)),1,0)
    pred_df['tn'] = np.where(((pred_df[y.name]==0)&(pred_df['prediction']==0)),1,0)
    pred_df['fp'] = np.where(((pred_df[y.name]==0)&(pred_df['prediction']==1)),1,0)
    pred_df['fn'] = np.where(((pred_df[y.name]==1)&(pred_df['prediction']==0)),1,0)
    pred_df['decile'] = pd.qcut(pred_df['proba'],q=10,duplicates='drop')
    return pred_df 

def population_stability_index(population,sample,model,name):
    """
    name represents the dataset under consideration (test,validaiton) 
    """
    def get_pred_fr(x,model):
        pred_df = x.copy()
        pred_df['proba'] = model.predict_proba(x)[:,1]
        return pred_df 
    
    pred_df_population = get_pred_fr(population,model)
    pred_df_sample = get_pred_fr(sample,model)
    pred_df_population['psi_decile'],proba_cuts = pd.qcut(
        pred_df_population['proba'],
        q = 10,
        duplicates='drop',
        retbins=True
    )
    proba_cuts[0] = 0 
    proba_cuts[-1] = 1
    pred_df_population['psi_decile'] = pd.cut(pred_df_population['proba'],bins = proba_cuts)
    pred_df_sample['psi_decile'] = pd.cut(pred_df_sample['proba'],bins = proba_cuts)
    score_b = pred_df_sample.groupby('psi_decile').agg({
        'proba':'count'
    }).sort_index(ascending = False)
    score_a = pred_df_population.groupby('psi_decile').agg({
        'proba':'count'
    }).sort_index(ascending = False)
    score_a.columns = ['population_A']
    score_b.columns = [name]
    psi_df = pd.concat([score_a,score_b],axis=1).sort_index()
    psi_df['A%'] = np.round((psi_df['population_A']/psi_df['population_A'].sum())*100)
    psi_df['B%'] = np.round((psi_df[name]/psi_df[name].sum())*100)
    psi_df['A% - B%'] = psi_df['A%']- psi_df['B%']
    psi_df['log(A%/B%)'] = np.log(psi_df['A%']/psi_df['B%'])

    psi_df['A%'] = psi_df[['A%']].applymap(lambda x: "{0:.0f}%".format(x))
    psi_df['B%'] = psi_df[['B%']].applymap(lambda x: "{0:.0f}%".format(x))

    psi_df['psi'] = (psi_df['A% - B%']/100)*psi_df['log(A%/B%)']
    psi_df['A% - B%'] = psi_df[['A% - B%']].applymap(lambda x: "{0:.0f}%".format(x))

    psi_df.sort_index(ascending = False,inplace = True)
    
    return psi_df

def auc_gini(x,y,model):
    preds = model.predict_proba(x)[:,1]
    fpr,tpr,thresholds = roc_curve(y,preds)
    auc_ = auc(fpr,tpr)
    gini = (2*auc_)-1
    return {'auc':auc_,'gini':gini}

def error_analysis_feature_bins(x,y,model,t,f,bins):
    """
    Dividing the feature values into bins, showing the model performance in  each bin.
    t - threshold
    f - feature_name
    bins - [optimal,integer,actual_bins]
    """
    pred_df = prediction_frame(x,y,model,t)
    if type(bins)==int:
        pred_df[f]= pd.qcut(pred_df[f],q = bins,duplicates = 'drop')
    elif type(bins)==str:
        pred_df[f] = pd.qcut(pred_df[f],q = optimal_nobins(pred_df[f])['iqr'],duplicates='drop')
    elif type(bins)==list:
        bins = pd.IntervalIndex.from_tuples(bins)
        pred_df[f] = pd.cut(pred_df[f],bins = bins)

    mc = pred_df.groupby(f).agg({i:sum for i in ['tn','tp','fn','fp']}).reset_index()
    mc['+ preds'] = mc[['tp','fp']].sum(axis=1)
    mc['- preds'] = mc[['tn','fn']].sum(axis=1)
    mc['+ actual'] = mc[['tp','fn']].sum(axis=1)
    mc['- actual'] = mc[['tn','fp']].sum(axis=1)
    mc['true preds'] = mc[['tn','tp']].sum(axis=1)
    mc['false preds'] = mc[['fp','fn']].sum(axis=1)
    mc['precision'] = mc['tp']/mc['+ preds']
    mc['recall'] = mc['tp'] /mc['+ actual']
    mc['f1'] = (2*mc['precision']*mc['recall'])/(mc['precision']+mc['recall'])
    mc['accuracy'] = mc[['tp','tn']].sum(axis=1)/mc[['tp','tn','fn','fp']].sum(axis=1)
    return mc 

def get_metrices(x,y,model,thresh):
    pred_df = prediction_frame(x,y,model,thresh)
    dict_final = {}
    for i in ['tp','tn','fp','fn']:
        dict_final[i] = pred_df[i].sum()

    tp,tn = dict_final['tp'],dict_final['tn']
    fp,fn = dict_final['fp'],dict_final['fn']
    
    precision = np.round(tp/(tp+fp),2)
    recall = np.round(tp/(tp+fn),2)
    f1 = np.round((2*precision*recall)/(precision+recall),2)
    accuracy = np.round((tp+tn)/(tp+fn+fp+fn),2)
    return {'precision':precision,'recall':recall, 'f1':f1,'accuracy':accuracy}


def confusion_matrix_formatted(x,y,model,name,thresh):
    pred_df = prediction_frame(x,y,model,thresh)
    dict_final = {}
    for i in ['tp','tn','fp','fn']:
        dict_final[i] = pred_df[i].sum()
    d = pd.DataFrame(index = [0,1],columns= pd.MultiIndex.from_tuples([
        ('{}_{}'.format(name,thresh),'Predicted',0),
        ('{}_{}'.format(name,thresh),'Predicted',1)
    ]))
    d.index.name = 'Actual'
    d.loc[0] = [dict_final['tn'],dict_final['fp']]
    d.loc[1] = [dict_final['fn'],dict_final['tp']]
    d.loc['Total'] = [int(i) for i in d['{}_{}'.format(name,thresh)]['Predicted'].sum()]
    d[('{}_{}'.format(name,thresh),'Predicted','total')] = d.sum(axis=1)
    return d

def calculate_psi(expected,actual,buckettype='bins',buckets=10,axis=0):
    def psi(expected_array,actual_array,buckets):
        def scale_range(input_,min_,max_):
            input_ +=-(np.min(input_))
            input_ /=np.max(input_)/(max_-min_)
            input_ +=min
            return input_
        breakpoints = np.arange(0,buckets+1)/(buckets)*100
        if buckettype=='bins':
            breakpoints = scale_range(breakpoints,np.min(expected_array),np.max(expected_array))
        elif buckettype=='quantiles':
            breakpoints = np.stack([np.percentile(expected_array,b) for b in breakpoints])
        expected_percents = np.histogram(expected_array,breakpoints)[0]/len(expected_array)
        actual_percents = np.histogram(actual_array,breakpoints)[0]/len(actual_array)
        def sub_psi(e_perc,a_perc):
            if a_perc ==0:
                a_perc= 0.0001
            if e_perc ==0:
                e_perc=0.0001
            value = (e_perc - a_perc)*np.log(e_perc/a_perc)
            return (value)
        psi_value = np.sum(sub_psi(
            expected_percents[i],actual_percents[i]
        ) for i in range(0,len(expected_percents)))

        return (psi_value)
    if len(expected.shape)==1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_value = np.empty(expected.shape[axis])
    for i in range(0,len(psi_values)):
        if len(psi_values)==1:
            psi_values = psi(expected,actual,buckets)
        elif axis==0:
            psi_values[i] = psi(expected[:,i],actual[:,i],buckets)
        elif axis==1:
            psi_values[i] = psi(expected[i,:],actual[i,:],buckets)
    return (psi_values)


def calculate_psi_category(initial,new):
    df1 = pd.DataFrame(initial.value_counts(normalize = True).reset_index())
    df2 = pd.DataFrame(new.value_counts(normalize = True).reset_index())
    df = pd.merge(df1,df2,how = 'left',on='index')
    df.columns = ['index','initial_perc','new_perc']
    df.fillna(0.0001,inplace = True)
    df['PSI'] = (df['new_perc']-df['initial_perc'])*np.log(df['new_perc']/df['initial_perc'])
    return np.round(np.sum(df['PSI']),5)

def psi_alert_label(psi):
    return np.where(psi<0.1,'low',np.where(psi>=0.2,'high','medium'))

def calculate_covariate_drift(old_data,new_data):
    li_common_cols = old_data.columns.intersection(new_data.columns)
    old_data = old_data[li_common_cols]
    new_data = new_data[li_common_cols]
    num_cols = old_data._get_numeric_data().columns
    obj_cols = list(set(li_common_cols)-set(num_cols))
    t = pd.DataFrame(columns = ['PSI Value','Drift Level'])
    
    for var in num_cols:
        psi = calculate_psi(old_data[var],new_data[var],buckettype='quantiles',buckets = 10,axis=1)
        t.loc[var] = [psi,psi_alert_label(psi)]
    for  var in obj_cols:
        psi = calculate_psi_category(old_data[var],new_data[var])
        t.loc[var] = [psi,psi_alert_label(psi)]
    return t


def detect_model_drift(model,data_old,y_old,data_new,y_new,tolerance,thresh):
    model_score_old = get_metrices(data_old,y_old,model,thresh)
    f1_old = model_score_old['f1']
    precision_old = model_score_old['precision']
    recall_old = model_score_old['recall'] 

    new_scores = get_metrices(data_new,y_new,model,thresh)
    f1_new = new_scores['f1']
    precision_new = new_scores['precision']
    recall_new = new_scores['recall']

    precision_flag = False
    recall_flag = False
    f1_flag = False
    if f1_new<(f1_old-f1_old*tolerance):
        f1_flag=True
    if precision_new <(precision_old-precision_old*tolerance):
        precision_flag = True
    if recall_new <(recall_old-recall_old*tolerance):
        recall_flag = True
    t = pd.DataFrame(columns = ['Drift Observed'])
    t.loc['f1'] = [f1_flag]
    t.loc['recall'] = [recall_flag]
    t.loc['precision'] = [precision_flag]
    return t 

def calculate_residual_drift(model,x_new,y_new,x_old,y_old):
    """
    for regression models
    """
    old_predictions = model.predict(x_old)
    new_predictions = model.predict(x_new)
    residuals_old = y_old - old_predictions 
    residuals_new = y_new - new_predictions

    psi = calculate_psi(residuals_old,residuals_new,buckettype='quantiles',buckets=10,axis=1)
    output = pd.DataFrame(columns = ['psi_value','drift_level'])
    output.loc['observed_residual_shift'] = [psi,psi_alert_label(psi)]
    return output

def chi_selection(data, target_name):
    """
    Perform Chi-Square test of independence to evaluate the 
    relationship between features and the target variable.
    
    The function computes the Chi-Square statistic and the p-value 
    for each feature to help with feature selection in 
    machine learning models. A lower p-value indicates a 
    stronger relationship between the feature and the target.
    
    This test is used to identify if there is an association between categorical features and the target, 
    and it helps in determining the most relevant features for further analysis.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing the features and the target variable.
    
    target_name : str
        The name of the target variable (column).
    
    Returns:
    --------
    pandas DataFrame
        A DataFrame containing the Chi-Square statistics and p-values for each feature, 
        sorted by the Chi-Square statistic.
        The DataFrame has the following columns:
        - `p_value`: The p-value of the Chi-Square test.
        - `chi_square_stats`: The Chi-Square statistic.
    
    Notes:
    ------
    - The function handles both categorical and continuous features.
    - Continuous features are binned into categories before applying the Chi-Square test.
    - The Chi-Square statistic measures how much the observed values differ 
    - from the expected values under the null hypothesis.
    - Smaller p-values (usually below 0.05) suggest that the feature is **not independent** of the target, 
    indicating a potential for being important in predicting the target.
    
    Example:
    --------
    # Example usage
    result = chi_selection(df, 'Target')
    print(result)
    """
    
    # Make a deep copy of the data
    cdata = data.copy()
    
    # Import necessary module
    from scipy import stats
    
    # Initialize variables for the target and feature set
    temp_df = cdata.drop(target_name, axis=1)
    temp_y = cdata[target_name]
    
    # Initialize an empty DataFrame to store processed features
    temp_new = pd.DataFrame()
    
    # Try to identify categorical columns, handle exceptions if any
    try:
        object_cols = cdata.describe(include='O').T.index.tolist()
    except Exception as e:
        print(f"Error in identifying categorical columns: {e}")
        object_cols = []
    
    # Identify non-categorical columns (excluding the target column)
    cols_categorized = [i for i in cdata.columns if i not in object_cols + [target_name]]
    
    # Handle continuous features by filling missing values and binning
    for j in cols_categorized:
        try:
            mode_j = temp_df[j].mode().values[0]
            temp_df[j].fillna(mode_j, inplace=True)
            nob = optimal_nobins(temp_df[j])['iqr']
            temp_new[j + '_bin'] = pd.qcut(temp_df[j], q=nob, duplicates='drop')
        except Exception as e:
            print(f"Error processing feature {j}: {e}")
            cols_categorized.remove(j)
    
    # Fill remaining missing values with 0
    try:
        temp_df.fillna(0, inplace=True)
    except Exception as e:
        print(f"Error while filling missing values: {e}")
    
    # Separate features and target variable
    X, y = temp_new, temp_y
    
    # Initialize lists to store p-values and chi-square statistics
    p_values = []
    chi_square_stats = []
    
    # Calculate p-values and Chi-Square statistics for each feature
    for feature in X.columns:
        try:
            chi2_stat, p_val, _, _ = stats.chi2_contingency(pd.crosstab(X[feature], y))
            p_values.append(p_val)
            chi_square_stats.append(chi2_stat)
        except Exception as e:
            print(f"Error calculating Chi-Square for feature {feature}: {e}")
            p_values.append(None)
            chi_square_stats.append(None)
    
    # Create the output DataFrame with p-values and Chi-Square statistics
    out = pd.DataFrame({
        'feature': X.columns,
        'p_value': p_values,
        'chi_square_stats': chi_square_stats
    })
    
    # Sort the output DataFrame by Chi-Square statistics and set feature as index
    out = out.sort_values(by='chi_square_stats', ascending=True).set_index('feature')
    
    return out


def entropy(values, probability=False):
    """
    Calculate the Shannon entropy of a dataset or probability distribution.

    Entropy is a measure of the uncertainty or disorder in a dataset. It quantifies
    the amount of information required to describe the state of the system. In machine
    learning, entropy is commonly used for feature selection, information gain calculation,
    and decision tree splitting. Higher entropy indicates greater uncertainty or disorder.

    This function can compute entropy in two ways:
    1. When `probability=False`, it computes entropy based on categorical data.
       The data should be a list or array of categorical values.
    2. When `probability=True`, it computes entropy based on a given probability distribution.
       The input should be a list or array of probabilities, where the values must sum to 1.

    Parameters:
    -----------
    values : array-like
        The input values representing categorical data or probabilities.
        If `probability=False`, `values` should be a list or array of categorical data.
        If `probability=True`, `values` should be a list of probabilities that sum to 1.
    
    probability : bool, optional, default=False
        If True, the input `values` represents a probability distribution (should sum to 1).
        If False, the input `values` represents categorical data, and the function computes
        entropy based on the relative frequencies of the categories.

    Returns:
    --------
    float
        The entropy value of the dataset or probability distribution.
        The value ranges from 0 (no uncertainty) to log2(n) (maximum uncertainty),
        where `n` is the number of distinct categories.

    Raises:
    -------
    ValueError
        If `probability=True` and the values do not sum to 1 or if any value is 
        outside the range [0, 1].
    
    Notes:
    ------
    - When `probability=False`, this function calculates entropy from categorical data 
      using the frequencies of the unique values.
    - Entropy is maximized when all outcomes are equally likely and minimized when one 
      outcome is certain (i.e., all values are the same).
    - If `probability=True`, this function uses the formula for entropy from information theory.
    - A higher entropy value indicates more uncertainty or disorder in the dataset.

    Example:
    --------
    # Example 1: Entropy of categorical data
    values = [1, 1, 2, 2, 2, 3, 3, 3, 3]
    print(entropy(values))  # Output: 1.4591479170272448
    
    # Example 2: Entropy of a probability distribution
    prob_values = [0.2, 0.3, 0.5]
    print(entropy(prob_values, probability=True))  # Output: 1.3709505944546687
    """
    
    if probability:
        # Ensure that the values are valid probabilities (sum to 1 and between 0 and 1)
        if not np.all(values >= 0) or not np.isclose(np.sum(values), 1):
            raise ValueError("Input values must be valid probabilities that sum to 1.")
        
        # Handle log(0) by using np.where to replace 0s with a small number (or just zero contribution)
        return -np.sum(values * np.log2(values + np.finfo(float).eps))  # Adding epsilon to avoid log(0)
    else:
        # Calculate entropy from raw counts (discrete values)
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)
        return -np.sum(probs * np.log2(probs + np.finfo(float).eps))  # Avoid log(0) by adding epsilon


def compute_probabilities(labels):
    """
    Compute the probability distribution of unique labels in the input data.
    
    This function calculates the probability of each unique label in the input array 
    by counting its occurrences and dividing by the total number of labels. It returns 
    a dictionary where the keys are the unique labels and the values are their 
    corresponding probabilities.

    Parameters:
    -----------
    labels : array-like
        A sequence (list, numpy array, pandas Series) containing categorical labels.
    
    Returns:
    --------
    dict
        A dictionary where keys are the unique labels and values are their corresponding probabilities.

    Raises:
    -------
    ValueError: 
        If `labels` is not an array-like structure or is empty.
    
    Example:
    --------
    labels = ['A', 'B', 'A', 'A', 'B', 'C']
    result = compute_probabilities(labels)
    print(result)  # Output: {'A': 0.5, 'B': 0.333, 'C': 0.16666}
    """
    
    # Check if input is valid (array-like structure)
    if not hasattr(labels, '__iter__'):
        raise ValueError("Input should be an array-like structure (e.g., list, numpy array, or pandas Series).")
    
    # Check if the input is empty
    if len(labels) == 0:
        raise ValueError("Input array cannot be empty.")
    
    # Get unique labels and their respective counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / np.sum(counts)
    
    # Map each unique label to its probability
    out = dict(zip(unique_labels, probabilities))
    
    return out


def cross_entropy(p, q, probabilities=False, verbose=True):
    """
    Compute the cross-entropy between two distributions or sets of labels.

    Cross-entropy is a measure of the difference between two probability distributions.
    It quantifies the "distance" between the true distribution `p` and the estimated distribution `q`.
    
    If the inputs `p` and `q` are raw labels (not probabilities), the function first computes the
    probabilities of each label and then calculates the cross-entropy. If `p` and `q` are already 
    probabilities, the function directly computes the cross-entropy.

    Parameters:
    -----------
    p : array-like
        The first distribution (labels or probabilities).
    
    q : array-like
        The second distribution (labels or probabilities).
    
    probabilities : bool, default=False
        If False, `p` and `q` are treated as raw labels, and their probabilities will be computed.
        If True, `p` and `q` are treated as probability distributions.
    
    verbose : bool, default=True
        If True, prints intermediate steps (probabilities, final labels, etc.) for debugging purposes.
    
    Returns:
    --------
    float
        The cross-entropy value between the two distributions or label sets.
    
    Example:
    --------
    p = ['A', 'A', 'B', 'B', 'B']
    q = ['A', 'A', 'A', 'B', 'C']
    result = cross_entropy(p, q)
    print(result)  # Output: a scalar value representing the cross-entropy
    """
    
    if not probabilities:
        labels_p = list(np.unique(p))
        labels_q = list(np.unique(q))
        
        # Determine the union of labels from both p and q
        final_labels = labels_q if len(labels_q) > len(labels_p) else labels_p
        
        # Compute probabilities for p and q
        p = compute_probabilities(p)
        q = compute_probabilities(q)
        
        # Handle missing labels
        for label in final_labels:
            if label not in p:
                p[label] = 0
            if label not in q:
                q[label] = 0
        
        if verbose:
            print(f"Computed probabilities for p: {p}")
            print(f"Computed probabilities for q: {q}")
            print(f"Final labels: {final_labels}")
        
        # Compute cross-entropy
        cross_entropy_measure = []
        for label in final_labels:
            if q[label] != 0:
                cross_entropy_measure.append(p[label] * np.log2(1 / q[label]))
            else:
                cross_entropy_measure.append(0)
    
    else:
        # Directly compute cross-entropy if probabilities are given
        p = np.array(p)
        q = np.array(q)
        
        cross_entropy_measure = p * np.log2(1 / q)
        cross_entropy_measure[np.isinf(cross_entropy_measure)] = 0  # Handle log(0)
    
    if verbose:
        print(f"Cross-entropy measure: {cross_entropy_measure}")
    
    return np.sum(cross_entropy_measure)


def joint_entropy(x, y):
    """
    Compute the joint entropy between two distributions or sets of labels.
    
    Joint entropy measures the amount of uncertainty or information shared between two variables.
    It quantifies how much uncertainty remains when we observe both variables simultaneously.

    Parameters:
    -----------
    x : array-like
        The first distribution (labels or values) representing one random variable.
    
    y : array-like
        The second distribution (labels or values) representing another random variable.
    
    Returns:
    --------
    float
        The joint entropy value between the two variables.

    Example:
    --------
    x = ['A', 'A', 'B', 'B', 'C']
    y = ['X', 'Y', 'Y', 'X', 'Z']
    result = joint_entropy(x, y)
    print(result)  # Output: a scalar value representing the joint entropy
    """
    
    # Get unique values and counts for x and y
    x_values, x_counts = np.unique(x, return_counts=True)
    y_values, y_counts = np.unique(y, return_counts=True)

    # Create joint frequency matrix
    joint_counts = np.zeros((len(x_values), len(y_values)), dtype=int)
    for i in range(len(x)):
        x_idx = np.where(x_values == x[i])[0][0]
        y_idx = np.where(y_values == y[i])[0][0]
        joint_counts[x_idx, y_idx] += 1
    
    # Calculate joint probabilities
    joint_probs = joint_counts / len(x)

    # Compute joint entropy, handling zero probabilities
    return -np.sum(joint_probs * np.log2(joint_probs + (joint_probs == 0)))

def mutual_information_value(x, y, verbose=False):
    """
    Compute the Mutual Information (MI) between two variables (features).
    
    Mutual Information measures the amount of information that knowing one variable provides
    about the other. It is based on the concept of entropy and the relationship between the
    joint distribution of the variables and their marginal distributions.
    
    Formula:
    --------
    MI(x, y) = sum(p(x, y) * log2(p(x, y) / (p(x) * p(y)))
    
    Parameters:
    -----------
    x : array-like
        The first variable (feature).
    
    y : array-like
        The second variable (target).
    
    verbose : bool, optional (default=False)
        If True, intermediate steps and debug information are printed.
    
    Returns:
    --------
    float
        The computed Mutual Information between the two variables.
    
    Example:
    --------
    x = [1, 2, 1, 2, 1]
    y = [1, 1, 2, 2, 1]
    result = mutual_information_value(x, y)
    print(result)  # Output: A scalar value representing the mutual information.
    """
    
    # Ensure that x and y are Pandas Series
    try:
        x.name = 'feature'
        y.name = 'target'
    except:
        pass
    
    if type(y) != pd.core.series.Series:
        y = pd.Series(y, name='target')
        
    if type(x) != pd.core.series.Series:
        x = pd.Series(x, name='feature')
    
    # Create a contingency table (cross-tabulation)
    output = pd.crosstab(x, y, margins=True, margins_name='total')
    if verbose:
        print(output)
    
    # Normalize by the total number of observations
    output /= len(x)
    if verbose:
        print(output)
    
    # Function to compute Mutual Information for a given pair of values
    def compute_mi(join, marginals):
        return join * np.log2(join / reduce(lambda x, y: x * y, marginals))
    
    # Initialize list to store MI values
    total_mi = []
    
    # Extract the number of rows and columns (excluding the 'total' margin)
    rows = len(output.index) - 1
    cols = len(output.columns) - 1
    
    # Iterate through the contingency table (excluding margins) to calculate MI
    for r in range(rows):
        for c in range(cols):
            value = output.iloc[r, c]
            marg_a, marg_b = output.iloc[r, cols], output.iloc[rows, c]
            
            if verbose:
                print(value, [marg_a, marg_b], [r, c])
            
            if value != 0:
                mi_value = compute_mi(value, [marg_a, marg_b])
            else:
                mi_value = 0
            
            # Handle infinity values (e.g., log(0)) by setting them to zero
            if mi_value == np.inf:
                mi_value = 0
            
            total_mi.append(mi_value)
    
    if verbose:
        print(total_mi)
    
    # Return the total Mutual Information
    return np.sum(total_mi)


def mutual_information_frame(df, target_name):
    """
    Calculate Mutual Information (MI) scores for all features in a DataFrame relative to a target variable.
    
    The function discretizes numerical features into optimal bins before computing MI scores.
    Non-numerical (categorical) columns are ignored.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing features and the target variable.
    
    target_name : str
        Name of the target column in the DataFrame.
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with two columns:
        - 'features': Names of the feature columns.
        - 'mi_score': Corresponding mutual information scores with the target variable.
    
    Notes:
    ------
    - Numerical columns are discretized using an optimal binning strategy.
    - Categorical columns and the target column are excluded from the computation.
    
    Example:
    --------
    result = mutual_information_frame(data, target_name='Target')
    print(result)
    """

    # Copy input DataFrame to avoid modification
    cdata = df.copy()
    
    # Separate features and target
    temp_df = cdata.drop(target_name, axis=1)
    temp_y = cdata[target_name]
    temp_new = pd.DataFrame()
    
    # Identify categorical columns
    try:
        object_cols = cdata.describe(include='O').T.index.tolist()
    except Exception:
        object_cols = []
    
    # Identify numerical columns
    cols_categorized = [i for i in cdata.columns if i not in object_cols + [target_name]]
    
    # Bin numerical columns
    for j in cols_categorized:
        try:
            # Compute optimal number of bins and discretize
            nob = optimal_nobins(temp_df[j])['iqr']
            temp_new[j + '_bin'] = pd.qcut(temp_df[j], q=nob, duplicates='drop')
        except Exception as e:
            print(f"Error binning column '{j}': {e}")
    
    # Prepare features (X) and target (y)
    X, y = temp_new, temp_y
    
    # Initialize output dictionary
    out_dict = {'features': X.columns, 'mi_score': []}
    
    # Compute MI for each feature
    for i in X.columns:
        try:
            mi_score = mutual_information_value(X[i], y)
            out_dict['mi_score'].append(mi_score)
        except Exception as e:
            print(f"Error calculating MI for feature '{i}': {e}")
            out_dict['mi_score'].append(None)
    
    # Return results as a DataFrame
    return pd.DataFrame(out_dict)



def kl_divergence(p, q, probabilities=False, verbose=True):
    """
    Compute the Kullback-Leibler (KL) divergence between two distributions.

    KL divergence measures how one probability distribution (p) diverges 
    from a second reference probability distribution (q).

    Parameters:
    -----------
    p : array-like
        The first distribution or set of samples.

    q : array-like
        The second distribution or set of samples.

    probabilities : bool, optional (default=False)
        If True, `p` and `q` are treated as probabilities.
        If False, `p` and `q` are assumed to be samples and will be converted 
        into probability distributions.

    verbose : bool, optional (default=True)
        If True, prints intermediate steps and debugging information.

    Returns:
    --------
    float
        The KL divergence value.

    Notes:
    ------
    - KL divergence is not symmetric: KL(p || q) != KL(q || p).
    - The function handles zero probabilities by assigning zero contribution 
      to divergence in those cases.

    Example:
    --------
    # Example with probability inputs
    p = [0.4, 0.6]
    q = [0.5, 0.5]
    kl_divergence(p, q, probabilities=True)

    # Example with sample inputs
    p = [1, 2, 2, 3]
    q = [1, 1, 2, 3, 3]
    kl_divergence(p, q, probabilities=False)
    """
    import numpy as np

    if not probabilities:
        # Convert samples to probabilities
        labels_p = list(np.unique(p))
        labels_q = list(np.unique(q))

        # Combine labels
        final_labels = list(set(labels_p) | set(labels_q))
        
        # Compute probabilities
        p = compute_probabilities(p)
        q = compute_probabilities(q)

        # Ensure all labels are in both distributions
        for label in final_labels:
            p.setdefault(label, 0)
            q.setdefault(label, 0)

        if verbose:
            print(f"Probabilities (p): {p}")
            print(f"Probabilities (q): {q}")
            print(f"Labels: {final_labels}")

        # Compute KL divergence
        kl_divergence_measure = []
        for label in final_labels:
            if q[label] != 0:
                kl_divergence_measure.append(p[label] * np.log2(p[label] / q[label]))
            else:
                kl_divergence_measure.append(0)

    else:
        # Assume p and q are probability arrays
        p = np.array(p)
        q = np.array(q)

        # Calculate KL divergence
        kl_divergence_measure = p * np.log2(p / q)
        kl_divergence_measure[np.isinf(kl_divergence_measure)] = 0  # Handle division by zero

    if verbose:
        print(f"KL Divergence Components: {kl_divergence_measure}")

    return np.sum(kl_divergence_measure)


def plot_evaluation_curves(y_true, y_scores, title_pr='Precision-Recall Curve', title_roc='ROC Curve'):
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = auc(recall, precision)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Precision-Recall Curve
    axes[0].plot(recall, precision, color='blue',marker='*', label='Precision-Recall Curve', markersize=0.1)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'{title_pr} (AUC={average_precision:.2f})')
    axes[0].legend()

    # ROC Curve
    axes[1].plot(fpr, tpr, color='red', label='ROC Curve', marker='*',markersize=0.1)
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate (Recall)')
    axes[1].set_title(f'{title_roc} (AUC={roc_auc:.2f})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


