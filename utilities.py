import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.float_format',lambda x : '%.3f' % x)
pd.set_option('display.max_columns',10000)
# pd.set_option('display.max_colwidth',10000)

import numpy as np
np.set_printoptions(suppress=True,precision=5,floatmode='fixed')

import os
import pyodbc
import numpy as np
import sklearn as sk
import dateutil
import datetime
import itertools
import pytz
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing

def get_current_time_ist():
   ist = pytz.timezone('Asia/Kolkata')
   now_utc = datetime.now(pytz.utc)
   now_ist = now_utc.astimezone(ist)
   formatted_time = now_ist.strftime('%Y-%m-%d %I:%M:%S %p')
   output = str(formatted_time)
   replacement_dict = {':':'','-':'',' ':''}
   for old,new in replacement_dict.items():
       output = output.replace(old,new)
   return output

col_names = lambda tbl : list(execute('describe {}'.format(tbl)).col_name)
def execute(query):
   conn = pyodbc.connect('DSN=HIVEProd',autocommit= True)
   print('QUERY : {}'.format(query))
   out = pd.read_sql(query,conn)
   out.columns = [i.split('.')[1] if '.' in i else i for i in out.columns]
   return out

def create_table(query):
   def shape(table_name,query=False):
       y = len(col_names(table_name))
       x = execute('select count(*) from {}'.format(table_name)).iloc[0][0]
       return (x,y)
   table_name = query.lower().split('create table ')[1].split(' ')[0]
   print('TABLE NAME : {}'.format(table_name))
   print('QUERY:')
   print('\n{}'.format(query))
   con = pyodbc.connect('DSN=HiveProd', autocommit=True)
   cur = con.cursor()
   cur.execute('drop table if exists '+table_name)
   cur.execute(query)
   print('SHAPE : {}'.format(str(shape(table_name))))
   cur.close()

def execute_sql(query):
   con = pyodbc.connect('DSN=HiveProd', autocommit=True)
   cur = con.cursor()
   cur.execute(query)
   print('Execution Completed')
   cur.close()

def last_day(date):
   next_month = date.replace(day=28)+datetime.timedelta(days=4)
   return next_month -datetime.timedelta(days=next_month.day)

def get_yyyymm(dt):
   leads_rpt_dt_yyyy = str(last_day(dt).year)
   leads_rpt_dt_mm = str(last_day(dt).month)
   leads_rpt_dt_mm = '0'+leads_rpt_dt_mm if len(leads_rpt_dt_mm)==1 else leads_rpt_dt_mm
   mmyyyy = leads_rpt_dt_mm+leads_rpt_dt_yyyy
   yyyymm = leads_rpt_dt_yyyy+leads_rpt_dt_mm
   return mmyyyy,yyyymm

def get_previous_months(date_str,n):
   date = datetime.strptime(date_str,'%Y-%m-%d')
   end_dates = []
   for _ in range(n):
       date = date - relativedelta(months=1)
       end_of_month = date.replace(day=1)+relativedelta(day=31)
       end_dates.append(end_of_month.strftime('%Y-%m-%d'))
   return tuple(end_dates)

def col_names(tbl):
   return list(execute('describe {}'.format(tbl)).col_name)

def shape(table_name,query=False):
   y = len(col_names(table_name))
   x = execute("select count(*) from {}".format(table_name)).iloc[0][0]
   return (x,y)

def lag_n(input_date,n=-1):
   '''returns the last of the nth lag of input date'''
   return last_day(input_date+dateutil.relativedelta.relativedelta(months=n))


date_current = datetime.date.today()
date_lag = lag_n(date_current,n=-1)
month_name_current = date_current.strftime('%B')
month_name_lag = date_lag.strftime('%B')
year_current = date_current.year
year_lag = date_lag.year
# yyyymm format current month and previous_month
mmyyyy_current,yyyymm_current = get_yyyymm(date_current)
mmyyyy_lag,yyyymm_lag = get_yyyymm(date_lag)

print('date current = {}'.format(date_current))
print('date lag (-1)  = {}'.format(date_lag))
print('month name current = {}'.format(month_name_current))
print('month name lag  = {}'.format(month_name_lag))
print('year current = {}'.format(year_current))
print('year_lag = {}'.format(year_lag))
print('mmyyyy_current = {}'.format(mmyyyy_current))
print('yyyymm_current = {}'.format(yyyymm_current))
print('mmyyyy_lag = {}'.format(mmyyyy_lag))
print('yyyymm_lag = {}'.format(yyyymm_lag))


def load_data(query,chunksize):
   con = pyodbc.connect("DSN=HiveProd", autocommit= True)       
   dd = pd.read_sql(query,con,chunksize=chunksize)
   cnt=0
   df_chunk = pd.DataFrame()
   for chunk in dd:
       cnt=cnt+1
       chunk=chunk.replace(r'^\s*$',np.nan,regex=True)
       print(chunk.shape)
       df_chunk=df_chunk.append(chunk)
       del chunk
   df_chunk.columns = [i.split('.')[1] if '.' in i else i for i in df_chunk.columns]
   return df_chunk

def row_trend(x):
   dd = np.asarray(x)
   if len(dd) !=1:
       denum = (len(dd)-1)*len(dd)/2
       ddcopy = dd[:]
       addall = 0
       suball = 0
       for p in range(0,len(ddcopy)-1):
           element = dd[0]
           dd = dd[1:]
           check_eql = sum(dd == element)
           subtr = len(dd) - sum(dd > element) - check_eql
           addr = len(dd) - subtr - check_eql
           suball = suball + subtr
           addall = addall + addr
       return (-suball + addall)/denum
   else:
       return 0

def plot_heatmap(data):
   plt.figure(figsize = (15,7))
   feat_num= data.shape[0]
   corrMatrix = data.corr().abs().round(2)
   sn.heatmap(corrMatrix,annot=True)
   plt.show()

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
               "Feature1_Variance": variance_1,
               "Feature2_Variance": variance_2,
               "Correlation": corr_matrix.loc[feature_1, feature_2],
               "Dropping_Feature": to_drop,
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