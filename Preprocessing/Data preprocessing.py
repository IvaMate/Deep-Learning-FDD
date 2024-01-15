import os
import re
import pyarrow.parquet as pq
import pyarrow
#import fastparquet
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
import psutil
from sklearn.model_selection import KFold
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score

#choose columns to use
cols_c=['Set_temp', 'Room_temp_up', 'Room_temp_down', 
        'Room_temp_cw','Room_temp_ccw', 
        'Room_temp', 'Outside_temp', 'Humidity','Irradiation', 'Water_temp']

cols_b=['is_weekday','Hvac_state', 'Room_occupation', 
        'Window', 'Hvac_mode','Hvac_state_manual', 
        'is_detected', 'Orientation_S','Orientation_W', 
        'Orientation_N', 'Orientation_E','FS_0','FS_1','FS_2', 
        'FS_3']

window_size = 144
step_size =24  
S = int(step_size)           



# In[ ]: 
#1.	Cleaning per room (parquet to csv)
print('**CLEANING THE DATA**')

def clean_data(df):
    #Renaming
    df.index.name = 'Datetime'
    df.rename(columns={'SET_TEMP': 'Set_temp', 'HVAC_STATE': 'Hvac_state','ROOM_OCC': 'Room_occupation','WINDOW': 'Window',
                       'MODE': 'Hvac_mode','MANUAL': 'Hvac_state_manual','TEMP_UP': 'Room_temp_up','TEMP_DOWN': 'Room_temp_down',
                       'TEMP_CW': 'Room_temp_cw','TEMP_CCW': 'Room_temp_ccw','T_AIR_ANOM': 'Room_temp','FAN_SPEED_ANOM': 'Fan_speed',
                       'ENERGY_ANOM': 'Energy_consumption','RH': 'Humidity','T_WATER': 'Water_temp','ENV_TEMP': 'Outside_temp',
                       'IRRAD': 'Irradiation','DETECTED_ANOM': 'is_detected'}, inplace=True)
    #One Hot Encoding
    df.loc[df['Fan_speed'] ==0, 'FS_0'] = 1
    df.loc[df['Fan_speed'] ==1, 'FS_1'] = 1
    df.loc[df['Fan_speed'] ==2, 'FS_2'] = 1
    df.loc[df['Fan_speed'] ==3, 'FS_3'] = 1
    df.loc[df['Fan_speed'] >0, 'FS_0'] = 0
    df.loc[df['Fan_speed'] !=1, 'FS_1'] = 0
    df.loc[df['Fan_speed'] !=2, 'FS_2'] = 0
    df.loc[df['Fan_speed'] !=3, 'FS_3'] = 0
    #Removing
    
    df.drop(columns=['Fan_speed','FAN_SPEED_GOOD', 'T_AIR_GOOD','ENERGY_GOOD','IRRAD_ANG','INSERT_valve', 
                     'INSERT_fan1','INSERT_fanStop','INSERT_win'], inplace=True)
    return df

South = '.*_0_.*'
West = '.*_90_.*'
North = '.*_180_.*'
East = '.*_270_.*'
Y1 = '.*Y2015.*'
Y2 = '.*Y2016.*'
Y3 = '.*Y2017.*'
Y4 = '.*Y2018.*'
Y5 = '.*Y2019.*'
Y6 = '.*Y2021.*'

for j in range(1,5):
    print('Clean F:',j)
    path = f'/home/imatetic/Project/Data/Ex/sim'
    parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]

    for i, parquet_file in enumerate(parquet_files):
        df = pd.read_parquet(os.path.join(path, parquet_file))
        var = re.search("(Y\d{4}_R\d+)", parquet_file).group(1)
        if re.match(South, parquet_file):
            df['Orientation_S'] = 1
            df['Orientation_W'] = 0
            df['Orientation_N'] = 0
            df['Orientation_E'] = 0
        elif re.match(West, parquet_file):
            df['Orientation_S'] = 0
            df['Orientation_W'] = 1
            df['Orientation_N'] = 0
            df['Orientation_E'] = 0
        elif re.match(North, parquet_file):
            df['Orientation_S'] = 0
            df['Orientation_W'] = 0
            df['Orientation_N'] = 1
            df['Orientation_E'] = 0
        elif re.match(East, parquet_file):
            df['Orientation_S'] = 0
            df['Orientation_W'] = 0
            df['Orientation_N'] = 0
            df['Orientation_E'] = 1
            
        #CLEAN DATA
        df_cleaned= clean_data(df) 
        globals()[f'df_'+var] = df_cleaned
        x=globals()[f'df_'+var]
        x.to_csv(f'/home/imatetic/Project/Data/Ex/Rooms/F{j}/df_'+var+'.csv')

# In[ ]: 
#2. Merging the rooms
print('**MERGING ROOMS**')

for j in range(1,5):
    print('Merge F:',j)
    path = f'/home/imatetic/Project/Data/Ex/Rooms/F{j}'
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    Y1 = '.*_Y2015_*'
    Y2 = '.*_Y2016_*'
    Y3 = '.*_Y2017_*'
    Y4 = '.*_Y2018_*'
    Y5 = '.*_Y2019_*'
    Y6 = '.*_Y2021_*'
    l_15=[]
    l_16=[]
    l_17=[]
    l_18=[]
    l_19=[]
    l_21=[]
    #Organize to lists
    for i, file in enumerate(files):
        df = pd.read_csv(os.path.join(path, file), engine='python',skipfooter=1)
        if re.match(Y1, file):
            l_15.append(df)
        elif re.match(Y2, file):
            l_16.append(df)
        elif re.match(Y3, file):
            l_17.append(df)
        elif re.match(Y4, file):
            l_18.append(df)
        elif re.match(Y5, file):
            l_19.append(df)
        elif re.match(Y6, file):
            l_21.append(df)
    #MERGE
    ll_15=pd.concat(l_15)
    ll_16=pd.concat(l_16)
    ll_17=pd.concat(l_17)
    ll_18=pd.concat(l_18)
    ll_19=pd.concat(l_19)
    ll_21=pd.concat(l_21)
    #Export
    ll_15.to_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{j}/Y2015.csv')
    ll_16.to_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{j}/Y2016.csv')
    ll_17.to_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{j}/Y2017.csv')
    ll_18.to_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{j}/Y2018.csv')
    ll_19.to_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{j}/Y2019.csv')
    ll_21.to_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{j}/Y2021.csv')

# In[ ]: 
#3. Preprocessing sliding window
print('**SLIDING WINDOW PREPROCESSING**')
print('Window width: ', window_size)
print('Window step: ', step_size)

#is_weekday column creation
def new_cols(df):
    print('*New df*')
    print('new cols:')
    before=psutil.virtual_memory()
    s=dt.now()
    #Vremenski stupci
    df.Datetime = pd.to_datetime(df.Datetime)
    df['is_weekday']=0
    df['weekday']=df.Datetime.dt.dayofweek
    df.loc[df['weekday']>=5, 'is_weekday']=1
    df.loc[df['weekday']<5, 'is_weekday']=0
    df=df.drop(columns=['weekday'])
    
    #delete rows
    df.drop(df[df['Set_temp']==0].index,inplace=True)
    df.drop(df[df['Nan_data']==1].index,inplace=True)
    df.drop(columns=['Nan_data'], inplace=True)
    
    #change dtype
    df['Room_temp']=df['Room_temp'].astype(np.int8)
    df['is_weekday']=df['is_weekday'].astype(np.int8)
    
    #set old index
    df=df.rename(columns={'Unnamed: 0':'i'})
    df.set_index('i', inplace=True)
    print(df.shape)
    after=psutil.virtual_memory()
    ram= after.used-before.used #Bytes
    print('used memory: ', int(ram/(1024*1024*1024)),'GB')
    e=dt.now()
    running_mins=(e-s).total_seconds()/60 
    print('Time: %0.2f' % (running_mins),'min')
    return df
#Skip indexes that aren't monotonic
def skip_windows(df):
    print('Skip:')
    before=psutil.virtual_memory()
    s=dt.now()
    
    l = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df[i:i + window_size + 1]
        
        if np.all(np.diff(window.index) ==True) :
            l.append(window)
        else:
            continue
            
    
    new_df = pd.concat(l)
    new_df= new_df.reset_index(drop=True)
    print(new_df.shape)
    after=psutil.virtual_memory()
    ram= after.used-before.used #Bytes
    print('used memory: ', int(ram/(1024*1024*1024)),'GB')
    e=dt.now()
    running_mins=(e-s).total_seconds()/60 
    print('Time: %0.2f' % (running_mins),'min')
    return new_df
#Count groups of ones in a window
def groups(window, j):
    count = 0
    detected = False
    groups = 0
    
    for i in window:
        prev_count = count
        count +=i
        if count > prev_count and not detected:
            groups += 1
            detected = True
            
        if count == prev_count:
            detected = False            
    return groups
#Final columns
def stats_optimized(df):
    print('stat:')
    before=psutil.virtual_memory()
    s=dt.now()
    
    new_df = pd.DataFrame()
    for col in cols_c:
        new_df[f'{col}_min'] = df[col].rolling(window_size, step=step_size).min().astype(np.float32)
        new_df[f'{col}_max'] = df[col].rolling(window_size, step=step_size).max().astype(np.float32)
        new_df[f'{col}_sum'] = df[col].rolling(window_size, step=step_size).sum().astype(np.float32)
        new_df[f'{col}_mean'] = df[col].rolling(window_size, step=step_size).mean().astype(np.float32)
        new_df[f'{col}_median'] = df[col].rolling(window_size, step=step_size).median().astype(np.float32)
        new_df[f'{col}_std'] = df[col].rolling(window_size, step=step_size).std().astype(np.float32)
    for col in (cols_b):
        new_df[f'{col}_sum'] = df[col].rolling(window_size, step=step_size).sum().astype(np.float32)
        new_df[f'{col}_max'] = df[col].rolling(window_size, step=step_size).max().astype(np.float32)
        new_df[f'{col}_groups'] = df[col].rolling(window_size, step=step_size).apply(lambda x: groups(x, col)).astype(np.float32)

    new_df.drop(columns=['is_detected_sum','is_detected_groups'], inplace=True)    
    print(new_df.shape)
    after=psutil.virtual_memory()
    ram= after.used-before.used #Bytes
    print('used memory: ', int(ram/(1024*1024*1024)),'GB')
    e=dt.now()
    running_mins=(e-s).total_seconds()/60 
    print('Time: %0.2f' % (running_mins),'min')
    return new_df

dtypes = {'Set_temp': 'int8',  'Hvac_state': 'int8',  'Room_occupation': 'int8',  'Window': 'int8',  'Hvac_mode': 'int8',  
          'Hvac_state_manual': 'int8',  'Room_temp_up': 'int8',  'Room_temp_down': 'int8',  'Room_temp_cw': 'int8',  
          'Room_temp_ccw': 'int8',  'Room_temp': 'float32',  'Energy_consumption' : 'float32',  'Outside_temp': 'float32',  
          'Humidity': 'float32',  'Water_temp': 'float32',  'Irradiation' : 'float32',  'Nan_data': 'int8',  'is_detected': 'int8',  
          'Orientation_S': 'int8',  'Orientation_W': 'int8',  'Orientation_N': 'int8',  'Orientation_E': 'int8',  'FS_0': 'int8',  
          'FS_1': 'int8',  'FS_2':'int8',  'FS_3': 'int8'} 
i=0
for i in range(1, 4):
    print('sliding window preprocessing: F:',i)
    # Read data
    df15 = list(map(lambda n: stats_optimized(skip_windows(new_cols(pd.read_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{i}/Y{n}.csv', dtype=dtypes)))), [2015]))
    df16 = list(map(lambda n: stats_optimized(skip_windows(new_cols(pd.read_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{i}/Y{n}.csv', dtype=dtypes)))), [2016]))
    df17 = list(map(lambda n: stats_optimized(skip_windows(new_cols(pd.read_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{i}/Y{n}.csv', dtype=dtypes)))), [2017]))
    df18 = list(map(lambda n: stats_optimized(skip_windows(new_cols(pd.read_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{i}/Y{n}.csv', dtype=dtypes)))), [2018]))
    df19 = list(map(lambda n: stats_optimized(skip_windows(new_cols(pd.read_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{i}/Y{n}.csv', dtype=dtypes)))), [2019]))
    df21 = list(map(lambda n: stats_optimized(skip_windows(new_cols(pd.read_csv(f'/home/imatetic/Project/Data/Ex/Merge/F{i}/Y{n}.csv', dtype=dtypes)))), [2021]))

    # Drop NaN values
    df15 = list(map(lambda n: n.dropna(), df15))
    df16 = list(map(lambda n: n.dropna(), df16))
    df17 = list(map(lambda n: n.dropna(), df17))
    df18 = list(map(lambda n: n.dropna(), df18))
    df19 = list(map(lambda n: n.dropna(), df19))
    df21 = list(map(lambda n: n.dropna(), df21))
    
    # Concatenate data and export to CSV
    pd.concat(df15).to_csv(f'/home/imatetic/Project/Data/Ex/Export24/F{i}/Y2015.csv')
    pd.concat(df16).to_csv(f'/home/imatetic/Project/Data/Ex/Export24/F{i}/Y2016.csv')
    pd.concat(df17).to_csv(f'/home/imatetic/Project/Data/Ex/Export24/F{i}/Y2017.csv')
    pd.concat(df18).to_csv(f'/home/imatetic/Project/Data/Ex/Export24/F{i}/Y2018.csv')
    pd.concat(df19).to_csv(f'/home/imatetic/Project/Data/Ex/Export24/F{i}/Y2019.csv')
    pd.concat(df21).to_csv(f'/home/imatetic/Project/Data/Ex/Export24/F{i}/Y2021.csv')
