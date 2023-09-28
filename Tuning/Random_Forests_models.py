
# In[ ]:   
#RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve,auc, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
import psutil
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_score
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, accuracy_score
pd.set_option("display.max_columns", None)
from sklearn.model_selection import  KFold
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_fscore_support
import matplotlib as mpl
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
mpl.rcParams['font.size'] = 14  # Change the font size to 12
mpl.rcParams['axes.grid'] = False
y_true = []
y_pred = []

window_size = 144
step_size =144             
ANOM='F3'
iterations=50
print('RF data preprocessing', ANOM)

cols_c=['Set_temp', 'Room_temp_up', 'Room_temp_down', 
        'Room_temp_cw','Room_temp_ccw', 
        'Room_temp', 'Outside_temp', 'Humidity','Irradiation']

cols_b=['Hvac_state', 'Room_occupation', 
        'Window', 'Hvac_mode','Hvac_state_manual', 
        'is_detected', 'FS_0','FS_1','FS_2', 
        'FS_3','Orientation_S', 'Orientation_W', 'Orientation_N', 'Orientation_E']

dtypes = {'Set_temp': 'int8',  'Hvac_state': 'int8',  'Room_occupation': 'int8',  'Window': 'int8',  'Hvac_mode': 'int8',  
          'Hvac_state_manual': 'int8',  'Room_temp_up': 'int8',  'Room_temp_down': 'int8',  'Room_temp_cw': 'int8',  
          'Room_temp_ccw': 'int8',  'Room_temp': 'float32',  'Energy_consumption' : 'float32',  'Outside_temp': 'float32',  
          'Humidity': 'float32',  'Water_temp': 'float32',  'Irradiation' : 'float32',  'Nan_data': 'int8',  'is_detected': 'int8',  
          'Orientation_S': 'int8',  'Orientation_W': 'int8',  'Orientation_N': 'int8',  'Orientation_E': 'int8',  'FS_0': 'int8',  
          'FS_1': 'int8',  'FS_2':'int8',  'FS_3': 'int8'} 


tr1=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R4_Y2017.csv', dtype=dtypes)
tr2=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R5_Y2017.csv', dtype=dtypes)
tr3=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R6_Y2017.csv', dtype=dtypes)
tr=pd.concat([tr1,tr2,tr3])

v1=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R4_Y2018.csv', dtype=dtypes)
v2=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R5_Y2018.csv', dtype=dtypes)
v3=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R6_Y2018.csv', dtype=dtypes)
val=pd.concat([v1,v2,v3])

#CHECK
print('_______________________')
print('Original cleaned class balance:')
class_counts = tr['is_detected'].value_counts()
class_percentages = class_counts / len(tr) * 100
print('Train:\n',class_percentages, '\n',class_counts,'\n')
class_counts = val['is_detected'].value_counts()
class_percentages = class_counts / len(val) * 100
print('Val:\n',class_percentages, '\n',class_counts,'\n')


def groups(window):
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

def stats_optimized(df):
    new_df = pd.DataFrame()
    for col in cols_c:
        new_df[f'{col}_min'] = df[col].rolling(window=window_size).min().shift(-(window_size - 1))[::step_size].astype(np.float32)
        new_df[f'{col}_max'] = df[col].rolling(window=window_size).max().shift(-(window_size - 1))[::step_size].astype(np.float32)
        new_df[f'{col}_sum'] = df[col].rolling(window=window_size).sum().shift(-(window_size - 1))[::step_size].astype(np.float32)
        new_df[f'{col}_mean'] = df[col].rolling(window=window_size).mean().shift(-(window_size - 1))[::step_size].astype(np.float32)
        new_df[f'{col}_median'] = df[col].rolling(window=window_size).median().shift(-(window_size - 1))[::step_size].astype(np.float32)
        new_df[f'{col}_std'] = df[col].rolling(window=window_size).std().shift(-(window_size - 1))[::step_size].astype(np.float32)

    for col in (cols_b):
        #new_df[f'{col}_sum'] = df[col].rolling(window_size, step=step_size).sum().shift(-(window_size - 1))[::step_size].astype(np.float32)
        new_df[f'{col}_max'] = df[col].rolling(window=window_size).max().shift(-(window_size - 1))[::step_size].astype(np.float32)
#        new_df[f'{col}_groups'] = df[col].rolling(window=window_size).max().shift(-(window_size - 1))[::step_size].apply(lambda x: groups(x))
        new_df[f'{col}_groups'] = df[col].rolling(window=window_size).max().shift(-(window_size - 1))[::step_size].apply(lambda x: groups([x]))


    new_df.drop(columns=['is_detected_groups'], inplace=True)    
    return new_df

train_df=(stats_optimized(tr)).dropna()
val_df=(stats_optimized(val)).dropna()

#CHECK
print('_______________________')
print('Class balance after undersampling:')
class_counts = train_df['is_detected_max'].value_counts()
class_percentages = class_counts / len(train_df) * 100
print('Train:\n',class_percentages, '\n',class_counts,'\n')
class_counts = val_df['is_detected_max'].value_counts()
class_percentages = class_counts / len(val_df) * 100
print('Val:\n',class_percentages, '\n',class_counts,'\n')


X_train = train_df.drop('is_detected_max', axis=1)
X_val = val_df.drop('is_detected_max', axis=1)
Y_train = train_df['is_detected_max']
Y_val = val_df['is_detected_max']
# Shuffle the training data
X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train, random_state=42)
# Initialize lists to store hyperparameters and scores
hyperparams = []
scores = []


print('Optimize_new')
# Define the hyperparameter search space
search_space = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 20),
    'max_features': Categorical(['sqrt', 'log2']),
    'min_samples_split': Integer(2, 100),
    'min_samples_leaf': Integer(1, 10),
    'criterion':Categorical(['gini','entropy']),
    #'max_samples': Integer(1, 10)
}
# Define the model with default hyperparameters
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)
# Define the search object with Bayesian optimization
search = BayesSearchCV(
    model,
    search_space,
    n_iter=iterations,  # <--- number of iterations
    cv=None,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy'
)


#total_iterations
for i in range(search.n_iter):
    #search.fit(X_train, Y_train)
    search.fit(X_train_shuffled, Y_train_shuffled)
    best_params = search.best_params_
    y_pred = search.predict(X_val)
    accuracy = accuracy_score(Y_val, y_pred)
    auc = roc_auc_score(Y_val, y_pred)
    print('Iteration {}:'.format(i + 1))
    print('    Hyperparameters: {}'.format(best_params))
    print('    Accuracy: {:.2f}'.format(accuracy))
    print('    AUC: {:.2f}'.format(auc))
    hyperparams.append(best_params)
    scores.append(auc)
# Print the best hyperparameters found by the search
best_index = scores.index(max(scores))
print('Best hyperparameters: {}'.format(hyperparams[best_index]))


# %%
