
# In[ ]:   
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
import psutil
from sklearn.model_selection import KFold
from sklearn.metrics import auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
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
from sklearn.utils import resample
from sklearn.utils import shuffle
mpl.rcParams['font.size'] = 14  # Change the font size to 12
mpl.rcParams['axes.grid'] = False


dtypes = {'Set_temp': 'int8', 'Hvac_state': 'int8', 'Room_occupation': 'int8', 'Window': 'int8', 'Hvac_mode': 'int8', 'Hvac_state_manual': 'int8', 'Room_temp_up': 'int8', 'Room_temp_down': 'int8', 'Room_temp_cw': 'int8', 'Room_temp_ccw': 'int8', 'Room_temp': 'float32', 'Outside_temp': 'float32', 'Water_temp': 'int8', 'Nan_data': 'int8', 'is_detected': 'float32', 'Orientation_S': 'int8', 'Orientation_W': 'int8', 'Orientation_N': 'int8', 'Orientation_E': 'int8', 'FS_0': 'float32', 'FS_1': 'float32', 'FS_2': 'float32', 'FS_3': 'float32'}

print('Load')
window_size = 144
step_size =144             
ANOM='F3'


cols_c=['Set_temp', 'Room_temp_up', 'Room_temp_down', 
        'Room_temp_cw','Room_temp_ccw', 
        'Room_temp', 'Outside_temp', 'Humidity','Irradiation']

cols_b=['Hvac_state', 'Room_occupation', 
        'Window', 'Hvac_mode','Hvac_state_manual', 
        'is_detected', 'Orientation_S','Orientation_W', 
        'Orientation_N', 'Orientation_E','FS_0','FS_1','FS_2', 
        'FS_3']

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

t1=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R4_Y2019.csv', dtype=dtypes)
t2=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R5_Y2019.csv', dtype=dtypes)
t3=pd.read_csv(f'/home/imatetic/Project/14.6.Final_models/data/{ANOM}/R6_Y2019.csv', dtype=dtypes)
te=pd.concat([t1,t2,t3])

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
te_df=(stats_optimized(te)).dropna()

#CHECK
print('_______________________')
print('Class balance after undersampling:')
class_counts = train_df['is_detected_max'].value_counts()
class_percentages = class_counts / len(train_df) * 100
print('Train:\n',class_percentages, '\n',class_counts,'\n')
class_counts = val_df['is_detected_max'].value_counts()
class_percentages = class_counts / len(val_df) * 100
print('Val:\n',class_percentages, '\n',class_counts,'\n')
class_counts = te_df['is_detected_max'].value_counts()
class_percentages = class_counts / len(te_df) * 100
print('Val:\n',class_percentages, '\n',class_counts,'\n')



#SPLIT
X_train = train_df.drop('is_detected_max', axis=1)
Y_train = train_df['is_detected_max']
print(X_train.shape,Y_train.shape)
X_val = val_df.drop('is_detected_max', axis=1)
Y_val = val_df['is_detected_max']
print(X_val.shape,Y_val.shape)
X_test = te_df.drop('is_detected_max', axis=1)
Y_test = te_df['is_detected_max']
print(X_test.shape,Y_test.shape)


# Shuffle the training data
X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train, random_state=42)
#X_train_shuffled = X_train
#Y_train_shuffled = Y_train

t=0.5

model=RandomForestClassifier(
criterion='gini',
max_depth=20,
max_features= 'sqrt',
min_samples_leaf=1,
min_samples_split= 2,
n_estimators=1000,
random_state=42,
n_jobs=-1)

print(model.get_params())

s=dt.now()
#train
model.decision_threshold = t
#model.fit(X_train, Y_train)
model.fit(X_train_shuffled, Y_train_shuffled)
# VALIDATION
vpredictions = model.predict(X_val)
vy_true = Y_val
vy_pred = vpredictions
vauc = roc_auc_score(vy_true, vy_pred)
print(f'\nResults of VALIDATION set')
print('Accuracy: {:.2f}'.format(accuracy_score(vy_true, vpredictions)))
print('Precision: {:.2f}'.format(precision_score(vy_true, vpredictions)))
print('Recall: {:.2f}'.format(recall_score(vy_true, vpredictions)))
print('F1 Score: {:.2f}'.format(f1_score(vy_true, vpredictions)))
print('AUC: {:.2f}'.format(vauc))
e=dt.now()
running_secs=(e-s).seconds
running_mins=(e-s).total_seconds()/60
print('finished: %0.2f'% (running_secs), 's, %0.2f' % (running_mins),'min')
print('\n')


#average confusion matrix FIX
plt.figure() # create a new figure
cf_matrix = confusion_matrix(vy_true, vy_pred)
cf_matrix = cf_matrix.astype(np.float64)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap)
plt.title('Valdiation Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

vfpr_te, vtpr_te, _ = roc_curve(vy_true, vy_pred)
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(vfpr_te, vtpr_te, color='navy', lw=1, label='Test ROC curve (area = %0.2f)' % vauc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

#################################################################

# TEST
s=dt.now()
predictions = model.predict(X_test)
y_true = Y_test
y_pred = predictions
tauc = roc_auc_score(y_true, y_pred)
print(f'\nResults of TEST set')
print('Accuracy: {:.2f}'.format(accuracy_score(y_true, y_pred)))
print('Precision: {:.2f}'.format(precision_score(y_true, y_pred)))
print('Recall: {:.2f}'.format(recall_score(y_true, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(y_true, y_pred)))
print('AUC: {:.2f}'.format(tauc))
e=dt.now()
running_secs=(e-s).seconds
running_mins=(e-s).total_seconds()/60
print('finished: %0.2f'% (running_secs), 's, %0.2f' % (running_mins),'min')
print('\n')

#average confusion matrix FIX
plt.figure() # create a new figure
cf_matrix = confusion_matrix(y_true, y_pred)
cf_matrix = cf_matrix.astype(np.float64)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap)
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()



fpr_te, tpr_te, _ = roc_curve(y_true, y_pred)
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_te, tpr_te, color='navy', lw=1, label='Test ROC curve (area = %0.2f)' % tauc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()


#SAVE
x = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
x.to_csv(f'/home/imatetic/Project/14.6.Final_models/Preds/{ANOM}_RF_anomalies.csv')
