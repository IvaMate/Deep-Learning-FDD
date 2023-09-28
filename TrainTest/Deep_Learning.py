# DL run

# In[ ]:   #LIBS and PARAMS
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from torch.utils.data import DataLoader,random_split, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix,roc_curve,auc, roc_auc_score,  f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import KFold
from itertools import chain
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
mpl.rcParams['font.size'] = 14  
mpl.rcParams['axes.grid'] = False

######################## Parameters
s_size=144
t_size=1
w_size=144 
b_size=1000 
num_epochs=30
ANOM='F3'
MODEL='LSTM'
print(ANOM, MODEL)
seed=True
###################Set GPU
device_id = 1
torch.cuda.set_device(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

if seed==True:
    torch.manual_seed(42)  # Set the random seed for CPU
    torch.cuda.manual_seed_all(42)  # Set the random seed for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    pass


Cols=['Datetime', 'Set_temp', 'Hvac_state', 'Room_occupation', 'Window',
       'Hvac_mode', 'Hvac_state_manual', 'Room_temp_up', 'Room_temp_down',
       'Room_temp_cw', 'Room_temp_ccw', 'Room_temp', 'Outside_temp',
       'Irradiation', 'Orientation_S',
       'Orientation_W', 'Orientation_N', 'Orientation_E', 'FS_0', 'FS_1',
       'FS_2', 'FS_3', 'is_detected']
Cols2=['Set_temp', 'Hvac_state', 'Room_occupation', 'Window',
       'Hvac_mode', 'Hvac_state_manual', 'Room_temp_up', 'Room_temp_down',
       'Room_temp_cw', 'Room_temp_ccw', 'Room_temp', 'Outside_temp',
       'Irradiation',  'Orientation_S',
       'Orientation_W', 'Orientation_N', 'Orientation_E', 'FS_0', 'FS_1',
       'FS_2', 'FS_3', 'is_detected']
Cols3=['Set_temp', 'Hvac_state', 'Room_occupation', 'Window',
       'Hvac_mode', 'Hvac_state_manual', 'Room_temp_up', 'Room_temp_down',
       'Room_temp_cw', 'Room_temp_ccw', 'Room_temp', 'Outside_temp',
       'Irradiation',  'Orientation_S',
       'Orientation_W', 'Orientation_N', 'Orientation_E', 'FS_0', 'FS_1',
       'FS_2', 'FS_3', 'is_detected']
dtypes = {'Set_temp': 'int8', 'Hvac_state': 'int8', 'Room_occupation': 'int8', 'Window': 'int8', 'Hvac_mode': 'int8', 'Hvac_state_manual': 'int8', 'Room_temp_up': 'int8', 'Room_temp_down': 'int8', 'Room_temp_cw': 'int8', 'Room_temp_ccw': 'int8', 'Room_temp': 'float32', 'Outside_temp': 'float32', 'Water_temp': 'int8', 'Nan_data': 'int8', 'is_detected': 'float32', 'Orientation_S': 'int8', 'Orientation_W': 'int8', 'Orientation_N': 'int8', 'Orientation_E': 'int8', 'FS_0': 'float32', 'FS_1': 'float32', 'FS_2': 'float32', 'FS_3': 'float32'}


print('Load Data')
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

#Preprocessing

class HvacDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[Cols2]
        self.window_size = w_size
        self.step_size = s_size
        self.t_value = 1
        self.scaler = StandardScaler()
        self.scaler.fit(self.data.drop(columns=["is_detected"]))
        self.windows, self.targets = self.generate_windows_and_targets()

    def generate_windows_and_targets(self):
        windows = []
        targets = []
        for i in range(0, len(self.data) - self.window_size + 1, self.step_size):
            window = self.data[i:i + self.window_size]
            target = window["is_detected"].values
            windows.append(window)
            targets.append(target)
        return windows, targets
    
    def __getitem__(self, index):
        window = self.windows[index]
        target = self.targets[index]
        #window_data = window.drop(columns=["is_detected"]).values
        #scaler
        window_data = self.scaler.transform(window.drop(columns=["is_detected"]))
        x = torch.from_numpy(window_data).permute(1, 0)
        
        if target.sum() >= self.t_value:
            y = torch.tensor([1], dtype=torch.float32)
        else:
            y = torch.tensor([0], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.windows)

class For_RF(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[Cols3]
        self.window_size = w_size
        self.step_size = s_size
        self.t_value = 1
        #self.scaler = StandardScaler()
        #self.scaler.fit(self.data.drop(columns=["is_detected"]))
        self.windows, self.targets = self.generate_windows_and_targets()

    def generate_windows_and_targets(self):
        windows = []
        targets = []
        for i in range(0, len(self.data) - self.window_size + 1, self.step_size):
            window = self.data[i:i + self.window_size]
            target = window["is_detected"].values
            windows.append(window)
            targets.append(target)
        return windows, targets
    
    def __getitem__(self, index):
        window = self.windows[index]
        target = self.targets[index]
        window_data = window.drop(columns=["is_detected"]).values
        #scaler
        #window_data = self.scaler.transform(window.drop(columns=["is_detected"]))
        x = torch.from_numpy(window_data)#.permute(1, 0)
        
        if target.sum() >= self.t_value:
            y = torch.tensor([1], dtype=torch.float32)
        else:
            y = torch.tensor([0], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.windows)

def collate_fn_padd(batch,w_size=w_size):
    max_len = w_size
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [len(x) for x in inputs]
    padded_inputs = []
    for x in inputs:
        if x.shape[1] == max_len:
            padded_inputs.append(x)
        else:
            pad = torch.zeros((x.shape[0], max_len - x.shape[1]))
            padded_x = torch.cat([x, pad], dim=1)
            padded_inputs.append(padded_x)

    return torch.stack(padded_inputs), torch.stack(targets)

# Create datasets
train_dataset = HvacDataset(tr) 
val_dataset = HvacDataset(val)
test_dataset = HvacDataset(te)

#train_dataset2 = For_RF(tr) 
#val_dataset2 = For_RF(val)
#test_dataset2 = For_RF(te)

train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=collate_fn_padd)
val_dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=False, collate_fn=collate_fn_padd)
test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, collate_fn=collate_fn_padd)

def check_data_balance(datasets):
    for dataset_name, dataset in datasets.items():
        # Initialize counters for each class
        class_counts = {0: 0, 1: 0}
        total_samples = len(dataset)

        # Iterate over the dataset
        for _, y in dataset:
            class_label = int(y.item())
            class_counts[class_label] += 1

        # Calculate percentages
        class_percentages = {class_label: count / total_samples * 100 for class_label, count in class_counts.items()}

        # Print the class counts
        print(f"{dataset_name} Data:")
        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count} samples")

        # Print the class percentages
        for class_label, percentage in class_percentages.items():
            print(f"Class {class_label}: {percentage:.2f}% samples")
        print()

datasets = {
    'train_dataset': train_dataset,
    'val_dataset': val_dataset,
    'test_dataset': test_dataset,
    #'RF_train_dataset': train_dataset2,
    #'RF_val_dataset': val_dataset2,
    #'RF_test_dataset': test_dataset2
}

print('_______________________')
check_data_balance(datasets)
print('_______________________')

#  CHECK LENGTH

#print('Original dataset shape: ',tr.shape,val.shape,te.shape)
#print('For DL: ',len(train_dataset),len(val_dataset),len(test_dataset))
#print('For RF: ',len(train_dataset2),len(val_dataset2),len(test_dataset2))
#result = train_dataset.__getitem__(0)
#tensor_shape = result[0].shape if isinstance(result[0], torch.Tensor) else None
#print("Tensor shape:", tensor_shape)


# In[ ]:##################Set model


if MODEL == 'LSTM':
#Best hyperparameters: {'learning_rate': 0.017935854871712148, 
# 'weight_decay': 0.00231525425602806, 'hidden_size': 64, 'num_layers': 1}
#Best hyperparameters: {'learning_rate': 0.024983187788319143, 'weight_decay': 0.0001, 
# 'hidden_size': 66, 'num_layers': 2}
#Best hyperparameters: {'learning_rate': 0.049323808483915176, 'weight_decay': 0.0001, 
# 'hidden_size': 64, 'num_layers': 1}

    print(MODEL)
    rate=0.017935854871712148
    L2=0.00231525425602806
    class Net(nn.Module):
        def __init__(self, input_size=21, hidden_size=64, output_size=1, num_layers=1):
            super(Net, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            x = x.permute(0, 2, 1)
            
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.fc(out)
            out = torch.sigmoid(out)
            return out
    
    criterion = nn.BCELoss()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=rate, weight_decay=L2) # Add weight decay to the optimizer
    
elif MODEL == 'CNN':
    print(MODEL)
#Best hyperparameters: {'learning_rate': 0.011490006504672167, 
# 'weight_decay': 0.0009607758717223296, 'kernel_size': 2, 'num_filters': 93, 
# 'stride': 3}
#Best hyperparameters: {'learning_rate': 0.0203661724429349,
#  'weight_decay': 0.0001, 'kernel_size': 2, 'num_filters': 104, 'stride': 3}
#Best hyperparameters: {'learning_rate': 0.001, 'weight_decay': 0.0001, 'kernel_size': 3, 
# 'num_filters': 128, 'stride': 3}

    rate=0.001
    L2=0.0001
    class Net(nn.Module): 
        def __init__(self, input_channels=21, num_filters=128, kernel_size=3,stride=3):
            super(Net, self).__init__()
            self.conv_layer1 = nn.Conv1d(input_channels, num_filters, kernel_size)
            self.relu1 = nn.ReLU()
            self.max_pool1 = nn.MaxPool1d(kernel_size, stride)
            self.conv_layer2 = nn.Conv1d(num_filters, num_filters, kernel_size)
            self.relu2 = nn.ReLU()
            self.max_pool2 = nn.MaxPool1d(kernel_size, stride)
            self.fc1 = nn.Linear(num_filters * 15, 1).to(device)
        def forward(self, x):
            out = self.conv_layer1(x)
            out = self.relu1(out) 
            out = self.max_pool1(out)
            out = self.conv_layer2(out) 
            out = self.relu2(out)
            out = self.max_pool2(out) 
            out = out.view(out.size(0), -1) 
            out = self.fc1(out)
            out = torch.sigmoid(out)
            return out
    criterion = nn.BCELoss()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=rate, weight_decay=L2) # Add weight decay to the optimizer
else:
#Best hyperparameters: {'learning_rate': 0.007123381892567762, 'weight_decay': 0.0001, 'hidden_size': 90, 
# 'kernel_size': 2, 'dropout': 0.3209398868251996, 
# 'num_filters': 109, 'stride': 2}
#Best hyperparameters: {'learning_rate': 0.015553367468250548, 'weight_decay': 0.0001, 'hidden_size': 71,
#  'kernel_size': 2, 'dropout': 0.5, 'num_filters': 36, 'stride': 2}
#Best hyperparameters: {'learning_rate': 0.016987355956073754, 'weight_decay': 0.0001, 'hidden_size': 64,
#  'kernel_size': 2, 'dropout': 0.5, 'num_filters': 128, 'stride': 3}

    print(MODEL)
    rate=0.016987355956073754
    L2= 0.0001
    class Net(nn.Module):
        def __init__(self, input_channels=21, num_filters=128, 
                     kernel_size=2, hidden_size=64, 
                     dropout=  0.5, stride=3):
            super(Net, self).__init__()
            self.conv_layer1 = nn.Conv1d(input_channels, num_filters, kernel_size)
            self.bn1 = nn.BatchNorm1d(num_filters)
            self.relu1 = nn.ReLU()
            self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=stride)
            self.conv_layer2 = nn.Conv1d(num_filters, num_filters, kernel_size)
            self.bn2 = nn.BatchNorm1d(num_filters)
            self.relu2 = nn.ReLU()
            self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=stride)
            self.gru = nn.GRU(num_filters, hidden_size, batch_first=True) 
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out = self.conv_layer1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.max_pool1(out)
            out = self.conv_layer2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.max_pool2(out)
            out = out.permute(0, 2, 1) 
            out, _ = self.gru(out)  
            out = out[:, -1, :]  
            out = self.dropout(out)
            out = self.fc2(out)
            out = torch.sigmoid(out)
            return out
    criterion = nn.BCELoss()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=rate, weight_decay=L2)


##################Train-Val-Test
def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            X_batch, y_batch = data
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch.float())
            loss = F.binary_cross_entropy(outputs, y_batch.float())
            running_loss += loss.item()
            y_pred += torch.round(outputs.squeeze()).cpu().tolist()
            y_true += y_batch.cpu().numpy().tolist()
    loss = running_loss / len(dataloader)
    return loss, y_true, y_pred

def train(model, optimizer, criterion, train_dataloader, device):
    model.train()
    train_loss = 0
    y_true = []
    y_pred = []
    for i, data in enumerate(train_dataloader):
        X_batch, y_batch = data
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch.float())
        loss = criterion(outputs, y_batch.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred += torch.round(outputs.squeeze()).cpu().tolist()
        y_true += y_batch.cpu().numpy().tolist()
    total_loss = train_loss / len(train_dataloader)
    return total_loss, y_true, y_pred


train_losses=[]
val_losses=[]
lfpr_val=[]
ltpr_val=[]
best_val_acc = 0.0
patience = 3 
count = 0  
best_val_loss=100
best_auc = 0.0
best_model_path = f'/home/imatetic/Project/14.6.Final_models/Models/best_model-{ANOM}-{MODEL}.pt'
s=dt.now()
for epoch in range(num_epochs):
    train_loss, y_true_tr, y_pred_tr = train(model, optimizer, criterion, train_dataloader, device)
    val_loss, y_true_val, y_pred_val = evaluate(model, val_dataloader)
    
    auc_val = roc_auc_score(y_true_val, y_pred_val)
    acc_val = accuracy_score(y_true_val, y_pred_val)
    vf1_test = f1_score(y_true_val, y_pred_val)
    vprecision_test = precision_score(y_true_val, y_pred_val, zero_division=1)
    vrecall_test = recall_score(y_true_val, y_pred_val, zero_division=1)
    acc_train = accuracy_score(y_true_tr, y_pred_tr)

    # Losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        count = 0
        torch.save(model.state_dict(), best_model_path)
        print('Best model saved. AUC {:.2f} Validation loss: {:.2f}'.format(auc_val, best_val_loss))
    else:
        count += 1
        print(count)

    if count == patience:
        print('No improvement in validation loss after {} epochs. Early stopping.'.format(patience))
        break

    print('Epoch [{}/{}], Train Loss: {:.2f}, Train Accuracy: {:.2f}, Val loss: {:.2f}, Val Accuracy: {:.2f}, AUC: {:.2f}'
          .format(epoch + 1, num_epochs, train_loss, acc_train, val_loss, acc_val, auc_val))



e=dt.now()
running_secs=(e-s).seconds
running_mins=(e-s).total_seconds()/60
print('finished: %0.2f'% (running_secs), 's, %0.2f' % (running_mins),'min')
print('\n')

#Plot loss curve
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

#Confusion matrix - Test
plt.figure() # create a new figure
cf_matrix = confusion_matrix(y_true_val, y_pred_val)
cf_matrix = cf_matrix.astype(np.float64)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap)
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Print overall accuracy, mean, and standard deviation of accuracies
print('________________________')
print('\nOverall Val metrics:')
print('Accuracy: %0.2f ' % (acc_val))
print('Precision: %0.2f ' % (vprecision_test))
print('Recall: %0.2f ' % (vrecall_test))
print('F1 score: %0.2f ' % (vf1_test))
print('AUC: %0.2f '% (auc_val))
print('________________________')

s=dt.now()
print('test')
best_model = Net().to(device)
best_model.load_state_dict(torch.load(best_model_path))
#TEST
test_loss, y_true_te, y_pred_te = evaluate(best_model, test_dataloader)
#metrics
acc_test = accuracy_score(y_true_te, y_pred_te)
auc_test = roc_auc_score(y_true_te, y_pred_te)
f1_test = f1_score(y_true_te, y_pred_te)
precision_test = precision_score(y_true_te, y_pred_te)
recall_test = recall_score(y_true_te, y_pred_te)
fpr_te, tpr_te, _ = roc_curve(y_true_te, y_pred_te)
print('________________________')
print('\nOverall test metrics:')
print('Accuracy: %0.2f ' % (acc_test))
print('Precision: %0.2f  '% (precision_test))
print('Recall: %0.2f ' % (recall_test))
print('F1 score: %0.2f ' % (f1_test))
print('AUC: %0.2f '% (auc_test))
print('________________________')
e=dt.now()
running_secs=(e-s).seconds
running_mins=(e-s).total_seconds()/60
print('finished: %0.2f'% (running_secs), 's, %0.2f' % (running_mins),'min')
print('\n')

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_te, tpr_te, color='navy', lw=2, label='Test ROC curve (area = %0.2f)' % auc_test)
#plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 0.1])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

#Confusion matrix - Test
plt.figure() # create a new figure
cf_matrix = confusion_matrix(y_true_te, y_pred_te)
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


flattened_list = [item for sublist in y_true_te for item in sublist]
x = pd.DataFrame({'True': flattened_list})
yy= pd.DataFrame({'Predicted': y_pred_te})
print(x.shape,yy.shape)

# SAVE ALL results
flattened_list = [item for sublist in y_true_te for item in sublist]
x = pd.DataFrame({'True': flattened_list, 'Predicted': y_pred_te})
x.to_csv(f'/home/imatetic/Project/14.6.Final_models/Preds/{ANOM}_{MODEL}.csv')
x

# %%
best_model_path
# %%
