
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.metrics import roc_auc_score,  f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.size'] = 14  
plt.rcParams['axes.grid'] = False

######################## Parameters
seed=True
MODEL='CNN'
ANOM='F1'
s_size=144
t_size=1
w_size=144 
b_size=1000 
rate=0.0007
L2=0.0001
input_channels=21
num_filters=32
kernel_size=3
stride=2
dropout=0.5
hidden_size=100
num_epochs=15
num_layers=2
k = 4
b_it=50


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

if seed==True:
    torch.manual_seed(42) 
    torch.cuda.manual_seed_all(42)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    pass


###################Set GPU
device_id = 1
torch.cuda.set_device(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

##################Set data

print('Load Data')
dtypes = {'Set_temp': 'int8', 'Hvac_state': 'int8', 'Room_occupation': 'int8', 'Window': 'int8', 'Hvac_mode': 'int8', 'Hvac_state_manual': 'int8', 'Room_temp_up': 'int8', 'Room_temp_down': 'int8', 'Room_temp_cw': 'int8', 'Room_temp_ccw': 'int8', 'Room_temp': 'float32', 'Outside_temp': 'float32', 'Water_temp': 'int8', 'Nan_data': 'int8', 'is_detected': 'float32', 'Orientation_S': 'int8', 'Orientation_W': 'int8', 'Orientation_N': 'int8', 'Orientation_E': 'int8', 'FS_0': 'float32', 'FS_1': 'float32', 'FS_2': 'float32', 'FS_3': 'float32'}

#3 kata
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

#CHECK BALANCE
print('_______________________')
print('Original class balance:')
class_counts = tr['is_detected'].value_counts()
class_percentages = class_counts / len(tr) * 100
print('Train:\n',class_percentages)
class_counts = val['is_detected'].value_counts()
class_percentages = class_counts / len(val) * 100
print('Val:\n',class_percentages)
class_counts = te['is_detected'].value_counts()
class_percentages = class_counts / len(te) * 100
print('Test:\n',class_percentages)
print('_______________________')

class HvacDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        data.reset_index(inplace=True)
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
        window_data = self.scaler.transform(window.drop(columns=["is_detected"]))
        x = torch.from_numpy(window_data).permute(1, 0)
        
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

train_dataset = HvacDataset(tr) 
val_dataset=HvacDataset(val)
test_dataset=HvacDataset(te)


def check_data_balance(datasets):
    for dataset_name, dataset in datasets.items():
        class_counts = {0: 0, 1: 0}
        total_samples = len(dataset)

        for _, y in dataset:
            class_label = int(y.item())
            class_counts[class_label] += 1

        class_percentages = {class_label: count / total_samples * 100 for class_label, count in class_counts.items()}

        print(f"{dataset_name} Data:")
        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count} samples")

        for class_label, percentage in class_percentages.items():
            print(f"Class {class_label}: {percentage:.2f}% samples")
        print()

datasets = {
    'train_dataset': train_dataset,
    'val_dataset': val_dataset,
    'test_dataset': test_dataset
}

print('_______________________')
check_data_balance(datasets)
print('_______________________')



# Create dataloaders with undersampled datasets
train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=collate_fn_padd)
val_dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=False, collate_fn=collate_fn_padd)
test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, collate_fn=collate_fn_padd)


print('Optimize')
######################## Parameters
#BAYES    
from scipy.stats import uniform, randint, rv_discrete
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical, Dimension
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
num_epochs=15
input_size=21
num_classes=2
it=0

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

#chose model
if MODEL == 'LSTM':
    print(MODEL)
    print(ANOM)
    class Net(nn.Module):
        def __init__(self, input_size=21, hidden_size=hidden_size, output_size=1, num_layers=num_layers):
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

    model = Net().to(device)
    space = [Real(0.0001, 0.1, name='learning_rate'),
         Real(0.0001, 0.1, name='weight_decay'),
         Integer(64, 128, name='hidden_size'),  
         Integer(1, 5, name='num_layers')
         ]
    @use_named_args(space)
    def objective(weight_decay,learning_rate,hidden_size,num_layers):
        global it
        it += 1
        model = Net(hidden_size=hidden_size, num_layers=num_layers).to(device)
        print(f"\n Iteration:{it}, \n Model parameters:hidden_size:{hidden_size}, weight_decay:{weight_decay}, learning_rate={learning_rate:.4f},num_layers:{num_layers}")
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #############################
        best_val_acc = 0.0
        patience = 3
        count = 0
        
        train_losses=[]
        val_losses=[]
        
        vaccuracies=[]
        vprecisions=[]
        vrecalls=[]
        vf1_scores=[]
        vmean_auc=[]
        
        for epoch in range(num_epochs):
            train_loss, y_true_tr, y_pred_tr = train(model, optimizer, criterion, train_dataloader, device)
            val_loss, y_true_val, y_pred_val = evaluate(model, val_dataloader)
        
            #Val
            acc_val = accuracy_score(y_true_val, y_pred_val)
            auc_val = roc_auc_score(y_true_val, y_pred_val)
            vf1_test = f1_score(y_true_val, y_pred_val)
            vprecision_test = precision_score(y_true_val, y_pred_val, zero_division=1)
            vrecall_test = recall_score(y_true_val, y_pred_val,zero_division=1)
        
            vaccuracies.append(acc_val)
            vprecisions.append(vprecision_test)
            vrecalls.append(vrecall_test)
            vf1_scores.append(vf1_test)
            vmean_auc.append(auc_val)
               
            #Tr
            acc_train = accuracy_score(y_true_tr, y_pred_tr)
               
            #losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        
            print('Epoch [{}/{}], Train Loss: {:.2f}, Train Accuracy: {:.2f}, Val loss: {:.2f}, Val Accuracy: {:.2f}, AUC: {:.2f}'
                .format(epoch+1, num_epochs, train_loss, acc_train, val_loss, acc_val, auc_val))

            #Early stopping
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                count = 0
            else:
                count += 1
            if count == patience:
                print('No improvement in validation accuracy after {} epochs. Early stopping.'.format(patience))
                break
        
        # Return the average validation loss across all folds
        return val_loss
elif MODEL == 'CNN':
    print(MODEL)
    print(ANOM)
    rate=0.0001
    L2=0.0001

    class Net(nn.Module): 
        def __init__(self, input_channels=21, num_filters=num_filters, kernel_size=kernel_size,stride=stride):
            super(Net, self).__init__()
            self.conv_layer1 = nn.Conv1d(input_channels, num_filters, kernel_size)
            self.relu1 = nn.ReLU()
            self.max_pool1 = nn.MaxPool1d(kernel_size, stride)
            self.conv_layer2 = nn.Conv1d(num_filters, num_filters, kernel_size)
            self.relu2 = nn.ReLU()
            self.max_pool2 = nn.MaxPool1d(kernel_size, stride)
            self.fc1 = None

        def forward(self, x):
            out = self.conv_layer1(x)
            out = self.relu1(out) 
            out = self.max_pool1(out)
            out = self.conv_layer2(out)
            out = self.relu2(out) 
            out = self.max_pool2(out)
            if self.fc1 is None:
                output_length = out.size(2)
                value = out.size(1) * output_length
                self.fc1 = nn.Linear(value, 1).to(x.device)
            
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = torch.sigmoid(out)
            return out
    model = Net().to(device)
    space = [Real(0.0001, 0.1, name='learning_rate'),
        Real(0.0001, 0.1, name='weight_decay'),
        Integer(2, 3, name='kernel_size'), 
        Integer(36, 128, name='num_filters'), 
        Integer(2, 3, name='stride')
        ]
    @use_named_args(space)
    def objective(weight_decay,learning_rate, kernel_size,num_filters,stride):
        global it
        it += 1
        model = Net(kernel_size=kernel_size, num_filters=num_filters,stride=stride).to(device)
        print(f"\n Iteration:{it}, \n Model parameters: weight_decay:{weight_decay}, learning_rate={learning_rate:.4f}, kernel_size={kernel_size},num_filters:{num_filters},stride:{stride}")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
           
        #############################
        best_val_acc = 0.0
        patience = 3  
        count = 0  
        
        train_losses=[]
        val_losses=[]
        
        vaccuracies=[]
        vprecisions=[]
        vrecalls=[]
        vf1_scores=[]
        vmean_auc=[]
        
        for epoch in range(num_epochs):
            train_loss, y_true_tr, y_pred_tr = train(model, optimizer, criterion, train_dataloader, device)
            val_loss, y_true_val, y_pred_val = evaluate(model, val_dataloader)
        
            #Val
            acc_val = accuracy_score(y_true_val, y_pred_val)
            auc_val = roc_auc_score(y_true_val, y_pred_val)
            vf1_test = f1_score(y_true_val, y_pred_val)
            vprecision_test = precision_score(y_true_val, y_pred_val, zero_division=1)
            vrecall_test = recall_score(y_true_val, y_pred_val,zero_division=1)
        
            vaccuracies.append(acc_val)
            vprecisions.append(vprecision_test)
            vrecalls.append(vrecall_test)
            vf1_scores.append(vf1_test)
            vmean_auc.append(auc_val)
            #Tr
            acc_train = accuracy_score(y_true_tr, y_pred_tr)
            #losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        
            print('Epoch [{}/{}], Train Loss: {:.2f}, Train Accuracy: {:.2f}, Val loss: {:.2f}, Val Accuracy: {:.2f}, AUC: {:.2f}'
                .format(epoch+1, num_epochs, train_loss, acc_train, val_loss, acc_val, auc_val))

            #Early stopping
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                count = 0
            else:
                count += 1
            if count == patience:
                print('No improvement in validation accuracy after {} epochs. Early stopping.'.format(patience))
                break
        
        # Return the average validation loss across all folds
        return val_loss
else:
    print(MODEL)
    print(ANOM)
    rate=0.1
    L2=0.0001
    class Net(nn.Module):
        def __init__(self, input_channels=21, num_filters=num_filters, kernel_size=kernel_size, hidden_size=hidden_size, dropout=dropout, stride=stride):
            super(Net, self).__init__()
            self.conv_layer1 = nn.Conv1d(input_channels, num_filters, kernel_size)
            self.bn1 = nn.BatchNorm1d(num_filters)
            self.relu1 = nn.ReLU()
            self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=stride)
            self.conv_layer2 = nn.Conv1d(num_filters, num_filters, kernel_size)
            self.bn2 = nn.BatchNorm1d(num_filters)
            self.relu2 = nn.ReLU()
            self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=stride)
            self.gru = nn.GRU(num_filters, hidden_size, batch_first=True)  # Add a GRU layer
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
    model = Net().to(device)

    space = [Real(0.0001, 0.1, name='learning_rate'),
        Real(0.0001, 0.1, name='weight_decay'),
        Integer(64, 128, name='hidden_size'), 
        Integer(2, 3, name='kernel_size'),  
        Real(0.0, 0.5, name='dropout'),
        Integer(36, 128, name='num_filters'),  
        Integer(2, 3, name='stride')
        ]
    
    @use_named_args(space)
    def objective(weight_decay,learning_rate, kernel_size,num_filters,hidden_size,stride,dropout):
        global it
        it += 1
        model = Net(kernel_size=kernel_size, num_filters=num_filters,hidden_size=hidden_size,dropout=dropout,stride=stride).to(device)
        print(f"\n Iteration:{it}, \n Model parameters: dropout{dropout},hidden_size{hidden_size},weight_decay:{weight_decay}, learning_rate={learning_rate:.4f}, kernel_size={kernel_size},dropout: {dropout},num_filters:{num_filters},stride:{stride}")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
           
        #############################
        best_val_acc = 0.0
        patience = 3  
        count = 0
        
        train_losses=[]
        val_losses=[]
        
        vaccuracies=[]
        vprecisions=[]
        vrecalls=[]
        vf1_scores=[]
        vmean_auc=[]
        
        for epoch in range(num_epochs):
            train_loss, y_true_tr, y_pred_tr = train(model, optimizer, criterion, train_dataloader, device)
            val_loss, y_true_val, y_pred_val = evaluate(model, val_dataloader)
        
            #Val
            acc_val = accuracy_score(y_true_val, y_pred_val)
            auc_val = roc_auc_score(y_true_val, y_pred_val)
            vf1_test = f1_score(y_true_val, y_pred_val)
            vprecision_test = precision_score(y_true_val, y_pred_val, zero_division=1)
            vrecall_test = recall_score(y_true_val, y_pred_val,zero_division=1)
        
            vaccuracies.append(acc_val)
            vprecisions.append(vprecision_test)
            vrecalls.append(vrecall_test)
            vf1_scores.append(vf1_test)
            vmean_auc.append(auc_val)
               
            #Tr
            acc_train = accuracy_score(y_true_tr, y_pred_tr)
               
            #losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        
            print('Epoch [{}/{}], Train Loss: {:.2f}, Train Accuracy: {:.2f}, Val loss: {:.2f}, Val Accuracy: {:.2f}, AUC: {:.2f}'
                .format(epoch+1, num_epochs, train_loss, acc_train, val_loss, acc_val, auc_val))

            #Early stopping
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                count = 0
            else:
                count += 1
            if count == patience:
                print('No improvement in validation accuracy after {} epochs. Early stopping.'.format(patience))
                break
                   
        return val_loss


s=dt.now()
avg_val_loss = gp_minimize(objective, space, n_calls=b_it)
best_hyperparams = {space[i].name: avg_val_loss.x[i] for i in range(len(space))}
best_val_loss = avg_val_loss.fun

from skopt.plots import plot_convergence
plot_convergence(avg_val_loss)
print('_____________________________________')
print(f"Results for: {MODEL}, {ANOM}")
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Corresponding avg validation loss: {best_val_loss:.4f}")
e=dt.now()
running_secs=(e-s).seconds
running_mins=(e-s).total_seconds()/60
print('finished: %0.2f'% (running_secs), 's, %0.2f' % (running_mins),'min')
