import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


#Postavljanje GPU-a za runanje modela
device_id = 1
torch.cuda.set_device(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Definirana struktura modela
rate=0.007123381892567762
L2= 0.0001
class Net(nn.Module):
    def __init__(self, input_channels=19, num_filters=109, 
                    kernel_size=2, hidden_size=90, 
                    dropout=  0.5, stride=2):
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


# Loadanje modela
M1 = torch.load('best_model-F1-CNNGRU.pt')
M2 = torch.load('best_model-F2-CNNGRU.pt')
M3 = torch.load('best_model-F3-CNNGRU.pt')

#Definirane značajki
cols=['Set_temp', 'Hvac_state', 'Room_occupation', 'Window', 'Hvac_mode', 'Hvac_state_manual', 
      'Room_temp_up','Room_temp_down', 'Room_temp_cw', 'Room_temp_ccw','Room_temp', 'Orientation_S',
      'Orientation_W', 'Orientation_N', 'Orientation_E','FS_0', 'FS_1','FS_2', 'FS_3']

#Preprocesiranje podataka
class HvacDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[cols]
        self.window_size = 96
        self.step_size = 96
        self.scaler = StandardScaler()
        self.scaler.fit(self.data)
        self.windows = self.generate_windows()

    def generate_windows(self):
        windows = []
        for i in range(0, len(self.data) - self.window_size + 1, self.step_size):
            window = self.data[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def __getitem__(self, index):
        window = self.windows[index]
        window_data = self.scaler.transform(window)
        x = torch.from_numpy(window_data).permute(1, 0)
        return x

    def __len__(self):
        return len(self.windows)

#Glavna funkcija koju pozivate
def detect(df,M1,M2,M3):
    #Preprocesiranje podataka
    x=HvacDataset(df)

    # Dohvaćanje prozora
    data_loader = torch.utils.data.DataLoader(x, batch_size=1)
    preds=[]
    for inputs in data_loader:
        inputs = inputs.float().to(device)

        # Rezultati modela
        output1 = M1(inputs)
        output2 = M2(inputs)
        output3 = M3(inputs)

        # Dobivanje vjerojatnosti rezultata 
        p1= torch.sigmoid(output1)
        p2= torch.sigmoid(output2)
        p3= torch.sigmoid(output3)
        
        #Dodavanje vjerojatnosti u numpy listu
        preds.append(p1.cpu().detach().numpy())
        preds.append(p2.cpu().detach().numpy())
        preds.append(p3.cpu().detach().numpy())

    #Transformacija podataka u listu
    p = np.ravel(preds).tolist()
    return p
