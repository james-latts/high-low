#### IMPORTS & GLOBALS ####
# Imports
import os
import time
import torch
import numpy as np
import pandas as pd
import datetime as dt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Globals
#FEATURE_COLUMNS = ["Change", "Hour", "Minute"]
FEATURE_COLUMNS = ["Hour", "Minute", "Price", "Last_Price", "Change", \
                   "10_sma", "25_sma", "50_sma", "100_sma", "200_sma", \
                   "10_ema", "25_ema", "50_ema", "100_ema", "200_ema"]


#### MAIN ####



#### FUNCTIONS ####
def get_x(row, df, window_size, feature_list):
    x = []
    for f in feature_list:
        values = df[f][row-window_size:row].values
        x.append(values)
    x = np.stack(x, axis = 1)
    return x

def get_y(row, df, window_size, feature_list):
    y = np.sum(df["Change"][row:row+window_size].values)
    y = 1 if y > 0 else 0
    return y

def train(epoch, train_dataloader):
    model.train()
    total_loss = 0.
    total_correct = 0
    total_pred = []
    for batch_number, batch_data in enumerate(train_dataloader):
        features, targets = batch_data
        features = features.to(device)
        targets = targets.view((-1, 1)).to(device).double()
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        pred = torch.sigmoid(output).cpu().detach().numpy()
        total_correct += np.sum(np.round(pred) == targets.cpu().detach().numpy())
        total_pred += pred.tolist()
    print('| epoch {:3d} | loss {:5.4f} | acc {:5.2f}'.format(
            epoch, total_loss / len(total_pred) * 100, total_correct / len(train_dataset) * 100))

def evaluate(eval_model, dataloader):
    eval_model.eval()
    total_loss = 0.
    total_correct = 0
    total_pred = []
    with torch.no_grad():
        for batch_number, batch_data in enumerate(dataloader):
            features, targets = batch_data
            features = features.to(device)
            targets = targets.view((-1, 1)).to(device).double()
            output = eval_model(features)
            total_loss += criterion(output, targets).item()
            pred = torch.sigmoid(output).cpu().detach().numpy()
            total_correct += np.sum(np.round(pred) == targets.cpu().detach().numpy())
            total_pred += pred.tolist()
    return total_loss, total_correct, total_pred


#### CLASSES ####
class TimeSeriesDataset(Dataset):
    """ Time Series Dataset """
    def __init__(self, data, transform=None):
        """
        Args:
            data (string): The time series data as a pandas dataframe.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]["x"]
        y = self.data.iloc[idx]["y"]
        if self.transform:
            x = self.transform(x)
        return x[0], y

class TimeSeriesModel(nn.Module):
    def __init__(self, classes, nlen, ninp, nhid, nlayers, dropout, bi):
        super(TimeSeriesModel, self).__init__()
        self.model_type = 'TimeSeries'
        self.bi = bi
        self.nlen = nlen
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = nn.Dropout(dropout)
        #self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout = dropout, batch_first = True, bidirectional = bi)
        self.linear = nn.Linear(ninp * nlen, 10)
        self.classify = nn.Linear(10, classes)

    def forward(self, features):
        #features = self.dropout(features)
        #features = self.lstm(features)
        #features = features[0].reshape((-1, (1 + bi) * self.nhid * self.nlen))
        features = self.dropout(features)
        features = self.linear(features)
        features = self.dropout(features)
        output = self.classify(features)
        return output


#### RUN MAIN ####
#
xjo_data_location = "/home/james/Documents/finance/data-acquisition/bar_chart/minute_data"
original_data_df = pd.DataFrame()
for xjo_csv_file in os.listdir(xjo_data_location):
    xjo_csv_location = xjo_data_location + "/" + xjo_csv_file
    csv_data_df = pd.read_csv(xjo_csv_location)
    original_data_df = pd.concat([original_data_df, csv_data_df])

#
data_df = pd.DataFrame()
data_df["DateTime"] = pd.to_datetime(original_data_df["Time"])
#data_df['Date'] = data_df['DateTime'].dt.date
data_df['Time'] = data_df['DateTime'].dt.time
data_df['Hour'] = data_df['DateTime'].dt.hour
data_df['Minute'] = data_df['DateTime'].dt.minute
data_df["Price"] = original_data_df["Open"]
data_df = data_df.sort_values("DateTime")


#
data_df = data_df[data_df["Time"] <= dt.time(16, 0, 0)]
data_df = data_df[data_df["Time"] >= dt.time(11, 0, 0)]
last_price_df = data_df[["Price"]].shift(1)
last_price_df.columns = ["Last_Price"]
data_df = pd.concat([data_df, last_price_df], axis = 1).dropna()
data_df["Change"] = ((data_df["Price"] - data_df["Last_Price"]) / data_df["Last_Price"]) * 100
data_df["10_sma"] = data_df["Price"].rolling(window=10).mean()
data_df["25_sma"] = data_df["Price"].rolling(window=25).mean()
data_df["50_sma"] = data_df["Price"].rolling(window=50).mean()
data_df["100_sma"] = data_df["Price"].rolling(window=100).mean()
data_df["200_sma"] = data_df["Price"].rolling(window=200).mean()
data_df["10_ema"] = data_df["Price"].ewm(span=10).mean()
data_df["25_ema"] = data_df["Price"].ewm(span=25).mean()
data_df["50_ema"] = data_df["Price"].ewm(span=50).mean()
data_df["100_ema"] = data_df["Price"].ewm(span=100).mean()
data_df["200_ema"] = data_df["Price"].ewm(span=200).mean()
data_df = data_df[["DateTime"] + FEATURE_COLUMNS].dropna()

x_window_size = 1
y_window_size = 15
batch_size = 500

test_data = data_df.query('20201101 <= DateTime').reset_index()
test_data_new = test_data.iloc[x_window_size:len(test_data) - y_window_size, :]
test_data_new["xIndex"] = test_data_new.index
test_data_new["x"] = test_data_new["xIndex"].apply(lambda x: get_x(x, test_data, x_window_size, FEATURE_COLUMNS))
test_data_new["y"] = test_data_new["xIndex"].apply(lambda x: get_y(x, test_data, y_window_size, FEATURE_COLUMNS))
test_dataset = TimeSeriesDataset(test_data_new[["x", "y"]])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
val_data = data_df.query('20200901 <= DateTime < 20201101').reset_index()
val_data_new = val_data.iloc[x_window_size:len(val_data) - y_window_size, :]
val_data_new["xIndex"] = val_data_new.index
val_data_new["x"] = val_data_new["xIndex"].apply(lambda x: get_x(x, val_data, x_window_size, FEATURE_COLUMNS))
val_data_new["y"] = val_data_new["xIndex"].apply(lambda x: get_y(x, val_data, y_window_size, FEATURE_COLUMNS))
val_dataset = TimeSeriesDataset(val_data_new[["x", "y"]])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

train_data = data_df.query('20200901 > DateTime').reset_index()
train_data_new = train_data.iloc[x_window_size:len(train_data) - y_window_size, :]
train_data_new["xIndex"] = train_data_new.index
train_data_new["x"] = train_data_new["xIndex"].apply(lambda x: get_x(x, train_data, x_window_size, FEATURE_COLUMNS))
train_data_new["y"] = train_data_new["xIndex"].apply(lambda x: get_y(x, train_data, y_window_size, FEATURE_COLUMNS))
train_dataset = TimeSeriesDataset(train_data_new[["x", "y"]])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = 1
ninp = len(FEATURE_COLUMNS)
nlen = x_window_size
nhid = 100
nlayers = 5
dropout = 0.5
bi = True
model = TimeSeriesModel(classes, nlen, ninp, nhid, nlayers, dropout, bi).to(device).double()

weight = (len(train_data_new) - sum(train_data_new["y"])) / sum(train_data_new["y"])

criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(weight), reduction='sum')
optimizer = torch.optim.Adam(model.parameters())

best_val_loss = float("inf")
best_val_acc = float(0)
epochs = 100 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(epoch, train_dataloader)
    val_loss, val_acc, val_pred = evaluate(model, val_dataloader)
    print('-' * 89)
    print('| end of epoch {:3d} | val loss {:5.4f} | val_acc {:5.2f} '.format(epoch, 
        val_loss / len(val_dataset) * 100, val_acc / len(val_dataset) * 100))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_val_pred = val_pred
        best_model = model

    optimizer.step()
    
test_loss, test_acc, test_pred = evaluate(best_model, test_dataloader)
print('=' * 89)
print('| End of training | test loss {:5.4f} | test acc {:5.2f}'.format(
    test_loss /len(test_dataset) * 100, test_acc / len(test_dataset) * 100))
print('| Best val epoch {:3d} | val loss {:5.4f} | val acc {:5.2f}'.format(
    best_epoch, best_val_loss /len(val_dataset) * 100, best_val_acc / len(val_dataset) * 100))
print('=' * 89)

from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(val_data_new["y"], best_val_pred, pos_label=1)
auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic val')
plt.legend(loc="lower right")
plt.show()

gmeans = np.sqrt(tpr * (1-fpr))
optimal_threshold = thresholds[np.argmax(gmeans)]
pred = np.where(np.array(best_val_pred) >= optimal_threshold, 1, 0)
acc = np.sum(pred == val_data_new["y"].values.reshape((-1,1))) / len(val_data_new["y"])
print('best val acc {:5.2f}'.format(acc * 100))

fpr, tpr, thresholds = metrics.roc_curve(test_data_new["y"], test_pred, pos_label=1)
auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic test')
plt.legend(loc="lower right")
plt.show()

gmeans = np.sqrt(tpr * (1-fpr))
optimal_threshold = thresholds[np.argmax(gmeans)]
pred = np.where(np.array(test_pred) >= optimal_threshold, 1, 0)
acc = np.sum(pred == test_data_new["y"].values.reshape((-1,1))) / len(test_data_new["y"])
print('best test acc {:5.2f}'.format(acc * 100))