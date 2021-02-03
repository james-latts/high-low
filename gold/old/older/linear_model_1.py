#### IMPORTS & GLOBALS ####
# Imports
import os
import torch
import numpy as np
import pandas as pd
import datetime as dt
import torch.nn as nn
from sklearn import metrics
from itertools import product
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Globals
FEATURES = ["Sma", "Std", "Max", "Min", "Ema", "Srsi", "Rsi", "So"]
N = [7, 14, 21]
N = [x + 1 for x in range(1, 250)]
FEATURE_COLUMNS = list(map("_".join, product([str(n) for n in N], FEATURES)))
FEATURE_COLUMNS += ["MACD"]


#### MAIN ####



#### FUNCTIONS ####
def compute_rsi(data, time_window):
    # diff in one field(one day)
    diff = data.diff(1)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

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
    def __init__(self, X, Y, transform=None):
        """
        Args:
            data (string): The time series data as a pandas dataframe.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class TimeSeriesModel(nn.Module):
    def __init__(self, classes, ninp, nhid, dropout):
        super(TimeSeriesModel, self).__init__()
        self.model_type = 'TimeSeries'
        self.ninp = ninp
        self.nhid = nhid
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(ninp, nhid)
        self.linear_2 = nn.Linear(nhid, nhid // 2)
        self.linear_3 = nn.Linear(nhid // 2, nhid // 4)
        self.linear_4 = nn.Linear(nhid // 4, nhid // 8)
        self.linear_5 = nn.Linear(nhid // 8, nhid // 16)
        self.linear_6 = nn.Linear(nhid // 16, nhid // 32)
        self.linear_7 = nn.Linear(nhid // 32, nhid // 64)
        self.classify = nn.Linear(nhid // 64, classes)

    def forward(self, features):
        features = self.dropout(features)
        features = self.linear_1(features)
        features = self.dropout(features)
        features = self.linear_2(features)
        features = self.dropout(features)
        features = self.linear_3(features)
        features = self.dropout(features)
        features = self.linear_4(features)
        features = self.dropout(features)
        features = self.linear_5(features)
        features = self.dropout(features)
        features = self.linear_6(features)
        features = self.dropout(features)
        features = self.linear_7(features)
        features = self.dropout(features)
        output = self.classify(features)
        return output
#
xjo_data_location = "/home/james/Documents/finance/data-acquisition/bar_chart/15_minute_data"
original_data_df = pd.DataFrame()
for xjo_csv_file in os.listdir(xjo_data_location):
    xjo_csv_location = xjo_data_location + "/" + xjo_csv_file
    csv_data_df = pd.read_csv(xjo_csv_location)
    original_data_df = pd.concat([original_data_df, csv_data_df])
del xjo_data_location, csv_data_df

#
data_df = pd.DataFrame()
data_df["DateTime"] = pd.to_datetime(original_data_df["Time"])
data_df['Time'] = data_df['DateTime'].dt.time
data_df['Hour'] = data_df['DateTime'].dt.hour
data_df['Minute'] = data_df['DateTime'].dt.minute
data_df["Price"] = original_data_df["Last"]
data_df["High"] = original_data_df["High"]
data_df["Low"] = original_data_df["Low"]

# Cleaning
data_df = data_df[data_df["Time"] <= dt.time(16, 0, 0)]
data_df = data_df[data_df["Time"] >= dt.time(11, 0, 0)]

#
data_df = data_df.sort_values("DateTime").reset_index(drop = True)
last_price_df = data_df[["Price"]].shift(-1)
last_price_df.columns = ["Future_Price"]
data_df = pd.concat([data_df, last_price_df], axis = 1).dropna()

#
data_df["Change"] = data_df["Future_Price"] - data_df["Price"]
data_df["y"] = (data_df["Change"] > 0).astype(np.int)

# Simple moving average
for n in N:
    data_df[str(n) + "_Sma"] = (data_df["Price"].rolling(window=n).mean() / data_df["Price"] - 1) * 100

# Rolling standard deviation
for n in N:
    data_df[str(n) + "_Std"] = data_df["Price"].rolling(window=n).std()

# Rolling max
for n in N:
    data_df[str(n) + "_Max"] = (data_df["High"].rolling(window=n).max() / data_df["Price"] - 1) * 100
    
# Rolling min
for n in N:
    data_df[str(n) + "_Min"] = (data_df["Low"].rolling(window=n).min() / data_df["Price"] - 1) * 100

# Exponential moving average
for n in N:
    data_df[str(n) + "_Ema"] = (data_df["Price"].ewm(span=n).mean() / data_df["Price"] - 1) * 100

# MACD
for n in N:
    EMA12 = data_df["Price"].ewm(span=12).mean()
    EMA26 = data_df["Price"].ewm(span=26).mean()
    MACD = EMA12 - EMA26
    SignalLine = MACD.ewm(span=9).mean()
    data_df["MACD"] = np.where(MACD.values > SignalLine.values, 1, 0)

# Relative strength index
for n in N:
    data_df[str(n) + '_Rsi'] = compute_rsi(data_df["Price"], n)

# Stochastic relative strength index
for n in N:
    data_df[str(n) + '_Srsi'] = \
        (data_df[str(n) + "_Rsi"] - data_df[str(n) + "_Rsi"].rolling(window=n).min()) / \
        (data_df[str(n) + "_Rsi"].rolling(window=n).max() - data_df[str(n) + "_Rsi"].rolling(window=n).min())

# Stochastic oscilator
for n in N:
    data_df[str(n) + '_So'] = 100 * \
        (data_df["Price"] - data_df["Low"].rolling(window=n).min()) / \
        (data_df["High"].rolling(window=n).max() - data_df["Low"].rolling(window=n).min())

# Williams %R
for n in N:
    data_df[str(n) + '_Wr'] = 100 * \
        (data_df["High"].rolling(window=n).min() - data_df["Price"]) / \
        (data_df["High"].rolling(window=n).max() - data_df["Low"].rolling(window=n).min())


#
batch_size = 5000
train_data = data_df.query('20190101 > DateTime').dropna().reset_index(drop = True)
X = train_data[FEATURE_COLUMNS].values
y = train_data["y"].values
train_dataset = TimeSeriesDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  

#
val_data = data_df.query('20200101 > DateTime > 20190101').dropna().reset_index(drop = True)
X_v = val_data[FEATURE_COLUMNS].values
y_v = val_data["y"].values
val_dataset = TimeSeriesDataset(X_v, y_v)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = 1
ninp = len(FEATURE_COLUMNS)
nhid = 512
dropout = 0.5
model = TimeSeriesModel(classes, ninp, nhid, dropout).to(device).double()

weight = (len(train_data) - sum(train_data["y"])) / sum(train_data["y"])

criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(weight), reduction='sum')
optimizer = torch.optim.Adam(model.parameters())

best_val_loss = float("inf")
best_val_acc = float(0)
epochs = 1000 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
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
    
test_data = data_df.query('20200101 <= DateTime').dropna().reset_index(drop = True)
X_t = test_data[FEATURE_COLUMNS].values
y_t = test_data["y"].values
test_dataset = TimeSeriesDataset(X_t, y_t)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  

test_loss, test_acc, y_t_prob = evaluate(best_model, test_dataloader)
print('=' * 89)
print('| End of training | test loss {:5.4f} | test acc {:5.2f}'.format(
    test_loss /len(test_dataset) * 100, test_acc / len(test_dataset) * 100))
print('| Best val epoch {:3d} | val loss {:5.4f} | val acc {:5.2f}'.format(
    best_epoch, best_val_loss /len(val_dataset) * 100, best_val_acc / len(val_dataset) * 100))
print('=' * 89)

y_t_prob = np.array(y_t_prob).reshape((-1))
fpr, tpr, thresholds = metrics.roc_curve(y_t, y_t_prob, pos_label=1)
auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % auc)
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
pred = np.where(np.array(y_t_prob) >= optimal_threshold, 1, 0)
acc = np.sum(pred == y_t) / len(y_t)
print('\nBest test acc: {:5.2f}'.format(acc * 100))
del gmeans, lw, plt

report = metrics.classification_report(y_t, pred)
print("\nClassification report:\n", report)
