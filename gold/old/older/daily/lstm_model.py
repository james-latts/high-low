#### IMPORTS & GLOBALS ####
# Imports
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Globals
COLUMNS = ["Time", "Open", "High", "Low", "Last", "Change"]
FEATURE_COLUMNS = ["axjo_Change", "axjo_High", "axjo_Low", "axjo_Diff", \
                   "audusd_Change", "audusd_High", "audusd_Low", "audusd_Diff", \
                   "spx_Change", "spx_High", "spx_Low", "spx_Diff"]


#### MAIN ####



#### FUNCTIONS ####
def get_data(row, df, window_size, feature_list):
    x = []
    for f in feature_list:
        values = df[f][row-window_size:row].values
        x.append(values)
    x = np.stack(x, axis = 1)
    return x

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
        return x, y

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
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout = dropout, batch_first = True, bidirectional = bi)
        self.classify = nn.Linear((1 + bi) * nhid * nlen, classes)

    def forward(self, features):
        features = self.dropout(features)
        features = self.lstm(features)
        features = features[0].reshape((-1, (1 + bi) * self.nhid * self.nlen))
        features = self.dropout(features)
        output = self.classify(features)
        return output


#### RUN MAIN ####
axjo_data_location = "/home/james/Documents/finance/data-acquisition/bar_chart/axjo_1-1-2000_12-20-2020.csv"
axjo_data_df = pd.read_csv(axjo_data_location)
axjo_data_df["Time"] = pd.to_datetime(axjo_data_df["Time"])
axjo_data_df = axjo_data_df.drop("Volume", axis = 1)
axjo_data_df.columns = ["axjo_" + x for x in COLUMNS]

audusd_data_location = "/home/james/Documents/finance/data-acquisition/bar_chart/audusd_1-1-2000_12-20-2020.csv"
audusd_data_df = pd.read_csv(audusd_data_location)
audusd_data_df["Time"] = pd.to_datetime(audusd_data_df["Time"])
audusd_data_df = audusd_data_df.query('20130712 <= Time').reset_index(drop = True)
audusd_data_df = audusd_data_df.drop("Volume", axis = 1)
audusd_data_df.columns = ["audusd_" + x for x in COLUMNS]

spx_data_location = "/home/james/Documents/finance/data-acquisition/bar_chart/spx_1-1-2000_12-20-2020.csv"
spx_data_df = pd.read_csv(spx_data_location)
spx_data_df["Time"] = pd.to_datetime(spx_data_df["Time"])
spx_data_df = spx_data_df.query('20130712 <= Time').reset_index(drop = True)
spx_data_df = spx_data_df.drop("Volume", axis = 1)
spx_data_df.columns = ["spx_" + x for x in COLUMNS]

data_df = pd.merge(axjo_data_df, audusd_data_df, left_on = "axjo_Time", right_on = "audusd_Time")
data_df = pd.merge(data_df, spx_data_df, left_on = "axjo_Time", right_on = "spx_Time")

# Custom logic
data_df["axjo_Change"] = data_df["axjo_Change"] / data_df["axjo_Open"] * 100
data_df["axjo_High"] = data_df["axjo_High"] / data_df["axjo_Open"] * 100 - 100
data_df["axjo_Low"] = data_df["axjo_Low"] / data_df["axjo_Open"] * 100 - 100
data_df["axjo_Diff"] = (data_df["axjo_High"] - data_df["axjo_Low"]) / data_df["axjo_Open"] * 100
data_df["audusd_Change"] = data_df["audusd_Change"] / data_df["audusd_Open"] * 100
data_df["audusd_High"] = data_df["audusd_High"] / data_df["audusd_Open"] * 100 - 100
data_df["audusd_Low"] = data_df["audusd_Low"] / data_df["audusd_Open"] * 100 - 100
data_df["audusd_Diff"] = (data_df["audusd_High"] - data_df["audusd_Low"]) / data_df["audusd_Open"] * 100
data_df["spx_Change"] = data_df["spx_Change"] / data_df["spx_Open"] * 100
data_df["spx_High"] = data_df["spx_High"] / data_df["spx_Open"] * 100 - 100
data_df["spx_Low"] = data_df["spx_Low"] / data_df["spx_Open"] * 100 - 100
data_df["spx_Diff"] = (data_df["spx_High"] - data_df["spx_Low"]) / data_df["spx_Open"] * 100

data_df["Time"] = data_df["axjo_Time"]
data_df = data_df[["Time"] + FEATURE_COLUMNS]

data_df["y"] = (data_df["axjo_Change"] > 0).astype(np.int)

window_size = 20
batch_size = 200

test_data = data_df.query('20200101 < Time').reset_index()
test_data_new = test_data.iloc[window_size:len(test_data), :]
test_data_new["xIndex"] = test_data_new.index
test_data_new["x"] = test_data_new["xIndex"].apply(lambda x: get_data(x, test_data, window_size, FEATURE_COLUMNS))
test_dataset = TimeSeriesDataset(test_data_new[["x", "y"]])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
val_data = data_df.query('20181231 < Time < 20200101').reset_index()
val_data_new = val_data.iloc[window_size:len(val_data), :]
val_data_new["xIndex"] = val_data_new.index
val_data_new["x"] = val_data_new["xIndex"].apply(lambda x: get_data(x, val_data, window_size, FEATURE_COLUMNS))
val_dataset = TimeSeriesDataset(val_data_new[["x", "y"]])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

train_data = data_df.query('20190101 > Time').reset_index()
train_data_new = train_data.iloc[window_size:len(train_data), :]
train_data_new["xIndex"] = train_data_new.index
train_data_new["x"] = train_data_new["xIndex"].apply(lambda x: get_data(x, train_data, window_size, FEATURE_COLUMNS))
train_dataset = TimeSeriesDataset(train_data_new[["x", "y"]])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = 1
ninp = len(FEATURE_COLUMNS)
nlen = window_size
nhid = 20
nlayers = 1
dropout = 0.5
bi = True
model = TimeSeriesModel(classes, nlen, ninp, nhid, nlayers, dropout, bi).to(device).double()

weight = (len(train_data) - sum(train_data["y"])) / sum(train_data["y"])

criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(weight), reduction='sum')
optimizer = torch.optim.Adam(model.parameters())

best_val_loss = float("inf")
best_val_acc = float(0)
epochs = 500 # The number of epochs
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

    #scheduler.step()
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