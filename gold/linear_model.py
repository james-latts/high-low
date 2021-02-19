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


#### MAIN ####



#### FUNCTIONS ####
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
        self.classify = nn.Linear(nhid // 4, classes)

    def forward(self, features):
        features = self.dropout(features)
        features = self.linear_1(features)
        features = self.dropout(features)
        features = self.linear_2(features)
        features = self.dropout(features)
        features = self.linear_3(features)
        features = self.dropout(features)
        output = self.classify(features)
        return output
    
    
#### RUN MAIN ####
## STEP 1: Obtain and clean data
# Load in the data in csv
xjo_data_location = "/home/james/Documents/finance/high-low/gold/data/november_3_minute.csv"
original_data_df = pd.read_csv(xjo_data_location)

# Create cleaned dataframe
data_df = pd.DataFrame()
data_df["date_time"] = pd.to_datetime(original_data_df["Date"])
data_df["price"] = original_data_df["Close"]
data_df["log_price"] = np.log(data_df["price"])
data_df["open"] = original_data_df["Open"]
data_df["high"] = original_data_df["High"]
data_df["low"] = original_data_df["Low"]
data_df["close"] = original_data_df["Close"]
data_df["rsi.0"] = original_data_df["RSI.0"]
data_df["rvi.0"] = original_data_df["RVI.0"]
data_df["smi.0"] = original_data_df["SMI.0"]
data_df["smi.1"] = original_data_df["SMI.1"]
data_df["srsi.1"] = original_data_df["sRSI.0"]
data_df["srsi.1"] = original_data_df["sRSI.1"]
data_df["time"] = data_df["date_time"].dt.time
data_df = data_df.sort_values("date_time", ascending = True).set_index("date_time")

# Clean dataframe
data_df = data_df[data_df.index.date != dt.date(2020, 11, 16)] # remove first 110
data_df = data_df[data_df["time"] <= dt.time(8, 0, 0) and data_df["time"] <= dt.time(10, 0, 0)] # remove non trading


## STEP 2: Add additional features to the data and target variable
# Need to add in target


## STEP 3: Get train and validation datasets
# Train dataset
batch_size = 1000
train_start_date = "2020-05-01"
train_end_date =  "2020-09-30"
train_data = data_df.loc[train_start_date:train_end_date]
X = train_data[FEATURE_COLUMNS].values
y = train_data["y"].values
train_dataset = TimeSeriesDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  

# Validation dataset
val_start_date = "2020-10-01"
val_end_date =  "2020-10-31"
val_data = data_df.loc[val_start_date:val_end_date]
X_v = val_data[FEATURE_COLUMNS].values
y_v = val_data["y"].values
val_dataset = TimeSeriesDataset(X_v, y_v)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)


## STEP 4: Create the model and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = 1
ninp = len(FEATURE_COLUMNS)
nhid = 128
dropout = 0
model = TimeSeriesModel(classes, ninp, nhid, dropout).to(device).double()

# Set weight to balance classes
weight = (len(train_data) - sum(train_data["y"])) / sum(train_data["y"])

# Set loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(weight), reduction='sum')
optimizer = torch.optim.Adam(model.parameters())

# Initialise val, model and epochs
best_val_loss = float("inf")
best_val_acc = float(0)
epochs = 100 # The number of epochs
best_model = None

# Train model
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

## STEP 5: Calibrate the model
# Calibration dataset
cal_start_date = "2020-11-01"
cal_end_date =  "2020-11-30"
cal_data = data_df.loc[cal_start_date:cal_end_date]
X_t = cal_data[FEATURE_COLUMNS].values
y_t = cal_data["y"].values
cal_data = TimeSeriesDataset(X_t, y_t)
cal_dataloader = DataLoader(cal_data, batch_size=batch_size, num_workers=0)

cal_loss, cal_acc, y_c_prob = evaluate(best_model, cal_dataloader)
print('=' * 89)
print('| End of training | cal loss {:5.4f} | cal acc {:5.2f}'.format(
    cal_loss /len(cal_dataset) * 100, cal_acc / len(cal_dataset) * 100))
print('| Best val epoch {:3d} | val loss {:5.4f} | val acc {:5.2f}'.format(
    best_epoch, best_val_loss /len(val_dataset) * 100, best_val_acc / len(val_dataset) * 100))
print('=' * 89)

y_t = test_data["Positive"].values
y_t_prob = np.array(y_t_prob).reshape((-1))
fpr, tpr, thresholds = metrics.roc_curve(y_t, y_t_prob, pos_label=1)
auc = metrics.auc(fpr, tpr)
gmeans = np.sqrt(tpr * (1-fpr))
idx = np.argmax(gmeans) - 1
optimal_threshold = thresholds[idx]

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % auc)
plt.scatter(fpr[idx], tpr[idx], marker='o', color='black', label='Best')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic test')
plt.legend(loc="lower right")
plt.show()

pred = np.where(np.array(y_t_prob) >= optimal_threshold, 1, 0)
acc = np.sum(pred == y_t) / len(y_t)
print('\nBest test gmeans: {:5.2f}, Best test acc: {:5.2f}'.format(gmeans[idx], acc * 100))

guess = test_data[pred.astype(np.bool)]
no_guess = test_data[np.invert(pred.astype(np.bool))]
total = len(test_data)
actual_pos = sum(test_data["Positive"])
actual_neg = sum(np.where(test_data["Positive"] == 1, 0, 1))
pred_pos = len(guess)
pred_neg = len(no_guess)
correct_pos = sum(guess["Positive"])
correct_neg = sum(np.where(no_guess["Positive"] == 1, 0, 1))
print("total: ", total)
print("Actual 1's: ", actual_pos, "Pred 1's: ", pred_pos)
print("Actual 0's: ", actual_neg, "Pred 0's: ", pred_neg)
print("Correct 1's: ", correct_pos)
print("Correct 0's", correct_neg)

report = metrics.classification_report(y_t, pred)
print("\nClassification report:\n", report)

precision, recall, thresholds = metrics.precision_recall_curve(y_t, y_t_prob, pos_label=1)
fscore = (2 * precision * recall) / (precision + recall)
fscore = 0.75 * precision + 0.25 * recall
fscore = np.where(np.isnan(fscore), 0, fscore)
idx = np.argmax(fscore) - 1
optimal_threshold = thresholds[idx]

plt.figure()
lw = 2
plt.plot(recall, precision, color='darkorange',
         lw=lw)
plt.plot([0, 1], [0, 0], color='navy', lw=lw, linestyle='--')
plt.scatter(recall[idx], precision[idx], marker='o', color='black', label='Best')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.show()

pred = np.where(np.array(y_t_prob) >= optimal_threshold, 1, 0)
acc = np.sum(pred == y_t) / len(y_t)
print('\nBest test fscore: {:5.2f}, Best test acc: {:5.2f}'.format(fscore[idx], acc * 100))

guess = test_data[pred.astype(np.bool)]
no_guess = test_data[np.invert(pred.astype(np.bool))]
total = len(test_data)
actual_pos = sum(test_data["Positive"])
actual_neg = sum(np.where(test_data["Positive"] == 1, 0, 1))
pred_pos = len(guess)
pred_neg = len(no_guess)
correct_pos = sum(guess["Positive"])
correct_neg = sum(np.where(no_guess["Positive"] == 1, 0, 1))
print("total: ", total)
print("Actual 1's: ", actual_pos, "Pred 1's: ", pred_pos)
print("Actual 0's: ", actual_neg, "Pred 0's: ", pred_neg)
print("Correct 1's: ", correct_pos)
print("Correct 0's", correct_neg)

report = metrics.classification_report(y_t, pred)
print("\nClassification report:\n", report)


## STEP 7: Test the final calibrated model
# Test dataset
test_start_date = "2020-11-01"
test_end_date =  "2020-11-30"
test_data = data_df.loc[test_start_date:test_end_date]
X_t = test_data[FEATURE_COLUMNS].values
y_t = test_data["y"].values
test_dataset = TimeSeriesDataset(X_t, y_t)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)  

test_loss, test_acc, y_t_prob = evaluate(best_model, test_dataloader)
print('=' * 89)
print('| End of calibrating | test loss {:5.4f} | test acc {:5.2f}'.format(
    test_loss /len(test_dataset) * 100, test_acc / len(test_dataset) * 100))
print('| Best val epoch {:3d} | val loss {:5.4f} | val acc {:5.2f}'.format(
    best_epoch, best_val_loss /len(val_dataset) * 100, best_val_acc / len(val_dataset) * 100))
print('| Best cal threshold {:3d} | cal loss {:5.4f} | cal acc {:5.2f}'.format(
    best_epoch, best_val_loss /len(val_dataset) * 100, best_val_acc / len(val_dataset) * 100))
print('=' * 89)

y_t = test_data["Positive"].values
y_t_prob = np.array(y_t_prob).reshape((-1))
pred = np.where(np.array(y_t_prob) >= optimal_threshold, 1, 0)
acc = np.sum(pred == y_t) / len(y_t)
print('\nBest test fscore: {:5.2f}, Best test acc: {:5.2f}'.format(fscore[idx], acc * 100))

guess = test_data[pred.astype(np.bool)]
no_guess = test_data[np.invert(pred.astype(np.bool))]
total = len(test_data)
actual_pos = sum(test_data["Positive"])
actual_neg = sum(np.where(test_data["Positive"] == 1, 0, 1))
pred_pos = len(guess)
pred_neg = len(no_guess)
correct_pos = sum(guess["Positive"])
correct_neg = sum(np.where(no_guess["Positive"] == 1, 0, 1))
print("total: ", total)
print("Actual 1's: ", actual_pos, "Pred 1's: ", pred_pos)
print("Actual 0's: ", actual_neg, "Pred 0's: ", pred_neg)
print("Correct 1's: ", correct_pos)
print("Correct 0's", correct_neg)

report = metrics.classification_report(y_t, pred)
print("\nClassification report:\n", report)