#### IMPORTS & GLOBALS ####
# System imports
import os
import warnings; warnings.simplefilter(action='ignore', category=Warning) # stop warning messages from pandas
import datetime as dt
from copy import deepcopy

# External imports
import torch; torch.manual_seed(0) # set random seed so results are reproducable
import numpy as np; np.random.seed(0) # set random seed so results are reproducable
import pandas as pd
import torch.nn as nn
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Local imports
None

# Globals
OPTION_TYPE = "put" # choose option type to predict
TRADE_SIZE = 250 # $ value used per trade
WIN_RETURN_RATE = 1.88 # return if trade is correct
FEATURE_COLUMNS = ["price", "open", "high", "low", "volume"] # choose feature columns to use


#### MAIN ####
def main():
  return None


#### FUNCTIONS ####
def target(row):
  """ Creates target column for 1 minute time series data (15 min options) """
  if row["time"].minute % 5 == 0:
    name = "delta_{}".format(15)
    target = row[name]
  else:
    target = 0
  return target

def compute_rsi(data, time_window):
  diff = data.diff(1)
  up_chg = 0 * diff
  down_chg = 0 * diff    
  up_chg[diff > 0] = diff[diff > 0]
  down_chg[diff < 0] = diff[diff < 0]    
  up_chg_avg = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
  down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()    
  rs = abs(up_chg_avg/down_chg_avg)
  rsi = 100 - 100/(1+rs)
  return rsi
  
def compute_srsi(data, time_window):
  low_rsi, rsi, high_rsi = data
  srsi = (rsi - low_rsi) / (high_rsi - low_rsi)
  return srsi

def compute_so(data, time_window):
  high = data.rolling(window=time_window).max()
  low = data.rolling(window=time_window).min()
  so = (data - low)/(high - low) * 100
  return so
  
def compute_obv(data, time_window):
  price, volume = data
  diff = price.diff(1)
  chg = 0 * price
  chg[diff > 0] = volume[diff > 0]
  chg[diff < 0] = -volume[diff < 0] 
  obv = chg.rolling(time_window).sum()
  return obv

def get_data(row, df, window_size, feature_list):
  x = []
  if row.name < window_size - 1:
    return 0
  for f in feature_list:
    window_end = row.name
    window_start = window_end - window_size
    x.append(df[f][window_start:window_end].values)
  x = np.stack(x, axis = 1)
  return x

def predict(model, dataloader, criterion, threshold):
  """ Predicts on a given dataset for a given model and cut off threshold """
  model.eval()
  total_loss = 0.
  total_correct = 0
  total_prob = []
  with torch.no_grad():
    for batch_number, batch_data in enumerate(dataloader):
      features, targets = batch_data
      features = features.to(device)
      targets = targets.view((-1, 1)).to(device).double()
      model.init_hidden()
      output = model(features)
      loss = criterion(output, targets)
      total_loss += loss.item()
      prob = torch.sigmoid(output).cpu().detach().numpy()
      pred = np.where(prob > threshold, 1, 0)
      targ = targets.cpu().detach().numpy()
      total_correct += np.sum(pred == targ)
      total_prob += prob.tolist()
  return total_loss, total_correct, total_prob


#### CLASSES ####
class TimeSeriesDataset(Dataset):
  """ Time Series Dataset """
  def __init__(self, X, Y, transform=None):
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
  """ Time Series Model """
  def __init__(self, classes, input_size, window_size, hidden_layer_size, dropout_rate, device):
    super(TimeSeriesModel, self).__init__()
    self.model_type = 'TimeSeries'
    self.device = device
    self.input_size = input_size
    self.window_size = window_size
    self.hidden_layer_size = hidden_layer_size
    self.dropout_rate = dropout_rate
    
    self.dropout = nn.Dropout(self.dropout_rate)
    self.lstm_1 = nn.LSTM(input_size, hidden_layer_size, dropout=self.dropout_rate)
    self.lstm_2 = nn.LSTM(input_size, hidden_layer_size, dropout=self.dropout_rate)
    self.lstm_3 = nn.LSTM(input_size, hidden_layer_size, dropout=self.dropout_rate)
    self.lstm_4 = nn.LSTM(input_size, hidden_layer_size, dropout=self.dropout_rate)
    self.classify = nn.Linear(sum(self.window_size) * self.hidden_layer_size, classes)
  
  def init_hidden(self):
    self.hidden_cell_1 = \
      (torch.zeros(1, self.window_size[0], self.hidden_layer_size).to(self.device).double(),
      torch.zeros(1, self.window_size[0], self.hidden_layer_size).to(self.device).double())     
    self.hidden_cell_2 = \
      (torch.zeros(1, self.window_size[1], self.hidden_layer_size).to(self.device).double(),
      torch.zeros(1, self.window_size[1], self.hidden_layer_size).to(self.device).double())     
    self.hidden_cell_3 = \
      (torch.zeros(1, self.window_size[2], self.hidden_layer_size).to(self.device).double(),
      torch.zeros(1, self.window_size[2], self.hidden_layer_size).to(self.device).double())
    self.hidden_cell_4 = \
      (torch.zeros(1, self.window_size[3], self.hidden_layer_size).to(self.device).double(),
      torch.zeros(1, self.window_size[3], self.hidden_layer_size).to(self.device).double())     
  def forward(self, features):
    
    features_1, features_2, features_3, features_4 = features
    lstm_out_1, self.hidden_cell_1 = self.lstm_1(features_1, self.hidden_cell_1)  
    lstm_out_2, self.hidden_cell_2 = self.lstm_1(features_2, self.hidden_cell_2)  
    lstm_out_3, self.hidden_cell_3 = self.lstm_1(features_3, self.hidden_cell_3)  
    lstm_out_4, self.hidden_cell_4 = self.lstm_1(features_4, self.hidden_cell_4)
    lstm_out = nn.Concatenate((lstm_out_1, lstm_out_2, lstm_out_3, lstm_out_4), 0)
    lstm_out = self.dropout(lstm_out)
    output = self.classify(lstm_out.view(len(features), -1))
    return output
    
    
#### RUN MAIN ####
## STEP 1: Load data
# Print log message
print("=" * 80)
print("| Started loading dataset |")
print("-" * 80)

# Load in the data from csv
xjo_data_location = \
    "/home/james/Documents/finance/high-low/aud-nzd/data/minute/2020"
original_data_df = pd.DataFrame()
for xjo_csv_file in os.listdir(xjo_data_location):
    xjo_csv_location = xjo_data_location + "/" + xjo_csv_file
    csv_data_df = pd.read_csv(xjo_csv_location)
    original_data_df = pd.concat([original_data_df, csv_data_df])

# Create cleaned dataframe
data_df = pd.DataFrame()
data_df["date_time"] = pd.to_datetime(original_data_df["Time"])
data_df["price"] = original_data_df[["Last"]]
data_df["log_price"] = np.log(data_df["price"])
data_df["open"] = original_data_df[["Open"]]
data_df["high"] = original_data_df[["High"]]
data_df["low"] = original_data_df[["Low"]]
data_df["volume"] = original_data_df[["Volume"]]
data_df = data_df.sort_values("date_time", ascending = True).set_index("date_time")

# Add feature variables
# data_df["min"] = data_df["low"].rolling(window=14).min()
# data_df["max"] = data_df["high"].rolling(window=14).max()
# data_df["sma"] = (data_df["price"].rolling(window=14).mean() / data_df["price"] - 1) * 100
# data_df["std"] = data_df["price"].rolling(window=14).std()
# data_df["ema_12"] = (data_df["price"].ewm(span=12).mean() / data_df["price"] - 1) * 100
# data_df["ema_14"] = (data_df["price"].ewm(span=14).mean() / data_df["price"] - 1) * 100
# data_df["ema_26"] = (data_df["price"].ewm(span=26).mean() / data_df["price"] - 1) * 100
# data_df["low_rsi"] = compute_rsi(data_df["low"], 14)
# data_df["rsi"] = compute_rsi(data_df["price"], 14)
# data_df["high_rsi"] = compute_rsi(data_df["high"], 14)
# data_df["srsi"] = compute_srsi((data_df["low_rsi"], data_df["rsi"], data_df["high_rsi"]), 14)
# data_df["so"] = compute_so(data_df["price"], 14)
# data_df["obv"] = compute_obv((data_df["price"], data_df["volume"]), 14)

# Set the dates for train, validation and test start
train_start_date = str(data_df.index[0].date())
val_start_date = "2020-09-01"
test_start_date = "2020-11-01"
train_days = dt.datetime.strptime(val_start_date, "%Y-%m-%d") - dt.datetime.strptime(train_start_date, "%Y-%m-%d") - dt.timedelta(days=1)
val_days = dt.datetime.strptime(test_start_date, "%Y-%m-%d") - dt.datetime.strptime(val_start_date, "%Y-%m-%d") - dt.timedelta(days=1)
test_days = data_df.index[-1] - dt.datetime.strptime(test_start_date, "%Y-%m-%d") - dt.timedelta(days=1)
train_end_date = str((dt.datetime.strptime(train_start_date, "%Y-%m-%d") + train_days).date())
val_end_date = str((dt.datetime.strptime(val_start_date, "%Y-%m-%d") + val_days).date())
test_end_date = str((dt.datetime.strptime(test_start_date, "%Y-%m-%d") + dt.timedelta(days=test_days.days)).date())

# Scale features
train_data = data_df.loc[train_start_date:train_end_date]
scaler = MinMaxScaler()
scaler.fit(train_data[FEATURE_COLUMNS])
data_df[FEATURE_COLUMNS] = scaler.transform(data_df[FEATURE_COLUMNS])

# Add target variable
target_df = data_df[["log_price"]]
target_df["time"] = target_df.index.time
for i in range(1, 16):
  name = "delta_{}".format(i)
  target_df[name] = target_df[["log_price"]].diff(-i)
target_df["delta"] = target_df.apply(target, axis = 1)
if OPTION_TYPE in ["call", "high"]:
  #data_df["y"] = np.where(((target_df["delta"] - np.mean(target_df["delta"])) / np.std(target_df["delta"])) > 0.5, 1, 0) # to predict call/high options
  data_df["y"] = np.where(target_df["delta"] > 0, 1, 0) # to predict call/high options
elif OPTION_TYPE in ["put", "low"]:
  #data_df["y"] = np.where(((target_df["delta"] - np.mean(target_df["delta"])) / np.std(target_df["delta"])) < -0.5, 1, 0) # to predict call/high options
  data_df["y"] = np.where(target_df["delta"] < 0, 1, 0) # to predict put/low options  

# Create a window
window_size = [5, 15, 30, 60]
x_columns = ["x_{}".format(x) for x in window_size]
data_df = data_df.reset_index()
data_df["x_5"] = data_df.apply(lambda x: get_data(x, data_df, 5, FEATURE_COLUMNS), axis = 1)
data_df["x_15"] = data_df.apply(lambda x: get_data(x, data_df, 15, FEATURE_COLUMNS), axis = 1)
data_df["x_30"] = data_df.apply(lambda x: get_data(x, data_df, 30, FEATURE_COLUMNS), axis = 1)
data_df["x_60"] = data_df.apply(lambda x: get_data(x, data_df, 60, FEATURE_COLUMNS), axis = 1)
data_df = data_df.set_index("date_time")

# Clean dataframe
data_df = data_df[(data_df.index.weekday != 6)] # remove non trading times
data_df = data_df[(data_df.index.time >= dt.time(8, 0, 0)) | (data_df.index.weekday == 0)] # remove non trading times
data_df = data_df[(data_df.index.time >= dt.time(9, 0, 0)) | (data_df.index.weekday == 1)] # remove non trading times
data_df = data_df[(data_df.index.time <= dt.time(8, 0, 0)) | (data_df.index.time >= dt.time(11, 0, 0))] # remove non trading times
data_df = data_df[data_df.index.date != data_df.head(1).index.values[0].astype("datetime64[m]").astype(dt.datetime).date()] # remove first day
data_df = data_df[data_df.index.minute % 5 == 0]

# Print log message
print("-" * 80)
print("| Finished loading dataset |")
print("=" * 80 + "\n")


## STEP 2: Get train, validation and test datasets
# Print log message
print("=" * 80)
print("| Started splitting dataset into train, validation and test sets |")
print("-" * 80)

# Train dataset
batch_size = 1000
train_data = data_df.loc[train_start_date:train_end_date]
X = train_data[x_columns].values
y = train_data["y"].values
train_dataset = TimeSeriesDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  

# Validation dataset
val_data = data_df.loc[val_start_date:val_end_date]
X_v = val_data[x_columns].values
y_v = val_data["y"].values
val_dataset = TimeSeriesDataset(X_v, y_v)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# Test datasetput
test_data = data_df.loc[test_start_date:test_end_date]
X_t = test_data[x_columns].values
y_t = test_data["y"].values
test_dataset = TimeSeriesDataset(X_t, y_t)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)  

# Print log message
print("-" * 80)
print("| Finished splitting dataset into train, validation and test sets |")
print("=" * 80  + "\n")


## STEP 3: Create and train the model
# Print log message
print("=" * 80)
print("| Started training |")
print("-" * 80)

# Initial setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set to use GPU if avaliable
classes = 1 # predict either no trade or trade
number_windows = window_size
number_input_columns = len(FEATURE_COLUMNS) # number of data columns
number_hidden_layers = 2#number_input_columns // 2 # dont want too complicated model -> overfitting
dropout_rate = 0.50 # randomly doesnt use layers in training to prevent overfitting
epochs = 50 # the number of epochs to train for

# Create the model
model = TimeSeriesModel(classes, number_input_columns, number_windows, number_hidden_layers, dropout_rate, device).to(device).double()

# Set weight to balance classes -> stops it only predicting the majority class
weight = (len(train_data) - sum(train_data["y"])) / sum(train_data["y"])

# Set loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(weight), reduction="sum")
#criterion = BinaryDiceLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters())

# Initialise the best model and validation loss
best_val_loss = float("inf")
best_model = None

# Train model
for epoch in range(1, epochs + 1):
  model.train()
  total_loss = 0.
  total_correct = 0
  total_prob = []
  for batch_number, batch_data in enumerate(train_dataloader):
    features, targets = batch_data
    features = features.to(device)
    targets = targets.view((-1, 1)).to(device).double()
    optimizer.zero_grad()
    model.init_hidden()
    output = model(features)
    loss = criterion(output, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    total_loss += loss.item()
    prob = torch.sigmoid(output).cpu().detach().numpy()
    total_correct += np.sum(np.round(prob) == targets.cpu().detach().numpy())
    total_prob += prob.tolist()
    
  # Predict on validation dataset and print log message
  val_loss, val_acc, val_prob = predict(model, val_dataloader, criterion, 0.5)
  print("| epoch {:3d} | loss {:5.4f} | acc {:5.2f}".format(epoch, 
    total_loss / len(total_prob) * 100, total_correct / len(train_dataset) * 100))
  print("-" * 80)
  print("| end of epoch {:3d} | val loss {:5.4f} | val_acc {:5.2f} ".format(epoch, 
    val_loss / len(val_dataset) * 100, val_acc / len(val_dataset) * 100))
  print("-" * 80)
  
  # If the model val loss is better set the new model to this one
  if val_loss < best_val_loss:
    best_epoch = epoch
    best_model = deepcopy(model)
    best_val_loss = val_loss
    best_val_prob = val_prob
    
  # Step the optimiser forward now
  optimizer.step()

# Finish training so run best model and print results
train_loss, train_acc, train_prob = predict(best_model, train_dataloader, criterion, 0.5)
val_loss, val_acc, val_prob = predict(best_model, val_dataloader, criterion, 0.5)
print("| Finished training | train loss {:5.4f} | train acc {:5.2f}".format(
    train_loss /len(train_dataset) * 100, train_acc / len(train_dataset) * 100))
print("| Best val epoch {:3d} | val loss {:5.4f} | val acc {:5.2f}".format(
    best_epoch, val_loss /len(val_dataset) * 100, val_acc / len(val_dataset) * 100))
print("=" * 80 + "\n")


## STEP 4: Calibrate the model
# Print log message
print("=" * 80)
print("| Started calibrating the model |")
print("-" * 80)

# Get the validation known y and predicted prob y
y_v = val_data["y"].values
y_v_prob = np.array(val_prob).reshape((-1))

# Get fpr, tpr and thresholds (ROC curve and AUC is general measure of model strength)
fpr, tpr, thresholds = metrics.roc_curve(y_v, y_v_prob, pos_label=1)
auc = metrics.auc(fpr, tpr)
print("ROC/AUC score: {}".format(round(auc, 2)))

# Normally you would just pick threshold that maximised the geometric mean
# between fpr and tpr but we want to maximise profit function
profit = []
for thresh in thresholds:
  y_v_pred = np.where(y_v_prob >= thresh, True, False)
  total_pos = sum(y_v)
  pred_pos = sum(y_v_pred)
  correct_pos = sum(y_v[y_v_pred])
  profit += [correct_pos*WIN_RETURN_RATE - pred_pos]

# Pick index that maximises profit then assign that as optimal threshold
index = np.argmax(np.array(profit))
optimal_threshold = thresholds[index]
  
# gmeans threshold picking logic
#gmeans = np.sqrt(tpr * (1 - fpr))
#index = np.argmax(gmeans)
#optimal_threshold = thresholds[index]

# Print log message
print("-" * 80)
print("| Finished calibrating the model |")
print("=" * 80 + "\n")


## STEP 5: Test the final calibrated model
# Print log message
print("=" * 80)
print("| Started testing the model |")
print("-" * 80)

# First test the model on the training dataset (doesnt matter too much)
# Run the best model and get the actual and predicted results
train_loss, train_acc, train_prob = predict(best_model, train_dataloader, criterion, optimal_threshold)
y_t = train_data["y"].values
y_t_prob = np.array(train_prob).reshape((-1))
pred = np.where(np.array(y_t_prob) >= optimal_threshold, 1, 0)

# Calculate the raw results
prediction = train_data[pred.astype(np.bool)]
no_prediction = train_data[np.invert(pred.astype(np.bool))]
total = len(train_data)
actual_pos = sum(train_data["y"])
actual_neg = sum(np.where(train_data["y"] == 1, 0, 1))
pred_pos = len(prediction)
pred_neg = len(no_prediction)
correct_pos = sum(prediction["y"])
correct_neg = sum(np.where(no_prediction["y"] == 1, 0, 1))
acc_pos = round(correct_pos / pred_pos * 100, 2)
acc_neg = round(correct_neg / pred_neg * 100, 2)

# Print the raw results report
print("")
print("-" * 80)
print("| Train Dateset |")
print("-" * 80)
print("Raw results report:")
print("-" * 80)
print("| Actual 1's:", actual_pos, "| Pred 1's:", pred_pos, "| Correct 1's:", correct_pos, "| Accuracy 1's:", acc_pos)
print("| Actual 0's:", actual_neg, "| Pred 0's:", pred_neg, "| Correct 0's", correct_neg, "| Accuracy 0's:", acc_neg)
print("| Total Data Points:", total, "|")
print("-" * 80)

# Print the classification report
report = metrics.classification_report(y_t, pred)
print("Classification report:")
print("-" * 80)
print("")
print(report)
print("-" * 80)

# Second test the model on the validation dataset (doesnt matter too much)
# Run the best model and get the actual and predicted results
val_loss, val_acc, val_prob = predict(best_model, val_dataloader, criterion, optimal_threshold)
y_v = val_data["y"].values
y_v_prob = np.array(val_prob).reshape((-1))
pred = np.where(np.array(y_v_prob) >= optimal_threshold, 1, 0)

# Calculate the raw results
prediction = val_data[pred.astype(np.bool)]
no_prediction = val_data[np.invert(pred.astype(np.bool))]
total = len(val_data)
actual_pos = sum(val_data["y"])
actual_neg = sum(np.where(val_data["y"] == 1, 0, 1))
pred_pos = len(prediction)
pred_neg = len(no_prediction)
correct_pos = sum(prediction["y"])
correct_neg = sum(np.where(no_prediction["y"] == 1, 0, 1))
acc_pos = round(correct_pos / pred_pos * 100, 2)
acc_neg = round(correct_neg / pred_neg * 100, 2)

# Print the raw results report
print("")
print("-" * 80)
print("| Validation Dateset |")
print("-" * 80)
print("Raw results report:")
print("-" * 80)
print("| Actual 1's:", actual_pos, "| Pred 1's:", pred_pos, "| Correct 1's:", correct_pos, "| Accuracy 1's:", acc_pos)
print("| Actual 0's:", actual_neg, "| Pred 0's:", pred_neg, "| Correct 0's", correct_neg, "| Accuracy 0's:", acc_neg)
print("| Total Data Points:", total, "|")
print("-" * 80)

# Print the classification report
report = metrics.classification_report(y_v, pred)
print("Classification report:")
print("-" * 80)
print("")
print(report)
print("-" * 80)

# Finally test the model on the test dataset (this is the main dataset results we care about)
# Run the best model and get the actual and predicted results
test_loss, test_acc, test_prob = predict(best_model, test_dataloader, criterion, optimal_threshold)
y_t = test_data["y"].values
y_t_prob = np.array(test_prob).reshape((-1))
pred = np.where(np.array(y_t_prob) >= optimal_threshold, 1, 0)

# Calculate the raw results
predictions = test_data[pred.astype(np.bool)]
no_predictions = test_data[np.invert(pred.astype(np.bool))]
total = len(test_data)
actual_pos = sum(test_data["y"])
actual_neg = sum(np.where(test_data["y"] == 1, 0, 1))
pred_pos = len(predictions)
pred_neg = len(no_predictions)
correct_pos = sum(predictions["y"])
correct_neg = sum(np.where(no_predictions["y"] == 1, 0, 1))
acc_pos = round(correct_pos / pred_pos * 100, 2)
acc_neg = round(correct_neg / pred_neg * 100, 2)

# Print the raw results report
print("")
print("-" * 80)
print("| Test Dateset |")
print("-" * 80)
print("Raw results report:")
print("-" * 80)
print("| Actual 1's:", actual_pos, "| Pred 1's:", pred_pos, "| Correct 1's:", correct_pos, "| Accuracy 1's:", acc_pos)
print("| Actual 0's:", actual_neg, "| Pred 0's:", pred_neg, "| Correct 0's", correct_neg, "| Accuracy 0's:", acc_neg)
print("| Total Data Points:", total, "|")
print("-" * 80)

# Print the classification report
report = metrics.classification_report(y_t, pred)
print("Classification report:")
print("-" * 80)
print("")
print(report)
print("-" * 80)
print("")

# Print test profit result
print("-" * 80)
print("Testing Profit/Loss Report: ${} trades from {} to {}".format(TRADE_SIZE, test_start_date, test_end_date))
print("-" * 80)
trade_period = str((dt.datetime.strptime(test_end_date, "%Y-%m-%d") - dt.datetime.strptime(test_start_date, "%Y-%m-%d")).days)
number_of_trades = pred_pos
cost_of_trades = round(number_of_trades * TRADE_SIZE)
number_of_wins = correct_pos
return_of_wins = round(correct_pos * TRADE_SIZE * WIN_RETURN_RATE)
expected_return = (correct_pos*WIN_RETURN_RATE - pred_pos) / pred_pos
expected_profit = round(expected_return * cost_of_trades)
print("{} total trades with total cost of ${}".format(number_of_trades, cost_of_trades))
print("{} winning trades with total return of ${}".format(number_of_wins, return_of_wins))
print("")
print("Profit for {} days trading with ${} trades:".format(trade_period, TRADE_SIZE))
print("\tprofit = return - cost")
print("\tprofit = ${} - ${}".format(return_of_wins, cost_of_trades))
print("\tprofit = ${}".format(expected_profit))
print("")
print("Expected return per trade = {}%".format(round(expected_return * 100, 2)))
print("-" * 80)
print("")

# Print log message
print("-" * 80)
print("| Finished testing the model |")
print("=" * 80)