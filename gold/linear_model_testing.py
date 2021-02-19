#### IMPORTS & GLOBALS ####
# System imports
import warnings; warnings.simplefilter(action='ignore', category=Warning) # stop warning messages from pandas
import datetime as dt
from copy import deepcopy

# External imports
import torch; torch.manual_seed(0) # set random seed so results are reproducable
import numpy as np; np.random.seed(0) # set random seed so results are reproducable
import pandas as pd
import torch.nn as nn
from sklearn import metrics

# Local imports
from linear_model import * # not good practice to import like this but easier to test the main like this

# Globals
OPTION_TYPE = "put" # choose option type to predict
TRADE_SIZE = 250 # $ value used per trade
WIN_RETURN_RATE = 1.88 # return if trade is correct
FEATURE_COLUMNS = ["price", "log_price", "open", "high", "low", "close", 
  "rsi.0", "rvi.0", "smi.0", "smi.1", "srsi.1"] # choose feature columns to use


#### MAIN ####
## STEP 1: Load data
# Print log message
print("=" * 80)
print("| Started loading dataset |")
print("-" * 80)

# Load in the data from csv
data_location = "/home/james/Documents/finance/high-low/gold/data/november_3_minute.csv"
original_data_df = pd.read_csv(data_location)

# Create feature dataframe
data_df = pd.DataFrame()
data_df["date_time"] = pd.to_datetime(original_data_df["Date"])
data_df["price"] = original_data_df[["Close"]]
data_df["log_price"] = np.log(data_df["price"])
data_df["open"] = original_data_df[["Open"]]
data_df["high"] = original_data_df[["High"]]
data_df["low"] = original_data_df[["Low"]]
data_df["close"] = original_data_df[["Close"]]
data_df["rsi.0"] = original_data_df[["RSI.0"]]
data_df["rvi.0"] = original_data_df[["RVI.0"]]
data_df["smi.0"] = original_data_df[["SMI.0"]]
data_df["smi.1"] = original_data_df[["SMI.1"]]
data_df["srsi.1"] = original_data_df[["sRSI.0"]]
data_df["srsi.1"] = original_data_df[["sRSI.1"]]
data_df = data_df.sort_values("date_time", ascending = True).set_index("date_time")

# Add target variable
target_df = data_df[["log_price"]]
target_df["time"] = target_df.index.time
target_df["delta_1"] = target_df[["log_price"]].diff(-1)
target_df["delta_2"] = target_df[["log_price"]].diff(-2)
target_df["delta_3"] = target_df[["log_price"]].diff(-3)
target_df["delta_4"] = target_df[["log_price"]].diff(-4)
target_df["delta_5"] = target_df[["log_price"]].diff(-5)
target_df["delta"] = target_df.apply(target, axis = 1)
if OPTION_TYPE in ["call", "high"]:
  target_df["target"] = np.where(target_df["delta"] > 0, 1, 0) # to predict call/high options
elif OPTION_TYPE in ["put", "low"]:
  target_df["target"] = np.where(target_df["delta"] < 0, 1, 0) # to predict put/low options
data_df["y"] = target_df["target"]

# Clean dataframe
data_df = data_df[data_df.index.date != dt.date(2020, 11, 13)] # remove first 110
data_df = data_df[data_df.index.date != dt.date(2020, 11, 15)] # remove first 110
data_df = data_df[(data_df.index.time <= dt.time(8, 0, 0)) | (data_df.index.time >= dt.time(11, 0, 0))] # remove non trading times

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
batch_size = 100
train_start_date = "2020-11-16"
train_end_date =  "2020-11-20"
train_data = data_df.loc[train_start_date:train_end_date]
X = train_data[FEATURE_COLUMNS].values
y = train_data["y"].values
train_dataset = TimeSeriesDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  

# Validation dataset
val_start_date = "2020-11-21"
val_end_date =  "2020-11-24"
val_data = data_df.loc[val_start_date:val_end_date]
X_v = val_data[FEATURE_COLUMNS].values
y_v = val_data["y"].values
val_dataset = TimeSeriesDataset(X_v, y_v)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# Test dataset
test_start_date = "2020-11-24"
test_end_date =  "2020-11-26"
test_data = data_df.loc[test_start_date:test_end_date]
X_t = test_data[FEATURE_COLUMNS].values
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
number_input_columns = len(FEATURE_COLUMNS) # number of data columns
number_hidden_layers = number_input_columns // 2 # dont want too complicated model -> overfitting
dropout_rate = 0.0 # randomly doesnt use layers in training to prevent overfitting
epochs = 100 # the number of epochs to train for

# Create the model
model = TimeSeriesModel(classes, number_input_columns, number_hidden_layers, dropout_rate).to(device).double()

# Set weight to balance classes -> stops it only predicting the majority class
weight = (len(train_data) - sum(train_data["y"])) / sum(train_data["y"])

# Set loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(weight), reduction="sum")
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
  val_loss, val_acc, val_prob = predict(model, val_dataloader, criterion, 0.5, device)
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
    #best_val_prob = val_prob
    
  # Step the optimiser forward now
  optimizer.step()

# Finish training so run best model and print results
train_loss, train_acc, train_prob = predict(best_model, train_dataloader, criterion, 0.5, device)
val_loss, val_acc, val_prob = predict(best_model, val_dataloader, criterion, 0.5, device)
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
  pred_pos = sum(y_v_pred)
  correct_pos = sum(y_v[y_v_pred])
  profit += [correct_pos*WIN_RETURN_RATE - pred_pos]

# Pick index that maximises profit then assign that as optimal threshold
index = np.argmax(np.array(profit))
optimal_threshold = thresholds[index]

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
train_loss, train_acc, train_prob = predict(best_model, train_dataloader, criterion, optimal_threshold, device)
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
print("| Train Dateset | (Don't really care about these results)")
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
print(report)
print("-" * 80)

# Second test the model on the validation dataset (doesnt matter too much)
# Run the best model and get the actual and predicted results
val_loss, val_acc, val_prob = predict(best_model, val_dataloader, criterion, optimal_threshold, device)
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
print("| Validation Dateset | (Don't care too much about these results)")
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
print(report)
print("-" * 80)

# Finally test the model on the test dataset (this is the main dataset results we care about)
# Run the best model and get the actual and predicted results
test_loss, test_acc, test_prob = predict(best_model, test_dataloader, criterion, optimal_threshold, device)
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
print("| Test Dateset | (Care the most about these results)")
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
print("\n" + report)
print("-" * 80)
print("")

# Print test profit result
print("-" * 80)
print("Testing Profit/Loss Report: {}$ trades from {} to {}".format(TRADE_SIZE, test_start_date, test_end_date))
print("-" * 80)
trade_period = str((dt.datetime.strptime(test_end_date, "%Y-%m-%d") - dt.datetime.strptime(test_start_date, "%Y-%m-%d")).days)
number_of_trades = pred_pos
cost_of_trades = round(number_of_trades * TRADE_SIZE)
number_of_wins = correct_pos
return_of_wins = round(correct_pos * TRADE_SIZE * WIN_RETURN_RATE)
expected_return = (correct_pos*WIN_RETURN_RATE - pred_pos) / pred_pos
expected_profit = round(expected_return * cost_of_trades)
print("{} total trades with total cost of {}$".format(number_of_trades, cost_of_trades))
print("{} winning trades with total return of {}$".format(number_of_wins, return_of_wins))
print("")
print("Profit for {} days trading with {}$ trades:".format(trade_period, TRADE_SIZE))
print("\tprofit = return - cost")
print("\tprofit = {}$ - {}$".format(return_of_wins, cost_of_trades))
print("\tprofit = {}$".format(expected_profit))
print("")
print("Expected return per trade = {}%".format(round(expected_return * 100, 2)))
print("-" * 80)
print("")

# Print log message
print("-" * 80)
print("| Finished testing the model |")
print("=" * 80)  