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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Local imports
None

# Globals
FEATURE_COLUMNS = ["price", "open", "low", "high", "volume"]


#### MAIN ####
def main():
  return None


#### FUNCTIONS ####
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
  window_end = row.name
  window_start = window_end - window_size
  return [df[feature_list][window_start:window_end].values]
  
def predict(model, dataloader):
  """ Predicts on a given dataset for a given model and cut off threshold """
  model.eval()
  total_loss = 0.
  with torch.no_grad():
    for batch_number, batch_data in enumerate(dataloader):
      features, targets = batch_data
      features = features[0].to(device).double()
      targets = targets[0].to(device).double()
      output = model(features)
      loss = criterion(output, targets)
      total_loss += loss.item()
  return total_loss


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
  
class EncoderDecoderModel(nn.Module):
  """ Encoder-Decoder Model """
  def __init__(self, input_size, hidden_size, dropout_rate=0.5):
    super(EncoderDecoderModel, self).__init__()
    self.model_type = 'TimeSeries'
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    
    self.dropout = nn.Dropout(self.dropout_rate)
    
    self.encoder = nn.Linear(self.input_size, self.hidden_size)
    self.encoder_2
    self.decoder = nn.Linear(self.hidden_size, self.input_size)

  def forward(self, features):
    features = self.dropout(features)
    features = self.encoder(features)
    features = self.dropout(features)
    output = self.decoder(features)
    return output
  
  def encode(self, features):
    
    features = self.encoder(features)
    return features
  
  def decode(self, features):
    
    features = self.decoder(features)
    return features
  
class EncoderDecoderModel(nn.Module):
  """ Encoder-Decoder Model """
  def __init__(self, input_size, hidden_size, dropout_rate=0.25):
    super(EncoderDecoderModel, self).__init__()
    self.model_type = 'TimeSeries'
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    
    self.dropout = nn.Dropout(self.dropout_rate)
    
    self.encoder_1 = nn.Linear(self.input_size, self.input_size * 3 // 4)
    self.encoder_2 = nn.Linear(self.input_size * 3 // 4, self.input_size // 2)
    self.decoder_1 = nn.Linear(self.input_size // 2, self.input_size * 3 // 4)
    self.decoder_2 = nn.Linear(self.input_size * 3 // 4, self.input_size)
    
  def forward(self, features):
    features = self.encoder_1(features)
    features = self.dropout(features)
    features = self.encoder_2(features)
    features = self.dropout(features)
    features = self.decoder_1(features)
    features = self.dropout(features)
    features = self.decoder_2(features)
    return features
  
  def encode(self, features):
    
    features = self.encoder_1(features)
    features = self.encoder_2(features)
    return features
  
  def decode(self, features):
    
    features = self.decoder_1(features)
    features = self.decoder_2(features)
    return features
    
    
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
#data_df["min"] = data_df["low"].rolling(window=14).min()
#data_df["max"] = data_df["high"].rolling(window=14).max()
#data_df["sma"] = (data_df["price"].rolling(window=14).mean() / data_df["price"] - 1) * 100
#data_df["std"] = data_df["price"].rolling(window=14).std()
#data_df["ema_12"] = (data_df["price"].ewm(span=12).mean() / data_df["price"] - 1) * 100
#data_df["ema_14"] = (data_df["price"].ewm(span=14).mean() / data_df["price"] - 1) * 100
#data_df["ema_26"] = (data_df["price"].ewm(span=26).mean() / data_df["price"] - 1) * 100
#data_df["low_rsi"] = compute_rsi(data_df["low"], 14)
#data_df["rsi"] = compute_rsi(data_df["price"], 14)
#data_df["high_rsi"] = compute_rsi(data_df["high"], 14)
#data_df["srsi"] = compute_srsi((data_df["low_rsi"], data_df["rsi"], data_df["high_rsi"]), 14)
#data_df["so"] = compute_so(data_df["price"], 14)
#data_df["obv"] = compute_obv((data_df["price"], data_df["volume"]), 14)

# Set the dates for train, validation and test start
train_start_date = str(data_df.index[0].date() + dt.timedelta(days=1))
val_start_date = "2020-09-01"
test_start_date = "2020-11-01"
train_days = dt.datetime.strptime(val_start_date, "%Y-%m-%d") - dt.datetime.strptime(train_start_date, "%Y-%m-%d") - dt.timedelta(days=1)
val_days = dt.datetime.strptime(test_start_date, "%Y-%m-%d") - dt.datetime.strptime(val_start_date, "%Y-%m-%d") - dt.timedelta(days=1)
test_days = data_df.index[-1] - dt.datetime.strptime(test_start_date, "%Y-%m-%d") - dt.timedelta(days=1)
train_end_date = str((dt.datetime.strptime(train_start_date, "%Y-%m-%d") + train_days).date())
val_end_date = str((dt.datetime.strptime(val_start_date, "%Y-%m-%d") + val_days).date())
#val_end_date = ""
test_end_date = str((dt.datetime.strptime(test_start_date, "%Y-%m-%d") + dt.timedelta(days=test_days.days)).date())
#test_end_date = ""

# Scale features
train_data = data_df.loc[train_start_date:train_end_date]
scaler = MinMaxScaler()
scaler.fit(train_data[FEATURE_COLUMNS])
data_df[FEATURE_COLUMNS] = scaler.transform(data_df[FEATURE_COLUMNS])

# Create a window
window_size = 15
data_df = data_df.reset_index()
data_df.head(20).apply(lambda x: get_data(x, data_df, window_size, FEATURE_COLUMNS), axis = 1)[17][0][0]
data_df["x"] = data_df.apply(lambda x: get_data(x, data_df, window_size, FEATURE_COLUMNS), axis = 1)
data_df = data_df.set_index("date_time")

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
batch_size = 5000
train_data = data_df.loc[train_start_date:train_end_date]
X = train_data["x"].values
y = train_data["x"].values
train_dataset = TimeSeriesDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  

# Validation dataset
val_data = data_df.loc[val_start_date:val_end_date]
X_v = val_data["x"].values
y_v = val_data["x"].values
val_dataset = TimeSeriesDataset(X_v, y_v)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# Test datasetput
test_data = data_df.loc[test_start_date:test_end_date]
X_t = test_data["x"].values
y_t = test_data["x"].values
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
number_input_columns = window_size * len(FEATURE_COLUMNS) # number of data columns
number_hidden_layers = 1 # dont want too complicated model -> overfitting
dropout_rate = 0.50 # randomly doesnt use layers in training to prevent overfitting
epochs = 25 # the number of epochs to train for

# Create the model
model = EncoderDecoderModel(number_input_columns, number_hidden_layers).to(device).double()

# Set loss and optimizer
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters())

# Initialise the best model and validation loss
best_val_loss = float("inf")
best_model = None

# Train model
for epoch in range(1, epochs + 1):
  model.train()
  total_loss = 0.
  for batch_number, batch_data in enumerate(train_dataloader):
    features, targets = batch_data
    features = features[0].to(device).double()
    targets = targets[0].to(device).double()
    optimizer.zero_grad()
    output = model(features)
    loss = criterion(output, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    total_loss += loss.item()
    
  # Predict on validation dataset and print log message
  val_loss = predict(model, val_dataloader)
  print("| epoch {:3d} | loss {:5.4f} |".format(epoch, 
    total_loss / len(train_dataset) * 100))
  print("-" * 80)
  print("| end of epoch {:3d} | val loss {:5.4f} |".format(epoch, 
    val_loss / len(val_dataset) * 100))
  print("-" * 80)
  
  # If the model val loss is better set the new model to this one
  if val_loss < best_val_loss:
    best_epoch = epoch
    best_model = deepcopy(model)
    best_val_loss = val_loss
    
  # Step the optimiser forward now
  optimizer.step()

# Finish training so run best model and print results
train_loss = predict(best_model, train_dataloader)
val_loss = predict(best_model, val_dataloader)
print("| Finished training | train loss {:5.4f} |".format(
    train_loss /len(train_dataset) * 100))
print("| Best val epoch {:3d} | val loss {:5.4f} |".format(
    best_epoch, val_loss /len(val_dataset) * 100))
print("=" * 80 + "\n")


# Second test the model on the validation dataset (doesnt matter too much)
# Run the best model and get the actual and predicted results
test_pred = []
test_out = []
for batch_number, batch_data in enumerate(test_dataloader):
  features, targets = batch_data
  features = features[0].to(device).double()
  features = best_model.encode(features)
  output = features.cpu().detach().numpy().tolist()
  out = best_model.decode(features).cpu().detach().numpy().tolist()
  test_pred += [output]
  test_out += [out]
test_pred = [np.array(y) for x in test_pred for y in x]
test_out = [np.array(y) for x in test_out for y in x]

test_data["y"] = test_out
