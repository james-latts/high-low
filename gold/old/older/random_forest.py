#### IMPORTS & GLOBALS ####
# Imports
import os
import numpy as np
import pandas as pd
import datetime as dt
from sklearn import metrics
from itertools import product
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Globals
FEATURES = ["Sma", "Std", "Max", "Min", "Ema", "Srsi", "Rsi", "So"]
N = [7, 14, 21]
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
train_data = data_df.query('20200101 > DateTime').dropna().reset_index(drop = True)
X = train_data[FEATURE_COLUMNS].values
y = train_data["y"].values

clf = RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_split=14, oob_score=True, random_state=11)
clf.fit(X, y)

y_pred = clf.predict(X)

print("\nTrain acc:", sum(y == y_pred) / len(y) * 100)
print("\nOob score:", clf.oob_score_)

test_data = data_df.query('20200101 <= DateTime').dropna().reset_index(drop = True)
X_t = test_data[FEATURE_COLUMNS].values
y_t = test_data["y"].values

y_t_prob = clf.predict_proba(X_t)[:, 1]
y_t_pred = clf.predict(X_t)

print("\nTest acc:", sum(y_t == y_t_pred) / len(y_t) * 100)

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

report = metrics.classification_report(y_t, y_t_pred)
print("\nClassification report:\n", report)
