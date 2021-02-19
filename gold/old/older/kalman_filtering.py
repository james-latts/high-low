#### IMPORTS & GLOBALS ####
# Imports
import os
import time 
import math
import numpy as np
import pandas as pd
import datetime as dt
from sklearn import metrics
import matplotlib.pyplot as plt


# Globals


#### MAIN ####



#### FUNCTIONS ####


#### CLASSES ####



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


#
start = time.time()

#
price_data = data_df["Price"].values
price_points = len(price_data)

#
N = 5
A = [[1 if i == n else 0 for i in range(N)] for n in range(N)]
const = 0
P_init = (10**-7 * np.array(A)).tolist()
R = (10**-7 * np.array(A)).tolist()
Q = (10**-7 * np.array(A)).tolist()

#
KF = []
update = []
for n in range(N):
    KF.append(price_data[n])

for x in range(N, price_points - N):
    x_init = [[price_data[x - i]] for i in reversed(range(1, N + 1))]
    prediction = np.dot(A, x_init) + const
    P_min = np.dot(np.dot(A, P_init), A) + Q
    KF.append(prediction[N - 1].tolist()[0])
    y_min = prediction[N - 1]
    P_y_min = P_min + R
    K_gain = np.dot(P_min, np.linalg.inv(P_y_min))[N - 1][N - 1]
    x_init = prediction - K_gain*(y_min - price_data[x])
    update.append(x_init)
    P_init = P_min - K_gain*P_min

mse = metrics.mean_squared_error(KF, price_data[0:price_points - N])
print('MSE: ' + str(mse))
mae = metrics.mean_absolute_error(KF, price_data[0:price_points - N])
print('MAE: ' + str(mae))
rmse = math.sqrt(metrics.mean_squared_error(KF, price_data[0:price_points - N]))
print('RMSE: ' + str(rmse))

end = time.time()

elapsed = end - start
print("Time elapsed: " + str(elapsed) +" s")

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 25, 10

plt.plot(price_data, label='test data')
plt.plot(KF,'green', label='Kalman filter prediction')
plt.title('Kalman filter')
plt.legend()
plt.show()