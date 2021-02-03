#### IMPORTS & GLOBALS ####
# Imports
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

#### MAIN ####
# Load in the xjo data
xjo_data_location = "/home/james/Documents/finance/high-low/data/xjo/1-minute/2020"
original_data_df = pd.DataFrame()
for xjo_csv_file in os.listdir(xjo_data_location):
    xjo_csv_location = xjo_data_location + "/" + xjo_csv_file
    csv_data_df = pd.read_csv(xjo_csv_location)
    original_data_df = pd.concat([original_data_df, csv_data_df])
del xjo_data_location, xjo_csv_file, xjo_csv_location, csv_data_df

# Create cleaned dataframe
data_df = pd.DataFrame()
data_df["DateTime"] = pd.to_datetime(original_data_df["Time"])
data_df["Price"] = original_data_df["Last"]
data_df["Log_Price"] = np.log(data_df["Price"])
data_df["Time"] = data_df["DateTime"].dt.time
data_df = data_df.sort_values("DateTime", ascending = True).set_index("DateTime")
#data_df['Hour'] = data_df['DateTime'].dt.hour
#data_df['Minute'] = data_df['DateTime'].dt.minute
#data_df["High"] = original_data_df["High"]
#data_df["Low"] = original_data_df["Low"]

# Clean dataframe
data_df = data_df[data_df.index.date != dt.date(2020, 11, 16)]
data_df = data_df[data_df["Time"] <= dt.time(16, 0, 0)]
#data_df = data_df[data_df["Time"] >= dt.time(11, 0, 0)]
data_df = data_df.drop("Time", axis = 1)

# Add additional features
data_df["Delta"] = data_df["Log_Price"].diff(1) * 100

# Clean dataframe after additional features added
data_df = data_df.dropna()

np.log(data_df["Price"][-1]/data_df["Price"][0])/len(data_df["Price"])

data_df.loc["2020-08-14","Price"].plot()
np.mean(data_df.loc["2020-08-14","Delta"])
(data_df.loc["2020-08-14","Delta"] - np.mean(data_df.loc["2020-08-14","Delta"])).plot()
np.mean(data_df.loc["2020-08-14","Delta"] - np.log(data_df["Price"][-1]/data_df["Price"][0])/len(data_df["Price"])*100)

data_df["Delta"].hist()
X = data_df["Delta"].values
split = round(len(X) // 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))


from statsmodels.tsa.stattools import adfuller
series = np.mean(data_df["Delta"] - np.mean(data_df["Delta"]))
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


print(np.std(data_df.loc["2020-08-14","Delta"], ddof=1))
data_df.loc["2020-08-14","Delta"].reset_index(drop=True).plot()

# Exact solution - GBM Model
seed = 11
np.random.seed(seed)

point = 12331
points = 1000
sims = 100
price = [data_df["Price"] for x in range(sims)]
delta = [data_df["Delta"] for x in range(sims)]

mean_log_return = data_df["Delta"].ewm(span=30).mean()
std_log_return = data_df["Delta"].ewm(span=30).std(dfof=1)

So = data_df["Price"][point] * np.ones(sims)
mu = mean_log_return[point] * np.ones(sims)
sigma = std_log_return[point] * np.ones(sims)
N = 15
dt = 1./N

b = np.random.normal(0., 1., (int(N), sims))*np.sqrt(dt)
W = np.cumsum(b, axis = 0)

t = np.linspace(0, 1, N+1)
S = []
S.append(So * np.ones(sims))
for i in range(1, int(N+1)):
    if i != 1:
        for n in range(sims):    
            price[n][point + i - 1] = S[-1][n]
            delta = np.log(price[n]).diff(1)
            mu[n] = delta.ewm(span=30).mean()[point + i - 1]
            sigma[n] = delta.ewm(span=30).std(dfof=1)[point + i - 1]
    drift = (mu - 0.5 * sigma**2) * t[i]
    diffusion = sigma * W[i-1]
    S_temp = So*np.exp(drift + diffusion)
    S.append(S_temp)

plt.plot(t, S, label ='GBM')
data_df["Price"].reset_index(drop=True)[point:point+N].plot()

So
np.mean(S[-1])
1.96*np.std(S[-1], ddof = 1)
data_df["Price"].reset_index(drop=True)[point+N]

mean = np.mean(S[-1])
confidence_interval = 2.576*np.std(S[-1], ddof = 1)


point = 15000
points = 100
price = [data_df["Price"] for x in range(sims)]
delta = [data_df["Delta"] for x in range(sims)]
pred = []
actual = []

mean_log_return = data_df["Delta"].ewm(span=30).mean()
std_log_return = data_df["Delta"].ewm(span=30).std(dfof=1)

for n in range(points):
    So = data_df["Price"][point + n] * np.ones(sims)
    mu = mean_log_return[point + n] * np.ones(sims)
    sigma = std_log_return[point + n] * np.ones(sims)
    N = 15
    sim = 100
    dt = 1./N
    
    b = np.random.normal(0., 1., (int(N), sim))*np.sqrt(dt)
    W = np.cumsum(b, axis = 0)
    
    t = np.linspace(0, 1, N+1)
    S = []
    S.append(So * np.ones(sim))
    for i in range(1, int(N+1)):
        if i != 1:
            for n in range(sims):    
                price[n][point + i - 1] = S[-1][n]
                delta = np.log(price[n]).diff(1)
                mu[n] = delta.ewm(span=30).mean()[point + i - 1]
                sigma[n] = delta.ewm(span=30).std(dfof=1)[point + i - 1]
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
        
    new_S = np.mean(S[-1])
    actual_S = data_df["Price"][point+n+N+1]
    pred.append(1 if new_S > So[0] else 0)
    actual.append(1 if actual_S > So[0] else 0)

print(sum(np.array(pred) == np.array(actual)))
