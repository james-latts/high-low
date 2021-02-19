#### IMPORTS & GLOBALS ####
# Imports
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statistics import NormalDist


#### MAIN ####
# Load in the xjo data
xjo_data_location = \
    "/home/james/Documents/finance/high-low/aud-nzd/data/minute/2020"
original_data_df = pd.DataFrame()
for xjo_csv_file in os.listdir(xjo_data_location):
    xjo_csv_location = xjo_data_location + "/" + xjo_csv_file
    csv_data_df = pd.read_csv(xjo_csv_location)
    original_data_df = pd.concat([original_data_df, csv_data_df])
del xjo_data_location, xjo_csv_file, xjo_csv_location, csv_data_df

# Create cleaned dataframe from original data
data_df = pd.DataFrame()
data_df["DateTime"] = pd.to_datetime(original_data_df["Time"])
data_df["DateTime"] = data_df["DateTime"] + dt.timedelta(hours=5) # CST to UTC
data_df["Price"] = original_data_df["Last"]
data_df["Log_Price"] = np.log(data_df["Price"])
data_df = data_df.sort_values("DateTime", ascending = True) # Order by date

# Add log price delta
data_df["Delta"] = data_df["Log_Price"].diff(1) * 100

# Get rid of first/last 30 minutes after/before open/close
data_df["Time"] = data_df["DateTime"].dt.time
data_df = data_df[np.invert((data_df["DateTime"].dt.dayofweek == 6) & \
    (data_df["Time"] < dt.time(20, 30, 0)))]
data_df = data_df[np.invert((data_df["DateTime"].dt.dayofweek == 4) & \
    (data_df["Time"] > dt.time(19, 30, 0)))]
data_df = data_df.drop("Time", axis = 1)
    
# Set the date as the index
data_df = data_df.set_index("DateTime")

# Plot the variables
data_df["Price"].reset_index(drop=True).plot()
data_df["Delta"].reset_index(drop=True).plot()

# Get the mean and std over the period
means = np.mean(data_df)
stds = np.std(data_df)

# Exact solution - GBM Model
seed = 11
np.random.seed(seed)

point = 0
points = len(data_df) - 1
sims = 10000
price = [data_df["Price"] for x in range(sims)]
delta = [data_df["Delta"] for x in range(sims)]

So = data_df["Price"][point] * np.ones(sims)
mu = means["Delta"] * np.ones(sims)
sigma = stds["Delta"] * np.ones(sims)
N = points
dt = 1./N

b = np.random.normal(0., 1., (int(N), sims))*np.sqrt(dt)
#b = np.random.standard_t(14, (int(N), sims))*np.sqrt(dt)
W = np.cumsum(b, axis = 0)

t = np.linspace(0, 1, N+1)
S = []
S.append(So * np.ones(sims))
for i in range(1, int(N+1)):
    drift = (mu - 0.5 * sigma**2) * t[i]
    diffusion = sigma * W[i-1]
    S_temp = So*np.exp(drift + diffusion)
    S.append(S_temp)

if sims <= 1000:
    plt.plot(t, S, label ='GBM')
    plt.plot(t, data_df["Price"][point:point+N+1], 'k--')

expected_value = np.mean(S[-1])
standard_deviation = np.std(S[-1], ddof = 1) # 1 degree of freedom as sample
confidence_interval = 1.96 # 95%
lower_bound = expected_value - confidence_interval * standard_deviation
upper_bound = expected_value + confidence_interval * standard_deviation

print("(lower_bound, expected, upper_bound)")
print("({}, {}, {})".format(lower_bound, expected_value, upper_bound))
print("actual_value = {}".format(data_df["Price"].reset_index(drop=True)[point+N-1]))

z_score = (So[0] - expected_value) / standard_deviation
prob =  round((1 - NormalDist().cdf(z_score)) * 100, 2)

print("Prob(ST>=S0) = {}%".format(prob))
