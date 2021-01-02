#### IMPORTS & GLOBALS ####
# Imports
import os
import numpy as np
import pandas as pd
import datetime as dt

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

# Clean dataframe
data_df = data_df[data_df.index.date != dt.date(2020, 11, 16)]
data_df = data_df[data_df["Time"] <= dt.time(16, 0, 0)]
#data_df = data_df.drop("Time", axis = 1)

# Add additional features
data_df["Delta"] = data_df["Log_Price"].diff(1) * 100
data_df["Delta_15"] = data_df["Log_Price"].diff(15) * 100 / 15
data_df["Positive"] = (data_df["Delta_15"] > 0).astype(np.int)
data_df["Big_Positive"] = (data_df["Delta_15"] > np.mean(data_df["Delta_15"]) + 0.5 * np.std(data_df["Delta_15"])).astype(np.int)
(data_df["Delta_15"] - np.mean(data_df["Delta_15"])).hist()

big_positive_df = data_df[data_df["Big_Positive"].astype(np.bool)]
big_positive_df["Delta_15"].size
np.mean(big_positive_df["Delta_15"])
np.std(big_positive_df["Delta_15"])
big_positive_df["Delta_15"].hist()

positive_df = data_df[data_df["Positive"].astype(np.bool)]
positive_df["Delta_15"].size
np.mean(positive_df["Delta_15"])
np.std(positive_df["Delta_15"])
positive_df["Delta_15"].hist()

negative_df = data_df[np.invert(data_df["Positive"].astype(np.bool))]
negative_df["Delta_15"].size
np.mean(negative_df["Delta_15"])
np.std(negative_df["Delta_15"])
negative_df["Delta_15"].hist()


# Clean dataframe after additional features added
data_df = data_df.dropna()




#### Looking at time
np.sum(data_df["Big_Positive"]) / data_df["Big_Positive"].size

total = data_df[["Time", "Delta_15_Count"]].groupby("Time").count()
total.columns = ["Count"]
total["Positive"] = data_df[["Time", "Delta_15_Count"]].groupby("Time").sum()
total["Percentage"] = total["Positive"] / total["Count"] * 100
total["Mean"] = data_df[["Time", "Delta_15"]].groupby("Time").mean()
total["Std"] = data_df[["Time", "Delta_15"]].groupby("Time").std()

total["Percentage"].plot()
total["Mean"].plot()
total["Std"].plot()

train_start_date = "2020-09-01"
train_end_date = "2020-09-30"
train_df = data_df.loc[train_start_date:train_end_date]



test_start_date = "2020-10-01"
test_end_date =  "2020-10-31"
test_df = data_df.loc[test_start_date:test_end_date]

test_total = test_df[["Time", "Delta_15_Count"]].groupby("Time").count()
test_total.columns = ["Count"]
test_total["Positive"] = test_df[["Time", "Delta_15_Count"]].groupby("Time").sum()
test_total["Percentage"] = test_total["Positive"] / test_total["Count"] * 100
test_total["Mean"] = test_df[["Time", "Delta_15"]].groupby("Time").mean()
test_total["Std"] = test_df[["Time", "Delta_15"]].groupby("Time").std()

total["Percentage"].plot()
total["Mean"].plot()
total["Std"].plot()