#### IMPORTS ####
import torch
import numpy as np
import pandas as pd


#### MAIN ####
data_location = "/home/james/Documents/finance/data-acquisition/bar_chart/axjo_daily_1-1-2020_12-20-2020.csv"
data_df = pd.read_csv(data_location)
data_df = data_df.drop("Volume", axis = 1)
data_df["Time"] = pd.to_datetime(data_df["Time"])

data_df["Diff"] = data_df["Change"] / data_df["Open"] * 100
data_df["y"] = (data_df["Change"] > 0).astype(np.int)

test_data = data_df.query('20200101 < Time')["Change"].values
val_data = data_df.query('20181231 < Time < 20200101')["Change"].values
train_data = data_df.query('20190101 > Time')["Change"].values

