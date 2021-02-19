#### IMPORTS & GLOBALS ####
# Imports
import os
import numpy as np
import pandas as pd

#### MAIN ####
# Load in the xjo data
xjo_data_location = "/home/james/Documents/finance/high-low/aud-nzd/data/daily"
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
data_df["Low"] = original_data_df["Low"]
data_df["High"] = original_data_df["High"]
data_df["Log_Price"] = np.log(data_df["Price"])
data_df["Log_Low"] = np.log(data_df["Low"])
data_df["Log_High"] = np.log(data_df["High"])
data_df = data_df.sort_values("DateTime", ascending = True).set_index("DateTime")

# Add additional features
data_df["Delta"] = data_df["Log_Price"].diff(1) * 100
data_df["Range"] = data_df["Log_High"] - data_df["Log_Low"]

data_df["Price"].plot()
data_df["Delta"].plot()
data_df["Range"].plot()

sum(abs(data_df["Delta"]) > 1)

num_days = len(data_df)

np.mean(data_df)
np.std(data_df)

np.mean(data_df.iloc[0: num_days//2])
np.std(data_df.iloc[0: num_days//2])

np.mean(data_df.iloc[num_days//2: num_days])
np.std(data_df.iloc[num_days//2: num_days])

data_df["Delta"].hist()
data_df["Delta"].hist()

data_df.iloc[num_days//2: num_days]["Delta"].hist()


data_df["Positive"] = (data_df["Delta"] > 0).astype(np.int)
data_df["Big_Positive"] = (data_df["Delta"] > np.mean(data_df["Delta"]) + 1.645 * np.std(data_df["Delta"])).astype(np.int)


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