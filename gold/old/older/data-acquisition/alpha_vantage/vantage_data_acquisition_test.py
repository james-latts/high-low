#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:59:57 2020

@author: james
"""

# Imports
import csv
import requests
import pandas as pd

# Get the api key and request the data
api_path = "/home/james/Documents/finance/high_low/data_acquisition/alpha_vantage_key.txt"
with open(api_path, "r") as file:
    api_key = file.read().strip()
    
alpha_vantage_address = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=ASX:XJO&interval=1min&slice=year1month1&apikey={}".format(api_key)
download = requests.get(alpha_vantage_address)
decoded_content = download.content.decode('utf-8')
data = csv.reader(decoded_content.splitlines(), delimiter=',')

# Write the data to a csv and load into pandas
csv_path = "/home/james/Documents/finance/high_low/data_acquisition/data.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

data_df = pd.read_csv(csv_path)

