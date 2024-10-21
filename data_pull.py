import yfinance as yahooFinance
import pandas as pd

ibm = yahooFinance.Ticker("IBM")
data = ibm.history(period="max")

data.to_csv("dataset/ibm.csv")
ibm_data = pd.read_csv("dataset/ibm.csv")
print(ibm_data.shape[0])
