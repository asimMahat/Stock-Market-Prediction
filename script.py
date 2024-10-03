
import yfinance as yahooFinance
import pandas as pd

ibm = yahooFinance.Ticker("IBM")

data = ibm.history(period="max")
data.to_csv("ibm.csv")




ibm_data = pd.read_csv("ibm.csv")

print(ibm_data.shape[0])
