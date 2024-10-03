
import yfinance as yahooFinance
 
facebook = yahooFinance.Ticker("META")

data = facebook.history(period="max")
data.to_csv("facebook_meta_history.csv")




