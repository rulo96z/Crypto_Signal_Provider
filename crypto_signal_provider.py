import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from nomics import Nomics
import json
import plotly
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
from fbprophet import Prophet
import hvplot as hv
import hvplot.pandas 
import datetime as dt
from babel.numbers import format_currency


# Load .env environment variables
load_dotenv()

## Page expands to full width
st.set_page_config(layout='wide')

image = Image.open('images/crypto_image.jpg')
st.image(image,width = 600)

# Header for main and sidebar
st.title( "Crypto Signal Provider Web App")
st.markdown("""This app displays top 10 cryptocurrencies by market cap.""")
st.sidebar.title("Crypto Signal Settings")

# Get nomics api key
nomics_api_key = os.getenv("NOMICS_API_KEY")
nomics_url = "https://api.nomics.com/v1/prices?key=" + nomics_api_key
nomics_currency_url = ("https://api.nomics.com/v1/currencies/ticker?key=" + nomics_api_key + "&interval=1d,30d&per-page=10&page=1")

# Read API in json
nomics_df = pd.read_json(nomics_currency_url)

# Create an empty DataFrame for top cryptocurrencies by market cap
top_cryptos_df = pd.DataFrame()

# Get rank, crytocurrency, price, price_date, market cap
top_cryptos_df = nomics_df[['rank', 'logo_url', 'name', 'currency', 'price', 'price_date', 'market_cap']]

# This code gives us the sidebar on streamlit for the different dashboards
option = st.sidebar.selectbox("Dashboards", ('Top 10 Cryptocurrencies by Market Cap', 'Time-Series Forecasting - FB Prophet', '3rd Dashboard'))

# Rename column labels
columns=['Rank', 'Logo', 'Currency', 'Symbol', 'Price (USD)', 'Price Date', 'Market Cap']
top_cryptos_df.columns=columns

# Set rank as index
top_cryptos_df.set_index('Rank', inplace=True)

# Convert text data type to numerical data type
top_cryptos_df['Market Cap'] = top_cryptos_df['Market Cap'].astype('int')

# Convert Timestamp to date only
top_cryptos_df['Price Date']=pd.to_datetime(top_cryptos_df['Price Date']).dt.date

# Replace nomics ticker symbol with yfinance ticker symbol
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("LUNA","LUNA1")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("FTXTOKEN","FTT")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("UNI","UNI1")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("AXS2","AXS")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("SAND2","SAND")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("HARMONY","ONE1")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("HELIUM","HNT")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("GRT","GRT1")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("IOT","MIOTA")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("BLOCKSTACK","STX")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("FLOW2","FLOW")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("BITTORRENT","BTT")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("AMP2","AMP")
top_cryptos_df.loc[:,"Symbol"] = top_cryptos_df.loc[:,"Symbol"].str.replace("HOT","HOT1")

# Format Market Cap with commas to separate thousands
top_cryptos_df["Market Cap"] = top_cryptos_df.apply(lambda x: "{:,}".format(x['Market Cap']), axis=1)

# Formatting Price (USD) to currency
top_cryptos_df["Price (USD)"] = top_cryptos_df["Price (USD)"].apply(lambda x: format_currency(x, currency="USD", locale="en_US"))

# Convert your links to html tags 
def path_to_image_html(Logo):
    return '<img src="'+ Logo +'" width=30 >'

# Pulls list of cryptocurrencies from nomics and concatenates to work with Yahoo Finance
coin = top_cryptos_df['Symbol'] + "-USD"


# Creates a dropdown list of cryptocurrencies based on top 100 list
dropdown = st.sidebar.multiselect("Select 1 coin to analyze", coin, default=['SOL-USD'])

# Create start date for analysis
start = st.sidebar.date_input('Start Date', value = pd.to_datetime('2020-01-01'))

# Create end date for analysis
end = st.sidebar.date_input('End Date', value = pd.to_datetime('today'))

# This option gives users the ability to view the current top 100 cryptocurrencies
if option == 'Top 10 Cryptocurrencies by Market Cap':

    # Displays image in dataframe
    top_cryptos_df.Logo = path_to_image_html(top_cryptos_df.Logo)
    st.write(top_cryptos_df.to_html(escape=False), unsafe_allow_html=True)
    st.text("")
        
    # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        coin_list['Ticker'] = coin_choice

    # Displays dataframe of selected cryptocurrency
    st.subheader(f"Selected Crypto:  {dropdown}")
    st.dataframe(coin_list)
    st.text("")

    # Display coin_list into a chart
    st.subheader(f'Selected Crypto Over Time: {dropdown}')
    st.line_chart(coin_list['Adj Close'])


# This option gives users the ability to use sklearn
if option == 'Time-Series Forecasting - FB Prophet':

    st.subheader("Time-Series Forecasting - FB Prophet")

    # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        coin_list['Ticker'] = coin_choice

    # Reset the index so the date information is no longer the index
    coin_list_df = coin_list.reset_index().filter(['Date','Adj Close'])
    
    # Label the columns ds and y so that the syntax is recognized by Prophet
    coin_list_df.columns = ['ds','y']
    
    # Drop NaN values form the coin_list_df DataFrame
    coin_list_df = coin_list_df.dropna()

    # Call the Prophet function and store as an object
    model_coin_trends = Prophet()

    # Fit the time-series model
    model_coin_trends.fit(coin_list_df)

    # Create a future DataFrame to hold predictions
    # Make the prediction go out as far as 60 days
    future_coin_trends = model_coin_trends.make_future_dataframe(periods = 60, freq='D')

    # Make the predictions for the trend data using the future_coin_trends DataFrame
    forecast_coin_trends = model_coin_trends.predict(future_coin_trends)

    # Plot the Prophet predictions for the Coin trends data
    st.pyplot(model_coin_trends.plot(forecast_coin_trends));

    # Set the index in the forecast_coin_trends DataFrame to the ds datetime column
    forecast_coin_trends = forecast_coin_trends.set_index('ds')
    
    # View only the yhat,yhat_lower and yhat_upper columns in the DataFrame
    forecast_coin_trends_df = forecast_coin_trends[['yhat', 'yhat_lower', 'yhat_upper']]

    # From the forecast_coin_trends_df DataFrame, rename columns
    coin_columns=['Most Likely (Average) Forecast', 'Worst Case Prediction', 'Best Case Prediction']
    forecast_coin_trends_df.columns=coin_columns
    
    st.subheader(f'{dropdown} - Price Predictions')
    st.dataframe(forecast_coin_trends_df)