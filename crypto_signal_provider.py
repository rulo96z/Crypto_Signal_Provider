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

# Load .env environment variables
load_dotenv()

# Header for main and sidebar
st.title( "Crypto Signal Provider")
st.sidebar.title("Options")

# Get nomics api key
nomics_api_key = os.getenv("NOMICS_API_KEY")
nomics_url = "https://api.nomics.com/v1/prices?key=" + nomics_api_key
nomics_currency_url = ("https://api.nomics.com/v1/currencies/ticker?key=" + nomics_api_key + "&interval=1d,30d&per-page=10&page=1")

# Read API in json
nomics_df = pd.read_json(nomics_currency_url)

# Create an empty DataFrame for top cryptocurrencies
top_cryptos_df = pd.DataFrame()

# Get rank, crytocurrency, price, price_date, market cap
top_cryptos_df = nomics_df[['rank', 'logo_url', 'currency', 'name', 'price', 'price_date', 'market_cap']]

# This code gives us the sidebar on streamlit for the different dashboards
option = st.sidebar.selectbox("Dashboards", ('Top 10 Cryptocurrencies by Market Cap', '2nd Dashboard', '3rd Dashboard'))
#option_1 = st.sidebar.text_input("coin", value="{symbol}", max_chars=5)
# This is the Header for each page
st.header(option)

# This code gives us the Widget, for now its just an example but we can integrate into the Monte carlo simulation by nesting the code for Monte Carlo into this command.
# num_days = st.sidebar.slider('Amount to Invest', 1, 100000, 10)

# This option gives users the ability to view the current top 100 cryptocurrencies
if option == 'Top 10 Cryptocurrencies by Market Cap':

    # Create an empty DataFrame for top cryptocurrencies
    top_cryptos_df = pd.DataFrame()

    # Get rank, crytocurrency, price, price_date, market cap
    top_cryptos_df = nomics_df[['rank', 'logo_url', 'currency', 'name', 'price', 'price_date', 'market_cap']]

    # Rename column labels
    columns=['Rank', 'Logo', 'Symbol', 'Currency', 'Price (USD)', 'Price Date', 'Market Cap']
    top_cryptos_df.columns=columns

    # Set rank as index
    top_cryptos_df.set_index('Rank', inplace=True)

    # Convert to Price and Market Cap to currency format
    pd.options.display.float_format = '${:,.2f}'.format

    # Convert text data type to numerical data type
    top_cryptos_df['Market Cap'] = top_cryptos_df['Market Cap'].astype('float64')

    # Convert Timestamp to date only
    top_cryptos_df['Price Date']=pd.to_datetime(top_cryptos_df['Price Date']).dt.date

    # Convert your links to html tags 
    def path_to_image_html(Logo):
        return '<img src="'+ Logo +'" width=30 >'
        
    # Display image in dataframe
    top_cryptos_df.Logo = path_to_image_html(top_cryptos_df.Logo)
    st.write(top_cryptos_df.to_html(escape=False), unsafe_allow_html=True)


    # Pulls list of cryptocurrencies from Alpaca
    coin = top_cryptos_df['Symbol'] + "-USD"

    # Creates a dropdown list of cryptocurrencies based on top 100 list
    dropdown = st.multiselect("Select coin(s) to analyze against USD", coin)

    # Create start date for analysis
    start = st.date_input('Start', value = pd.to_datetime('today'))

    # Create end date for analysis
    end = st.date_input('End', value = pd.to_datetime('today'))

    # Line charts are created based on dropdown selection
    if len(dropdown) > 0: 
        coin_list = yf.download(dropdown,start,end)['Adj Close']
        # st.write('Selected list of cryptocurrencies')
        # st.write(coin_list)

        crypto_ticker = coin
        coin_list_df = list()
        for crypto in crypto_ticker:
            data = yf.download(dropdown,start,end)
            data['ticker'] = crypto  # add this column because the dataframe doesn't contain a column with the ticker
            coin_list_df.append(data)

        # combine all dataframes into a single dataframe
        crypto_ml_df = pd.concat(coin_list_df)


        # Display selected crypto into dataframe
        st.write('Selected Cryptocurrency in USD')
        st.dataframe(crypto_ml_df)

        # Display coin chart
        st.write('Selected Cryptocurrency Over Time')
        st.line_chart(coin_list)