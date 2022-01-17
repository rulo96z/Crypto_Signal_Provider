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
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier

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
option = st.sidebar.selectbox("Dashboards", ('Top 10 Cryptocurrencies by Market Cap', 'Time-Series Forecasting - FB Prophet', 'Keras Model', 'Machine Learning Classifier - AdaBoost'))

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
        coin_list.index=pd.to_datetime(coin_list.index).date

    # Displays dataframe of selected cryptocurrency
    st.subheader(f"Selected Crypto:  {dropdown}")
    st.dataframe(coin_list)
    st.text("")

    # Display coin_list into a chart
    st.subheader(f'Selected Crypto Over Time: {dropdown}')
    st.line_chart(coin_list['Adj Close'])


# This option gives users the ability to use FB Prophet
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
    forecast_coin_trends_df.index=pd.to_datetime(forecast_coin_trends_df.index).date
    
    st.subheader(f'{dropdown} - Price Predictions')
    st.dataframe(forecast_coin_trends_df)

# This option gives users the ability to use Keras Model
if option == 'Keras Model':

    # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        #coin_list['Ticker'] = coin_choice # This tests that the selected ticker is displayed in the DataFrame

    # Preparing the data
    # Displays dataframe of selected cryptocurrency.  Isolated columns as trading features for forecasting cryptocurrency.
    st.subheader(f"Keras Model")
    st.subheader(f"Selected Crypto:  {dropdown}")
    coin_training_df = coin_list#[["Close", "High", "Low", "Open", "Volume"]]
    coin_training_df.index=pd.to_datetime(coin_training_df.index).date
    st.dataframe(coin_training_df)

    # Define the target set y using "Close" column
    y = coin_training_df["Close"]

    # Define the features set for X by selecting all columns but "Close" column
    X = coin_training_df.drop(columns=["Close"])

    # Split the features and target sets into training and testing datasets
    # Assign the function a random_state equal to 1
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the scaler to teh features training dataset
    X_scaler = scaler.fit(X_train)

    # Fit the scaler to the features training dataset
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Define the number of inputs (features) to the model
    number_input_features = len(X_train.iloc[0])

    # Define the number of neurons in the output layer
    st.write("Create Network:")
    number_output_neurons = st.number_input("Enter number of neurons in output layer", 1)

    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer1 = (number_input_features + number_output_neurons)//2

    # Define the number of hidden noes for the second hidden layer
    hidden_nodes_layer2 = (hidden_nodes_layer1 + number_output_neurons)//2
    
    # Create the Sequential model instance
    nn = Sequential()

    # User selects activation for 1st hidden layer
    first_activation = st.selectbox("Choose 1st hidden layer activation function", ('relu','sigmoid', 'tanh'))

    # Add the first hidden layer
    nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation=first_activation))

    # User selects activation for 2nd hidden layer
    second_activation = st.selectbox("Choose 2nd hidden layer activation function", ('relu',' '))

    # Add the second hidden layer
    nn.add(Dense(units=hidden_nodes_layer2,activation=second_activation))

    # User selects activation for output layer
    output_activation = st.selectbox("Choose output layer activation function", ('sigmoid',' '))

    # Add the output layer to the model specifying the number of output neurons and activation function
    nn.add(Dense(units=number_output_neurons, activation=output_activation))

    # Display the Sequential model summary - WHY IS THIS NONE
    # st.write(nn.summary())

    # Define functions
    loss = st.selectbox("Choose loss function", ('binary_crossentropy',' '))
    optimizer = st.selectbox("Choose optimizer", ('adam',' '))
    metrics = st.selectbox("Choose evaluation metric", ('accuracy',' '))

    # Compile the Sequential model
    nn.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    # Fit the model using 50 epochs and the training data
    epochs = st.number_input("Enter number of epochs", 50)
    epochs = int(epochs)

    fit_model=nn.fit(X_train_scaled, y_train, epochs=epochs) #ERROR STARTS HERE
    
    # Evaluate the model loss and accuracy metrics using the evaluate method and the test data
    model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose =2)
    
    # Display the model loss and accuracy results
    st.write(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# User selects AdaBoost
if option == 'Machine Learning Classifier - AdaBoost':
        # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        #coin_list['Ticker'] = coin_choice # This tests that the selected ticker is displayed in the DataFrame
        
    # Preparing the data
    # Displays dataframe of selected cryptocurrency.  Isolated columns as trading features for forecasting cryptocurrency.
    st.subheader(f"AdaBoost")
    st.subheader(f"Selected Crypto:  {dropdown}")
    coin_yf_df = coin_list.drop(columns='Adj Close')
    coin_yf_df_copy = coin_yf_df
    coin_yf_df.index=pd.to_datetime(coin_yf_df.index).date
    st.dataframe(coin_yf_df)

    # Filter the date index and close columns
    coin_signals_df = coin_yf_df_copy.loc[:,["Close"]]
    
    # Use the pct_change function to generate returns from close prices
    coin_signals_df["Actual Returns"] = coin_signals_df["Close"].pct_change()

    # Drop all NaN values from the DataFrame
    coin_signals_df = coin_signals_df.dropna()

    # Set the short window and long window
    short_window = st.number_input("Set a short window:", 4)
    short_window = int(short_window)

    long_window = st. number_input("Set a long window:", 100)
    long_window = int(long_window)

    # Generate the fast and slow simple moving averages
    coin_signals_df["SMA Fast"] = coin_signals_df["Close"].rolling(window=short_window).mean()
    coin_signals_df["SMA Slow"] = coin_signals_df["Close"].rolling(window=long_window).mean()

    coin_signals_df = coin_signals_df.dropna()

    # Initialize the new Signal Column
    coin_signals_df['Signal'] = 0.0
    coin_signals_df['Signal'] = coin_signals_df['Signal']
   
    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    coin_signals_df.loc[(coin_signals_df["Actual Returns"] >= 0), "Signal"] = 1

    # When Actual Returns are less than 0, generate signal to sell stock short
    coin_signals_df.loc[(coin_signals_df["Actual Returns"] < 0), "Signal"] = -1

    # Calculate the strategy returns and add them to the coin_signals_df DataFrame
    coin_signals_df["Strategy Returns"] = coin_signals_df["Actual Returns"] * coin_signals_df["Signal"].shift()

    # Plot Strategy Returns to examine performance
    st.write(f"{dropdown} Performance by Strategy Returns")
    st.line_chart ((1 + coin_signals_df['Strategy Returns']).cumprod())
 
    # Spril data into training and testing datasets
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = coin_signals_df[['SMA Fast', 'SMA Slow']].shift().dropna()

    # Create the target set selecting the Signal column and assigning it to y
    y = coin_signals_df["Signal"]

    # Select the start of the training period
    training_begin = X.index.min()

    # Select the ending period for the trianing data with an offet timeframe
    training_end = X.index.min() + DateOffset(months=6)

    # Generate the X_train and y_train DataFrame
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end+DateOffset(days=1):]
    y_test = y.loc[training_end+DateOffset(days=1):]


####

    # Scale the features DataFrame
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Apply the scaler model to fit the X_train data
    X_scaler = scaler.fit(X_train)

    # Transform the X_train and X_test DataFrame using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    

####
    # Initiate the AdaBoostClassifier model instance
    ab_model = AdaBoostClassifier()

    # Fit the model using the training data
    ab_model.fit(X_train,y_train)

    # Use the testing dataset to generate the predictions for the new model
    ab_y_pred = ab_model.predict(X_test)

    # Backtest the AdaBoost Model to evaluate performance
    ab_testing_report = classification_report(y_test,ab_y_pred)

    # Print the classification report
    st.write(ab_testing_report)

    # Create a new empty predictions DataFrame.

    # Create a predictions DataFrame
    alt_predictions_df = pd.DataFrame(index=X_test.index)

    # Add the SVM model predictions to the DataFrame
    alt_predictions_df['Predicted'] = ab_y_pred

    # Add the actual returns to the DataFrame
    alt_predictions_df['Actual Returns'] = coin_signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    alt_predictions_df['Strategy Returns'] = (alt_predictions_df['Actual Returns'] * alt_predictions_df['Predicted'])

    st.subheader(f"Predictions: {dropdown}")
    st.dataframe(alt_predictions_df)

    st.subheader(f"Actual Returns vs. Strategy Returns")
    st.line_chart((1 + alt_predictions_df[['Actual Returns','Strategy Returns']]).cumprod())