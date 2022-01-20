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
import numpy as np
from tensorflow import keras
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# 2  PERFORM EXPLORATORY DATA ANALYSIS AND VISUALIZATION

# Function to normalize stock prices based on their initial price
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x

# Function to plot interactive plots using Plotly Express
print("Function to plot interactive plots using Plotly Express")
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()

# Function to concatenate the date, stock price, and volume in one dataframe
def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})

# Load .env environment variables
load_dotenv()

## Page expands to full width
st.set_page_config(layout='wide')

image = Image.open('images/crypto_image.jpg')
st.image(image,width = 600)

# Header for main and sidebar
st.title( "Crypto Signal Provider Web App")
st.markdown("""This app displays top 10 cryptocurrencies by market cap.""")
st.caption("NOTE: USDT & USDC are stablecoins pegged to the Dollar.")
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
option = st.sidebar.selectbox("Dashboards", ('Top 10 Cryptocurrencies by Market Cap', 'Time-Series Forecasting - FB Prophet', "LSTM Model", 'Keras Model', 'Machine Learning Classifier - AdaBoost', 'Support Vector Machines', 'Logistic Regression'))

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
    periods = st.number_input("Enter number of prediction days", 30)
    future_coin_trends = model_coin_trends.make_future_dataframe(periods = 30, freq='D')

    # Make the predictions for the trend data using the future_coin_trends DataFrame
    forecast_coin_trends = model_coin_trends.predict(future_coin_trends)

    # Plot the Prophet predictions for the Coin trends data
    st.markdown(f"Predictions Based on {dropdown} Trends Data")
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

    st.subheader("Price Prediction with Daily Seasonality")

    # Create the daily seasonality model
    model_coin_seasonality = Prophet(daily_seasonality=True)

    # Fit the model
    model_coin_seasonality.fit(coin_list_df)

    # Predict sales for ## of days out into the future
    # Start by making a future dataframe
    seasonality_periods = st.number_input("Enter number of future prediction days", 30)
    coin_trends_seasonality_future = model_coin_seasonality.make_future_dataframe (periods=seasonality_periods, freq='D')

    # Make predictions for each day over the next ## of days
    coin_trends_seasonality_forecast = model_coin_seasonality.predict(coin_trends_seasonality_future)

    seasonal_figs = model_coin_seasonality.plot_components(coin_trends_seasonality_forecast)
    st.pyplot(seasonal_figs)

    

# This option gives users the ability to use Keras Model
if option == 'Keras Model':

    # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)

    # Preparing the data
    # Displays dataframe of selected cryptocurrency.  Isolated columns as trading features for forecasting cryptocurrency.
    st.subheader(f"Keras Model")
    st.subheader(f"Selected Crypto:  {dropdown}")
    coin_training_df = coin_list
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

    # Define functions
    loss = st.selectbox("Choose loss function", ('binary_crossentropy',' '))
    optimizer = st.selectbox("Choose optimizer", ('adam',' '))
    metrics = st.selectbox("Choose evaluation metric", ('accuracy',' '))

    # Compile the Sequential model
    nn.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    # Fit the model using 50 epochs and the training data
    epochs = st.number_input("Enter number of epochs", 10)
    epochs = int(epochs)

    fit_model=nn.fit(X_train_scaled, y_train, epochs=epochs)
    
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
    coin_yf_df.index=pd.to_datetime(coin_yf_df.index).date
    st.dataframe(coin_yf_df)

    # Filter the date index and close columns
    coin_signals_df = coin_yf_df.loc[:,["Close"]]
    
    # Use the pct_change function to generate returns from close prices
    coin_signals_df["Actual Returns"] = coin_signals_df["Close"].pct_change()

    # Drop all NaN values from the DataFrame
    coin_signals_df = coin_signals_df.dropna()

    # Set the short window and long window
    short_window = st.number_input("Set a short window:", 2)
    short_window = int(short_window)

    long_window = st. number_input("Set a long window:", 10)
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
 
    # Split data into training and testing datasets
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = coin_signals_df[['SMA Fast', 'SMA Slow']].shift().dropna()

    # Create the target set selecting the Signal column and assigning it to y
    y = coin_signals_df["Signal"]

    st.subheader("Training Model")
    # Select the start of the training period
    st.caption(f'Training Begin Date starts at the selected "Start Date":  {start}')
    training_begin = X.index.min()


    # Select the ending period for the trianing data with an offet timeframe
    months = st.number_input("Enter number of months for DateOffset", 1)
    training_end = X.index.min() + DateOffset(months=months)
    st.caption(f'Training End Date ends:  {training_end}')

    # Generate the X_train and y_train DataFrame
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end+DateOffset(days=1):]
    y_test = y.loc[training_end+DateOffset(days=1):]

    # Scale the features DataFrame
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Apply the scaler model to fit the X_train data
    X_scaler = scaler.fit(X_train)

    # Transform the X_train and X_test DataFrame using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Initiate the AdaBoostClassifier model instance
    ab_model = AdaBoostClassifier()

    # Fit the model using the training data
    ab_model.fit(X_train,y_train)

    # Use the testing dataset to generate the predictions for the new model
    ab_y_pred = ab_model.predict(X_test)

    # Backtest the AdaBoost Model to evaluate performance
    st.write('**AdaBoost Testing Classification Report**')
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


            
            #### SUPPORT VECTOR MACHINES ####

    # This option gives users the ability to use SVM model
if option == 'Support Vector Machines':
       # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        
    # Displays dataframe of selected cryptocurrency.
    st.subheader(f'Support Vector Machines')
    st.subheader(f"Selected Crypto:  {dropdown}")
    coin_yf_df = coin_list.drop(columns='Adj Close')
    coin_yf_df_copy = coin_yf_df
    coin_yf_df.index=pd.to_datetime(coin_yf_df.index).date
    st.dataframe(coin_yf_df)

    # Filter the date index and close columns
    coin_signals_df = coin_yf_df_copy.loc[:,["Close"]]
    
    # Use the pct_change function to generate  returns from close prices
    coin_signals_df["Actual Returns"] = coin_signals_df["Close"].pct_change()

    # Drop all NaN values from the DataFrame
    coin_signals_df = coin_signals_df.dropna()

    # Generate the fast and slow simple moving averages (user gets to select window size)
    short_window = st.number_input("Set a short window:", 4)
    short_window = int(short_window)

    long_window = st. number_input("Set a long window:", 85)
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
 
    
    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    svm_X = coin_signals_df[['SMA Fast', 'SMA Slow']].shift().dropna()

    # Create the target set selecting the Signal column and assigning it to y
    svm_y = coin_signals_df["Signal"]

    # Select the start of the training period
    svm_training_begin = svm_X.index.min()
   
    
    #### Setting the training data to three or above 3 seems to throw off callculations where one signal or the other isnt predicted #### 
    #### Is there a way to make this a selection the user makes? ####

    # Select the ending period for the training data with an offset of 2 months
    months = st.number_input("Enter number of months for DateOffset", 2)
    svm_training_end = svm_X.index.min() + DateOffset(months=months)
    st.caption(f'Training End Date ends:  {svm_training_end}')
                                    
    # Generate the X_train and y_train DataFrame
    svm_X_train = svm_X.loc[svm_training_begin:svm_training_end]
    svm_y_train = svm_y.loc[svm_training_begin:svm_training_end]

    # Generate the X_test and y_test DataFrames
    svm_X_test = svm_X.loc[svm_training_end+DateOffset(days=1):]
    svm_y_test = svm_y.loc[svm_training_end+DateOffset(days=1):]

    # Scale the features DataFrame with StandardScaler
    svm_scaler = StandardScaler()

    # Apply the scaler model to fit the X_train data
    svm_X_scaler = svm_scaler.fit(svm_X_train)

    # Transform the X_train and X_test DataFrame using the X_scaler
    svm_X_train_scaled = svm_X_scaler.transform(svm_X_train)
    svm_X_test_scaled = svm_X_scaler.transform(svm_X_test)
    
    ## From SVM, instantiate SVC classifier model instance
    svm_model = svm.SVC()

    # Fit the model with the training data
    svm_model.fit(svm_X_train,svm_y_train)

    # Use the testing dataset to generate the predictions
    svm_y_pred = svm_model.predict(svm_X_test)

    # Use a classification report to evaluate the model using the predictions and testing data
    st.write('**Support Vector Machines Classification Report**')
    svm_testing_report = classification_report(svm_y_test,svm_y_pred)

    # Print the classification report
    st.write(svm_testing_report)

    # Create a predictions DataFrame
    svm_predictions_df = pd.DataFrame(index=svm_X_test.index)

    # Add the SVM model predictions to the DataFrame
    svm_predictions_df['Predicted'] = svm_y_pred

    # Add the actual returns to the DataFrame
    svm_predictions_df['Actual Returns'] = coin_signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    svm_predictions_df['Strategy Returns'] = (svm_predictions_df['Actual Returns'] * svm_predictions_df['Predicted'])

    st.subheader(f"Predictions: {dropdown}")
    st.dataframe(svm_predictions_df)

    st.subheader(f"Actual Returns vs. Strategy Returns")
    st.line_chart((1 + svm_predictions_df[['Actual Returns','Strategy Returns']]).cumprod())
    
      
        #### LSTM Model ####

    # This option gives users the ability to use LSTM model
if option == 'LSTM Model':
    # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        

    # Preparing the data
    # Displays dataframe of selected cryptocurrency.  Isolated columns as trading features for forecasting cryptocurrency.
    st.subheader(f"LSTM Model")
    st.subheader(f"Selected Crypto:  {dropdown}")
    coin_training_df = coin_list#[["Close", "High", "Low", "Open", "Volume"]]
    coin_training_df.index=pd.to_datetime(coin_training_df.index).date
    coin_training_df["Date"]=pd.to_datetime(coin_training_df.index).date
    st.dataframe(coin_training_df)
    
    
    stock_price_df = coin_training_df

    # Read the stocks volume data
    #stock_vol_df = pd.read_csv("stock.csv")
    stock_vol_df = coin_training_df

    # Sort the data based on Date
    stock_price_df = stock_price_df.sort_values(by = ['Date'])

    # Sort the data based on Date
    stock_vol_df = stock_vol_df.sort_values(by = ['Date'])

    # Check if Null values exist in stock prices data
    stock_price_df.isnull().sum()

    # Check if Null values exist in stocks volume data
    stock_vol_df.isnull().sum()


    # 4 TRAIN AN LSTM TIME SERIES MODEL

    # Let's test the functions and get individual stock prices and volumes
    price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'Close')

    # Get the close and volume data as training data (Input)
    training_data = price_volume_df.iloc[:, 1:3].values

    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_data)

    # Create the training and testing data, training data contains present day and previous day values
    X = []
    y = []
    for i in range(1, len(price_volume_df)):
        X.append(training_set_scaled [i-1:i, 0])
        y.append(training_set_scaled [i, 0])

    # Convert the data into array format
    X = np.asarray(X)
    y = np.asarray(y)

    # Split the data for training, the rest for testing.  
    split = int(0.7 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # Reshape the 1D arrays to 3D arrays to feed in the model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train.shape, X_test.shape

    # Create the model
    inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = keras.layers.LSTM(150, return_sequences= True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150, return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150)(x)
    outputs = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss="mse")
    model.summary()

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs = 20,
        batch_size = 32,
        validation_split = 0.2)

    # Make prediction
    predicted = model.predict(X)

    # Append the predicted values to the list
    test_predicted = []
    for i in predicted:
        test_predicted.append(i[0])

    # We take the median loss to evaluate the model
    media = sum(history.history["loss"]) / len(history.history["loss"])
    
    #############################################################################################################################
    # Step 5
    # Creando una tabla con las proyecciones
    df_predicted = price_volume_df[1:][['Date']]

    # Creando la columna de Predicciones
    df_predicted['predictions'] = test_predicted

    # Plot the data
    close = []
    for i in training_set_scaled:
        close.append(i[0])

    # Juntando la tabla final con (Dia, Prediccion y Cierre)
    df_predicted['Close'] = close[1:]

    # Plot the data
    #interactive_plot(df_predicted, "Original Vs Prediction")
    st.write(f"Evaluation loss {round(media*100, 6)}%")
    graphic_data = {}
    graphic_data["Actual"]=df_predicted["Close"]
    graphic_data["Prediction"] = df_predicted["predictions"]
    st.line_chart(graphic_data)
    st.subheader("Model Summary")
    st.text("""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 1, 1)]            0

 lstm (LSTM)                 (None, 1, 150)            91200

 dropout (Dropout)           (None, 1, 150)            0

 lstm_1 (LSTM)               (None, 1, 150)            180600

 dropout_1 (Dropout)         (None, 1, 150)            0

 lstm_2 (LSTM)               (None, 150)               180600

 dense (Dense)               (None, 1)                 151

=================================================================
Total params: 452,551
Trainable params: 452,551
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20

 1/14 [=>............................] - ETA: 56s - loss: 0.0290
 7/14 [==============>...............] - ETA: 0s - loss: 0.0508
14/14 [==============================] - ETA: 0s - loss: 0.0423
14/14 [==============================] - 6s 92ms/step - loss: 0.0423 - val_loss: 0.3835
Epoch 2/20

 1/14 [=>............................] - ETA: 0s - loss: 0.0472
 8/14 [================>.............] - ETA: 0s - loss: 0.0248
13/14 [==========================>...] - ETA: 0s - loss: 0.0237
14/14 [==============================] - 0s 11ms/step - loss: 0.0241 - val_loss: 0.1809
Epoch 3/20

 1/14 [=>............................] - ETA: 0s - loss: 0.0254
 7/14 [==============>...............] - ETA: 0s - loss: 0.0204
13/14 [==========================>...] - ETA: 0s - loss: 0.0195
14/14 [==============================] - 0s 10ms/step - loss: 0.0196 - val_loss: 0.1402
Epoch 4/20

 1/14 [=>............................] - ETA: 0s - loss: 0.0219
 7/14 [==============>...............] - ETA: 0s - loss: 0.0119
14/14 [==============================] - 0s 10ms/step - loss: 0.0100 - val_loss: 0.0500
Epoch 5/20

 1/14 [=>............................] - ETA: 0s - loss: 0.0049
 7/14 [==============>...............] - ETA: 0s - loss: 0.0030
14/14 [==============================] - 0s 12ms/step - loss: 0.0022 - val_loss: 0.0062
Epoch 6/20

 1/14 [=>............................] - ETA: 0s - loss: 1.8830e-04
 8/14 [================>.............] - ETA: 0s - loss: 5.7900e-04
14/14 [==============================] - 0s 10ms/step - loss: 6.1269e-04 - val_loss: 0.0046
Epoch 7/20

 1/14 [=>............................] - ETA: 0s - loss: 3.8925e-04
 6/14 [===========>..................] - ETA: 0s - loss: 6.3538e-04
13/14 [==========================>...] - ETA: 0s - loss: 5.2828e-04
14/14 [==============================] - 0s 11ms/step - loss: 5.2585e-04 - val_loss: 0.0015
Epoch 8/20

 1/14 [=>............................] - ETA: 0s - loss: 2.9108e-04
 9/14 [==================>...........] - ETA: 0s - loss: 4.7043e-04
14/14 [==============================] - 0s 10ms/step - loss: 4.1497e-04 - val_loss: 0.0029
Epoch 9/20

 1/14 [=>............................] - ETA: 0s - loss: 7.6592e-04
 8/14 [================>.............] - ETA: 0s - loss: 4.3651e-04
14/14 [==============================] - 0s 10ms/step - loss: 5.1143e-04 - val_loss: 0.0046
Epoch 10/20

 1/14 [=>............................] - ETA: 0s - loss: 4.3880e-04
 8/14 [================>.............] - ETA: 0s - loss: 6.3981e-04
14/14 [==============================] - 0s 9ms/step - loss: 5.8694e-04 - val_loss: 0.0021
Epoch 11/20

 1/14 [=>............................] - ETA: 0s - loss: 1.9836e-04
 9/14 [==================>...........] - ETA: 0s - loss: 7.9962e-04
14/14 [==============================] - 0s 9ms/step - loss: 6.7786e-04 - val_loss: 0.0023
Epoch 12/20

 1/14 [=>............................] - ETA: 0s - loss: 5.7881e-04
 8/14 [================>.............] - ETA: 0s - loss: 5.3330e-04
14/14 [==============================] - 0s 11ms/step - loss: 5.5330e-04 - val_loss: 0.0029
Epoch 13/20

 1/14 [=>............................] - ETA: 0s - loss: 3.2055e-04
 8/14 [================>.............] - ETA: 0s - loss: 4.2544e-04
14/14 [==============================] - 0s 10ms/step - loss: 4.5558e-04 - val_loss: 0.0045
Epoch 14/20

 1/14 [=>............................] - ETA: 0s - loss: 3.6282e-04
10/14 [====================>.........] - ETA: 0s - loss: 7.6032e-04
14/14 [==============================] - 0s 9ms/step - loss: 6.7844e-04 - val_loss: 0.0021
Epoch 15/20

 1/14 [=>............................] - ETA: 0s - loss: 5.4045e-04
 8/14 [================>.............] - ETA: 0s - loss: 7.3637e-04
14/14 [==============================] - 0s 10ms/step - loss: 6.6167e-04 - val_loss: 0.0014
Epoch 16/20

 1/14 [=>............................] - ETA: 0s - loss: 0.0010
 7/14 [==============>...............] - ETA: 0s - loss: 5.2453e-04
13/14 [==========================>...] - ETA: 0s - loss: 4.9783e-04
14/14 [==============================] - 0s 12ms/step - loss: 4.9552e-04 - val_loss: 0.0019
Epoch 17/20

 1/14 [=>............................] - ETA: 0s - loss: 5.9480e-04
 7/14 [==============>...............] - ETA: 0s - loss: 6.7032e-04
13/14 [==========================>...] - ETA: 0s - loss: 5.8481e-04
14/14 [==============================] - 0s 11ms/step - loss: 5.8287e-04 - val_loss: 0.0036
Epoch 18/20

 1/14 [=>............................] - ETA: 0s - loss: 8.9031e-04
 8/14 [================>.............] - ETA: 0s - loss: 4.5292e-04
14/14 [==============================] - 0s 11ms/step - loss: 4.2416e-04 - val_loss: 0.0018
Epoch 19/20

 1/14 [=>............................] - ETA: 0s - loss: 1.5390e-04
10/14 [====================>.........] - ETA: 0s - loss: 4.6502e-04
14/14 [==============================] - 0s 10ms/step - loss: 5.2157e-04 - val_loss: 0.0034
Epoch 20/20

 1/14 [=>............................] - ETA: 0s - loss: 7.4292e-04
 8/14 [================>.............] - ETA: 0s - loss: 5.8960e-04
14/14 [==============================] - 0s 10ms/step - loss: 5.7970e-04 - val_loss: 0.0024""")


    # Logistic Regresion 

if option == 'Logistic Regression':
    
    top_cryptos_df.Logo = path_to_image_html(top_cryptos_df.Logo)
    st.write(top_cryptos_df.to_html(escape=False), unsafe_allow_html=True)
    st.text("")

    st.subheader("Performance Forecasting - Logistic Regression Model")

    # Line charts are created based on dropdown selection
    if len(dropdown) > 0:
        coin_choice = dropdown[0] 
        coin_list = yf.download(coin_choice,start,end)
        coin_list['Ticker'] = coin_choice
    
    # Find 'Act Returns' and assign to new column.
    coin_list_df = coin_list
    coin_list_df['Actual Returns'] = coin_list_df['Adj Close'].pct_change()
    
    # Set 'Short' and 'Long' window
    short_window = 4
    long_window = 50
    
    # Find the rolling average and assign to new columns labeled SMA_Fast, SMA_Slow.
    coin_list_df['SMA_Slow'] = coin_list_df['Adj Close'].rolling(window=short_window).mean()
    coin_list_df['SMA_Fast'] = coin_list_df['Adj Close'].rolling(window=long_window).mean()
    
    # Assign SMA columns to X
    X = coin_list_df[['SMA_Fast', 'SMA_Slow']].shift().dropna().copy()

    # Create a new column named 'Signal' as a place holder.
    coin_list_df['Signal'] = 0.0
    
    # Set signals based on actual returns being greater or less than 0.
    coin_list_df.loc[(coin_list_df['Actual Returns'] >= 0), 'Signal'] = 1
    coin_list_df.loc[(coin_list_df['Actual Returns'] < 0), 'Signal'] = -1
    
    # Set y dataset to 'Signal' column and drop nan values.
    y = coin_list_df['Signal'].dropna()
    y = y.iloc[50:]
    coin_list_df.dropna()

    # Import train_test_split function to separate data into X and y datasets.
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)
    
    # Import StandardScaler() function and scale X_train and X_test datasets.
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Import LR and instantiate a model.
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    testing_predictions = lr_model.predict(X_test)

    
    # Import and print accuracy report.
    accuracy = accuracy_score(y_test, testing_predictions)
    res = round(accuracy,1) * 100
    st.subheader(f"Accuracy of the Logistic Regression Model is {res} %")
    st.write(accuracy)

    # Import and print classification report.
    training_report = classification_report(y_test, testing_predictions)
    res_1 = training_report
    st.subheader(f"{res_1}")
    
    testing_matrix = confusion_matrix(y_test, testing_predictions)
    # Print confusion matrix.
    st.subheader(f"Confusion Matrix")
    st.text("The Confusion Matrix is a table that allows visualization of the performance of an algorithm")
    st.caption("Predicted True / Actually True                Predicted False / Actually True")
    st.caption("Predicted False / Actually True               Predicted False / Actually False")
    st.write(testing_matrix)
    
    
    # Comments on Conclusion ect.
    st.subheader("Conclusions")
    st.text("""

    For this Logistic Regression Model I used a 4 & 100 day Moving Average as the feature, and a buy and sell signal 
    as the target. The Signals were defined as: days where the percent change was greater than or equal to 0 as a buy and less 
    than 0 as a sell. Its clear that the accuracy of this model is relativeley low, around 50%. I think its important to note that
    the Model is not a bad model, its the data that is the cause of the low accuracy. Logistic Regression does a good job of 
    classifying binary outcomes, when the features are robust and multi-dimensional. It has become clear that using a simple moving 
    average as the feature is not going to get great prediction results as far as price. I think that a Logistic Regression model 
    could work better with several classification features. One example of this could be Sentiment that is collected via twitter 
    or Social Media. Tracking if people are mentioning via social media whether they are 'Bullish' or 'Bearish' and then training 
    a LR model to this data could be interesting. Clearly this model is lacking features. With the accuracy of a model is 50% I 
    think its important to recognize that it is more an accuracy score of the data that was trained with.
    Data Scientists today spend much time engineering features, in fact according to a Forbes survey, data scientists spend 80%
    of their time on data preperation including feature engineering. Moving forward with this model I think it could benefit 
    greatly from collecting robust features.
    
    
    """)