# Crypto Signal Provider


![CRYPTO SIGNAL](images/crypto_image.jpg)


This is a web application that allows users to select a cryptocurrency, select a machine learning model, and pull historical data to predict its future price. The application was built with the idea of being interactive for the user. When possible we have tried to build a product that allows the user to play with parameters.  

Machine Learning Models included:

``` Time Series Forecasting - FB Prophet ```

``` LSTM Model ```

``` Keras & Deep Neural Network ```

``` AdaBoost ```

``` Support Vector Method ```

``` Logistic Regression ```


## Technologies

This Web App was written in Python 3.7 with the following packages and modules:

- Pandas
- numpy
- Streamlit
- yfinance
- PIL
- fbprophet
- babel.numbers
- tensorflow
- sklearn
- plotly
- matplotlib
- dotenv


## Installation Guide

This Web App was built with Python 3.7 in Jupyter Lab & VScode. If the User wants to interact with the Machine Learning Model, first install the following:
To get started using this application please go to [Python Download](https://www.python.org/downloads/) and select the version for your operating system. Then install the following libraries and packages.

``` sudo apt install python3-pip ```. This will install the pip that will make it easier to install the libraries.

``` pip install pandas ```

``` pip install numpy ```

``` pip install -U scikit-learn ```

``` pip install streamlit ```

``` python -m pip install -U matplotlib ```

``` pip install yfinance ```

``` pip install Pillow ```

``` pip install fbprophet ```

``` pip install hvplot ```

``` pip install --upgrade tensorflow ``` verify installation with ``` python -c "import tensorflow as tf;print(tf.__version__)" ```

``` pip install -U scikit-learn ```
  
``` pip install plotly-express ```

``` pip install python-dotenv ```


## Usage

The File of interest is labeled ``` crypto_signal_provider.py ```
To run the file open the terminal, navigate to the directory with crypto_signal_provider.py and run ``` streamlit run crypto_signal_provider.py ```
This web app is pulling historical data from the Nomics API, so please have your API key in a ``` .env ``` file in the working directory.
Have Fun...


## Contributors

This code was created in 2021 for a project at (Education Services at UCB). 

Additional updates/ uploads for usability was added by [Christina San Diego](mailto:cbuted@gmail.com)

Additional updates/ uploads for usability was added by [Stephen Thomas](mailto:stephenthomas43@gmail.com)

Additional updates/ uploads for usability was added by [Thomas Leahy](mailto:thomasleahy6@gmail.com)

Additional updates/ uploads for usability was added by [Raul Nogales](mailto:rulo.nogales@nogalesinvestments.com)

[Trilogy Education Services](https://www.trilogyed.com/)

[UC Berkeley Extension ](https://extension.berkeley.edu/)


## License

MIT License

Copyright (c) 2022 Raul 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


