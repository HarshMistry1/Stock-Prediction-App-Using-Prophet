# Importing Necessary Libraries
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from datetime import date
import streamlit as st
from plotly import graph_objs as go
import yfinance as yf


# Initializing dates from start to end
START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')


# Setting up an applcaition title
st.title('Stock Prediction and Forecasting')


# Specific stock names according to Yahoo Finance
stocks = ('GOOG', 'MSFT', 'GME')
selected_stocks = st.selectbox('Select stock for prediction', stocks)


# Creating slider for increase in years of prediction
n_years = st.slider('Years Of Prediction:', 1, 4)
period = n_years*365


# Downloading the data from Yahoo Finance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Load data ...')
data = load_data(selected_stocks)
data_load_state.text('Loading data done.')



# Visualization of downloaded data
st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))


plot_raw_data()


# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date':'ds', 'Close': 'y'})


# Initializing the model
model = Prophet()
model.fit(df_train)


# Making the predictions from model
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecasted Data')
st.write(forecast.tail())


# Plotting forecasted data along with their components
st.write('Forecast data')
fig_1 = plot_plotly(model, forecast)
st.plotly_chart(fig_1)

st.write('Forecast Components')
fig_2 = model.plot_components(forecast)
st.write(fig_2)
