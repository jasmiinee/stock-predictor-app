import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")

stocks = ("AAPL", "GOOG", "MSFT", "NVDA", "TSLA", "META")
selected_stocks = st.selectbox("Select data", stocks)
n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

# load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data done")

st.subheader('Raw data')
st.write(data.tail())

# plot data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date', ''], y=data['Open', selected_stocks], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date', ''], y=data['Close', selected_stocks], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# forecasting
df_train = data[[('Date', ''), ('Close', selected_stocks)]]
df_train.columns = ["ds", "y"]

m = Prophet()
m.fit(df_train)
predicted = m.make_future_dataframe(periods=period)
forecast = m.predict(predicted)

st.subheader('Forecasted data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast component')
fig2 = m.plot_components(forecast)
st.write(fig2)