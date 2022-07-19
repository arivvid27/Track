import subprocess
import sys

try:
    import streamlit as st
    from datetime import date

    import yfinance as yf
    from neuralprophet import NeuralProphet

    from plotly import graph_objs as go
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'streamlit'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'yfinance'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'neuralprophet'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'prophet'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'plotly'])
finally:
    import streamlit as st
    from datetime import date

    import yfinance as yf
    from neuralprophet import NeuralProphet
    from prophet import Prophet

    from plotly import graph_objs as go



START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Prediction")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "TSLA", "RIVN", "BA")
selected_stocks = st.selectbox("Select Dataset for Prediction", stocks)

n_years = st.slider("Years of Prediction:", 1, 10)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    global fig
    fig = go.Figure()
    fig.add_trace(go.Line(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Line(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = NeuralProphet()
metrics = m.fit(df_train)
future = m.make_future_dataframe(df=df_train, periods=period, n_historic_predictions=True)
forecast = m.predict(df=future)

n = Prophet()
mnetrics = n.fit(df.train)
nfuture = n.make_future_dataframe(periods=period)
nforecast = n.predict(nfuture)
nforecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


st.subheader('Forecast data (NeuralProphet')
st.write(forecast.tail())
st.write("Forecast Components (Prophet)")
st.write(nforecast.tail())

st.write('Forecast Data (NeuralProphet)')
fig1 = m.plot(forecast)
st.plotly_chart(fig1, use_container_width=True)
st.write("Forecast Components (Prophet)")
nfig1 = n.plot(nforecast)
st.plotly_chart(nfig1, use_container_widtch=True)

st.write("Forecast Components (NeuralProphet)")
fig2 = m.plot_components(forecast)
st.write(fig2)
st.write("Forecast Components (Prophet)")
nfig2 = n.plot_components(nforecast)
st.write(nfig2)
