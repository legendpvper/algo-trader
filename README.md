# AI Algorithmic Trading Dashboard

A web-based AI trading dashboard that predicts next-day stock prices using linear regression trained on technical indicators, and generates **Buy / Hold / Sell** signals based on the prediction.

Built as a school project to demonstrate algorithmic trading concepts using real market data.

---

## Features

- Real-time stock data fetched via `yfinance`
- Next-day close price prediction using Linear Regression
- Trading signal generation (Buy / Hold / Sell) based on predicted % change
- Interactive candlestick chart with moving averages (MA5, MA10, MA20)
- 5-day momentum and 10-day volatility subplots
- Model performance metrics (R² score, MAE)
- Feature weights breakdown

---

## Tech Stack

| Layer    | Technology                          |
|----------|-------------------------------------|
| Backend  | Python, Flask                       |
| ML Model | scikit-learn (Linear Regression)    |
| Data     | yfinance                            |
| Charts   | Plotly                              |
| Hosting  | Render                              |

---

## How It Works

1. Historical OHLC price data is fetched for the selected ticker
2. Six technical features are engineered from the raw data:
   - Close price
   - MA5 − MA20 crossover spread
   - 5-day momentum
   - Distance from MA10
   - 10-day rolling volatility
   - Daily high−low range
3. A Linear Regression model is trained on an 80/20 train/test split
4. The model predicts tomorrow's closing price
5. A signal is generated based on the predicted % change:
   - **BUY** — predicted gain > +1%
   - **SELL** — predicted drop > −1%
   - **HOLD** — within the ±1% neutral band

---

## Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the app
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Disclaimer

This project is for **educational purposes only** and does not constitute financial advice. Do not use this tool to make real investment decisions.
