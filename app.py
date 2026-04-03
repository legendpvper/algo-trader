"""
AI Algorithmic Trading — Flask Web App
=======================================
Run:
    pip install flask yfinance pandas numpy scikit-learn plotly
    python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


def fetch_ohlc(ticker: str, period: str) -> pd.DataFrame:
    """
    Download OHLC from yfinance and return a clean flat DataFrame.
    Handles both old (flat) and new (MultiIndex) column formats robustly.
    """
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the ticker symbol.")

    # Flatten MultiIndex columns: ("Close", "AAPL") -> "Close"
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    # Deduplicate columns that may appear after flattening
    raw = raw.loc[:, ~raw.columns.duplicated()]

    needed = [c for c in ["Close", "High", "Low", "Open", "Volume"] if c in raw.columns]
    df = raw[needed].copy()
    df = df[df["Close"].notna() & (df["Close"] > 0)]   # drop bad rows
    df.index = pd.to_datetime(df.index)                 # ensure DatetimeIndex
    df.sort_index(inplace=True)
    return df


def run_model(ticker: str, window: int = 90):
    df = fetch_ohlc(ticker, period=f"{window + 60}d")
    df = df.tail(window + 40)

    if len(df) < 30:
        raise ValueError(f"Not enough data for '{ticker}' (got {len(df)} rows).")

    # Feature engineering on the FULL df first — never slice before computing rolling features
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    df = df.copy()
    df["ma5"]             = close.rolling(5).mean()
    df["ma10"]            = close.rolling(10).mean()
    df["ma20"]            = close.rolling(20).mean()
    df["ma5_ma20_spread"] = (df["ma5"] - df["ma20"]) / close
    df["momentum_5d"]     = close.pct_change(5)
    df["dist_from_ma10"]  = (close - df["ma10"]) / close
    df["volatility_10d"]  = close.pct_change().rolling(10).std()
    df["high_low_range"]  = (high - low) / close
    df["next_close"]      = close.shift(-1)
    df.dropna(inplace=True)

    if len(df) < 25:
        raise ValueError("Not enough clean rows after feature engineering.")

    FEATURES = [
        "Close", "ma5_ma20_spread", "momentum_5d",
        "dist_from_ma10", "volatility_10d", "high_low_range",
    ]

    X = df[FEATURES].values.astype(float)
    y = df["next_close"].values.astype(float)

    split = int(len(X) * 0.8)
    model = LinearRegression()
    model.fit(X[:split], y[:split])

    y_pred_test     = model.predict(X[split:])
    r2              = float(r2_score(y[split:], y_pred_test))
    mae             = float(mean_absolute_error(y[split:], y_pred_test))
    pred_all        = model.predict(X)
    predicted_close = float(model.predict(X[-1].reshape(1, -1))[0])
    last_close      = float(df["Close"].iloc[-1])
    change_pct      = (predicted_close - last_close) / last_close * 100
    today_chg_pct   = (last_close - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100

    if change_pct > 1.0:
        signal = "BUY"
        signal_reason = (
            f"The model predicts a <strong>+{change_pct:.2f}% gain</strong> tomorrow. "
            f"MA5 is {'above' if float(df['ma5_ma20_spread'].iloc[-1]) > 0 else 'below'} MA20 "
            f"and 5-day momentum is {'positive' if float(df['momentum_5d'].iloc[-1]) > 0 else 'negative'}. "
            f"Entry today is recommended."
        )
    elif change_pct < -1.0:
        signal = "SELL"
        signal_reason = (
            f"The model predicts a <strong>{change_pct:.2f}% drop</strong> tomorrow. "
            f"Downward momentum detected. Consider exiting or avoiding new positions today."
        )
    else:
        signal = "HOLD"
        signal_reason = (
            f"Predicted change of <strong>{change_pct:.2f}%</strong> falls within the neutral "
            f"&plusmn;1% band. Insufficient edge to justify a trade today."
        )

    sig_color = "#4ade80" if signal == "BUY" else "#f87171" if signal == "SELL" else "#60a5fa"

    # ── Chart data: slice AFTER all features computed; convert everything to plain Python lists
    PLOT_N  = min(50, len(df))
    plot_df = df.tail(PLOT_N)

    # ISO date strings — avoids any Timestamp/numpy serialisation issues
    dates   = [d.strftime("%Y-%m-%d") for d in plot_df.index]
    close_v = [round(float(v), 4) for v in plot_df["Close"]]
    ma5_v   = [round(float(v), 4) for v in plot_df["ma5"]]
    ma10_v  = [round(float(v), 4) for v in plot_df["ma10"]]
    ma20_v  = [round(float(v), 4) for v in plot_df["ma20"]]
    mom_v   = [round(float(v) * 100, 4) for v in plot_df["momentum_5d"]]
    vol_v   = [round(float(v) * 100, 4) for v in plot_df["volatility_10d"]]

    # Model prediction overlay — test-set portion only, intersected with plot window
    pred_series    = pd.Series(pred_all, index=df.index)
    test_idx_set   = set(df.index[split:])
    plot_pred_rows = [(d.strftime("%Y-%m-%d"), round(float(pred_series[d]), 4))
                      for d in plot_df.index if d in test_idx_set]
    pred_dates = [r[0] for r in plot_pred_rows]
    pred_vals  = [r[1] for r in plot_pred_rows]

    last_date_str = plot_df.index[-1].strftime("%Y-%m-%d")
    next_date_str = (plot_df.index[-1] + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")

    price_min = round(min(close_v + ma5_v + ma10_v + ma20_v + [predicted_close]) * 0.97, 2)
    price_max = round(max(close_v + ma5_v + ma10_v + ma20_v + [predicted_close]) * 1.03, 2)

    # ── Plotly figure
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
        subplot_titles=("Price & Moving Averages", "5-Day Momentum (%)", "10-Day Volatility (%)")
    )

    fig.add_trace(go.Scatter(x=dates, y=close_v, name="Close",
        line=dict(color="#93c5fd", width=2), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma5_v, name="MA5",
        line=dict(color="#4ade80", width=1.2, dash="dot"), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma10_v, name="MA10",
        line=dict(color="#fbbf24", width=1.2, dash="dot"), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma20_v, name="MA20",
        line=dict(color="#f87171", width=1.2, dash="dot"), mode="lines"), row=1, col=1)

    if pred_vals:
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_vals, name="Model prediction",
            line=dict(color="#c084fc", width=1.5, dash="dash"), mode="lines", opacity=0.85),
            row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[last_date_str, next_date_str], y=[last_close, predicted_close],
        name="Tomorrow (predicted)", mode="lines+markers",
        line=dict(color=sig_color, width=2, dash="dot"),
        marker=dict(size=[6, 12], color=sig_color, symbol=["circle", "star"])),
        row=1, col=1)

    fig.add_annotation(
        x=next_date_str, y=predicted_close,
        text=f"<b>${predicted_close:.2f}<br>{change_pct:+.1f}%</b>",
        showarrow=True, arrowhead=2, arrowcolor=sig_color,
        font=dict(color=sig_color, size=12), bgcolor="#1e293b",
        bordercolor=sig_color, ax=40, ay=-40)

    mom_colors = ["#4ade80" if v >= 0 else "#f87171" for v in mom_v]
    fig.add_trace(go.Bar(x=dates, y=mom_v, marker_color=mom_colors,
        name="Momentum", showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_color="#334155", line_width=1, row=2, col=1)

    fig.add_trace(go.Scatter(x=dates, y=vol_v, fill="tozeroy",
        fillcolor="rgba(251,191,36,0.15)", line=dict(color="#fbbf24", width=1.2),
        name="Volatility", showlegend=False), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", family="'IBM Plex Mono', monospace"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1,
                    font=dict(size=11), orientation="h", y=1.02, x=0),
        margin=dict(l=70, r=30, t=60, b=40),
        height=620,
    )
    fig.update_yaxes(range=[price_min, price_max], tickprefix="$",
                     gridcolor="#1e293b", zerolinecolor="#334155",
                     tickfont=dict(size=10), row=1, col=1)
    for i in (2, 3):
        fig.update_yaxes(ticksuffix="%", gridcolor="#1e293b",
                         zerolinecolor="#334155", tickfont=dict(size=10), row=i, col=1)
    fig.update_xaxes(tickformat="%b %d", tickangle=-30, row=3, col=1)

    chart_json = json.loads(fig.to_json())

    last = df.iloc[-1]
    return {
        "ticker":          ticker.upper(),
        "last_close":      round(last_close, 2),
        "predicted_close": round(predicted_close, 2),
        "change_pct":      round(change_pct, 2),
        "today_chg_pct":   round(today_chg_pct, 2),
        "signal":          signal,
        "signal_reason":   signal_reason,
        "signal_color":    sig_color,
        "r2":              round(r2, 4),
        "mae":             round(mae, 2),
        "ma5":             round(float(last["ma5"]), 2),
        "ma10":            round(float(last["ma10"]), 2),
        "ma20":            round(float(last["ma20"]), 2),
        "momentum_5d":     round(float(last["momentum_5d"]) * 100, 2),
        "volatility_10d":  round(float(last["volatility_10d"]) * 100, 2),
        "date_range":      f"{df.index[0].date()} → {df.index[-1].date()}",
        "n_sessions":      len(df),
        "features": [
            {"name": "Close price",     "value": f"${last_close:.2f}",                         "coef": round(float(model.coef_[0]), 5)},
            {"name": "MA5-MA20 spread", "value": f"{float(last['ma5_ma20_spread'])*100:.2f}%", "coef": round(float(model.coef_[1]), 5)},
            {"name": "5d momentum",     "value": f"{float(last['momentum_5d'])*100:.2f}%",     "coef": round(float(model.coef_[2]), 5)},
            {"name": "Dist from MA10",  "value": f"{float(last['dist_from_ma10'])*100:.2f}%",  "coef": round(float(model.coef_[3]), 5)},
            {"name": "10d volatility",  "value": f"{float(last['volatility_10d'])*100:.2f}%",  "coef": round(float(model.coef_[4]), 5)},
            {"name": "High-low range",  "value": f"{float(last['high_low_range'])*100:.2f}%",  "coef": round(float(model.coef_[5]), 5)},
        ],
        "chart": chart_json,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data   = request.get_json()
    ticker = data.get("ticker", "AAPL").strip().upper()
    window = int(data.get("window", 90))
    try:
        result = run_model(ticker, window)
        return jsonify({"ok": True, "data": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
