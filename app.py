"""
AI Algorithmic Trading — Flask Web App
=======================================
Run:
    pip install flask yfinance pandas numpy scikit-learn plotly xgboost
    python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


def fetch_ohlc(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLC and return a clean flat DataFrame."""
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the ticker symbol.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()]
    needed = [c for c in ["Close", "High", "Low", "Open", "Volume"] if c in raw.columns]
    df = raw[needed].copy()
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer technical indicator features on the full DataFrame.
    Always call this before slicing — rolling windows need the full history.
    """
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # Moving averages
    df["ma5"]  = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()

    # MA-derived signals
    df["ma5_ma20_spread"] = (df["ma5"] - df["ma20"]) / close
    df["ma10_ma50_spread"] = (df["ma10"] - df["ma50"]) / close
    df["dist_from_ma10"]  = (close - df["ma10"]) / close
    df["dist_from_ma20"]  = (close - df["ma20"]) / close

    # Momentum
    df["momentum_1d"] = close.pct_change(1)
    df["momentum_5d"] = close.pct_change(5)
    df["momentum_10d"] = close.pct_change(10)

    # Volatility
    df["volatility_5d"]  = close.pct_change().rolling(5).std()
    df["volatility_10d"] = close.pct_change().rolling(10).std()
    df["volatility_20d"] = close.pct_change().rolling(20).std()

    # RSI (14-period)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12/26 EMA, signal 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20-period)
    bb_mid             = close.rolling(20).mean()
    bb_std             = close.rolling(20).std()
    df["bb_upper"]     = bb_mid + 2 * bb_std
    df["bb_lower"]     = bb_mid - 2 * bb_std
    df["bb_position"]  = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / bb_mid

    # Volume features
    df["volume_ma10"]    = volume.rolling(10).mean()
    df["volume_ratio"]   = volume / df["volume_ma10"].replace(0, np.nan)

    # Intraday range
    df["high_low_range"] = (high - low) / close

    # Target: next day's close
    df["next_close"] = close.shift(-1)

    df.dropna(inplace=True)
    return df


FEATURES = [
    "Close",
    "ma5_ma20_spread", "ma10_ma50_spread",
    "dist_from_ma10", "dist_from_ma20",
    "momentum_1d", "momentum_5d", "momentum_10d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width",
    "volume_ratio",
    "high_low_range",
]

FEATURE_LABELS = {
    "Close":           "Close price",
    "ma5_ma20_spread": "MA5−MA20 spread",
    "ma10_ma50_spread":"MA10−MA50 spread",
    "dist_from_ma10":  "Dist from MA10",
    "dist_from_ma20":  "Dist from MA20",
    "momentum_1d":     "1d momentum",
    "momentum_5d":     "5d momentum",
    "momentum_10d":    "10d momentum",
    "volatility_5d":   "5d volatility",
    "volatility_10d":  "10d volatility",
    "volatility_20d":  "20d volatility",
    "rsi":             "RSI (14)",
    "macd":            "MACD",
    "macd_signal":     "MACD signal",
    "macd_hist":       "MACD histogram",
    "bb_position":     "Bollinger position",
    "bb_width":        "Bollinger width",
    "volume_ratio":    "Volume ratio",
    "high_low_range":  "High−low range",
}



def backtest(df: pd.DataFrame, model, X: np.ndarray, split: int, signal_band: float = 1.0) -> dict:
    """
    Walk through the TEST SET day by day, generate a signal for each day
    using the model's predicted next-day close, then check whether
    the actual next-day price movement confirmed the signal.

    Returns a dict with hit rates, P&L curve, and per-signal breakdown.
    """
    test_df = df.iloc[split:].copy()
    X_test  = X[split:]

    pred_closes = model.predict(X_test)
    results = []

    for i in range(len(test_df) - 1):   # -1 because we need actual next-day close
        today_close  = float(test_df["Close"].iloc[i])
        pred_close   = float(pred_closes[i])
        actual_close = float(test_df["Close"].iloc[i + 1])   # what actually happened

        pred_chg_pct   = (pred_close   - today_close) / today_close * 100
        actual_chg_pct = (actual_close - today_close) / today_close * 100

        if pred_chg_pct > signal_band:
            signal = "BUY"
            correct = actual_chg_pct > 0      # BUY correct if price actually went up
        elif pred_chg_pct < -signal_band:
            signal = "SELL"
            correct = actual_chg_pct < 0     # SELL correct if price actually went down
        else:
            signal = "HOLD"
            correct = abs(actual_chg_pct) <= signal_band  # HOLD correct if stayed flat

        results.append({
            "date":           test_df.index[i].strftime("%Y-%m-%d"),
            "signal":         signal,
            "pred_chg_pct":   round(pred_chg_pct, 3),
            "actual_chg_pct": round(actual_chg_pct, 3),
            "correct":        correct,
        })

    if not results:
        return {}

    # ── Per-signal accuracy
    from collections import defaultdict
    counts   = defaultdict(int)
    correct  = defaultdict(int)
    for r in results:
        counts[r["signal"]] += 1
        if r["correct"]:
            correct[r["signal"]] += 1

    def hit_rate(sig):
        return round(correct[sig] / counts[sig] * 100, 1) if counts[sig] else None

    overall_correct = sum(1 for r in results if r["correct"])
    overall_rate    = round(overall_correct / len(results) * 100, 1)

    # ── Cumulative P&L curve (simple strategy: follow every BUY/SELL signal)
    # Start with $10,000. BUY = go long next day, SELL = go short, HOLD = stay flat.
    capital    = 10_000.0
    equity     = [capital]
    eq_dates   = [results[0]["date"]]
    position   = 0   # 0=flat, 1=long, -1=short

    for r in results:
        daily_return = r["actual_chg_pct"] / 100
        if r["signal"] == "BUY":
            position = 1
        elif r["signal"] == "SELL":
            position = -1
        else:
            position = 0
        capital *= (1 + position * daily_return)
        equity.append(round(capital, 2))
        eq_dates.append(r["date"])

    total_return = round((equity[-1] - 10_000) / 10_000 * 100, 2)

    # ── Build equity curve Plotly trace (returned as JSON)
    eq_fig = go.Figure()
    eq_color = "#4ade80" if total_return >= 0 else "#f87171"
    eq_fig.add_trace(go.Scatter(
        x=eq_dates, y=equity,
        mode="lines",
        line=dict(color=eq_color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({'74,222,128' if total_return >= 0 else '248,113,113'},0.08)",
        name="Portfolio value",
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))
    eq_fig.add_hline(y=10_000, line_color="#334155", line_width=1, line_dash="dot")
    eq_fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", family="'IBM Plex Mono', monospace"),
        margin=dict(l=60, r=20, t=20, b=40),
        height=220,
        showlegend=False,
        yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#1e293b",
                   zerolinecolor="#334155", tickfont=dict(size=10)),
        xaxis=dict(tickformat="%b %d", tickangle=-30, gridcolor="#1e293b",
                   tickfont=dict(size=10)),
    )

    # ── Signal log (last 20 for display)
    signal_log = results[-20:]

    return {
        "overall_rate":   overall_rate,
        "overall_correct":overall_correct,
        "total_signals":  len(results),
        "buy_rate":       hit_rate("BUY"),
        "sell_rate":      hit_rate("SELL"),
        "hold_rate":      hit_rate("HOLD"),
        "buy_count":      counts["BUY"],
        "sell_count":     counts["SELL"],
        "hold_count":     counts["HOLD"],
        "total_return":   total_return,
        "final_value":    round(equity[-1], 2),
        "equity_chart":   json.loads(eq_fig.to_json()),
        "signal_log":     signal_log,
    }

def run_model(ticker: str, window: int = 90):
    # Fetch extra history so rolling windows (MA50, RSI14) warm up properly
    df = fetch_ohlc(ticker, period=f"{window + 200}d")

    if len(df) < 80:
        raise ValueError(f"Not enough data for '{ticker}' (got {len(df)} rows).")

    df = add_features(df)
    df = df.tail(window + 100)   # trim to working window after features computed

    if len(df) < 25:
        raise ValueError("Not enough clean rows after feature engineering.")

    X = df[FEATURES].values.astype(float)
    y = df["next_close"].values.astype(float)

    split = int(len(X) * 0.8)

    # XGBoost — handles non-linear relationships and feature interactions
    # that linear regression cannot capture
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X[:split], y[:split],
        eval_set=[(X[split:], y[split:])],
        verbose=False,
    )

    y_pred_test     = model.predict(X[split:])
    r2              = float(r2_score(y[split:], y_pred_test))
    mae             = float(mean_absolute_error(y[split:], y_pred_test))
    pred_all        = model.predict(X)
    bt              = backtest(df, model, X, split)
    predicted_close = float(model.predict(X[-1].reshape(1, -1))[0])
    last_close      = float(df["Close"].iloc[-1])
    change_pct      = (predicted_close - last_close) / last_close * 100
    today_chg_pct   = (last_close - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100

    if change_pct > 1.0:
        signal = "BUY"
        signal_reason = (
            f"The model predicts a <strong>+{change_pct:.2f}% gain</strong> tomorrow. "
            f"RSI is {float(df['rsi'].iloc[-1]):.1f} and MACD histogram is "
            f"{'positive' if float(df['macd_hist'].iloc[-1]) > 0 else 'negative'}. "
            f"Entry today is recommended."
        )
    elif change_pct < -1.0:
        signal = "SELL"
        signal_reason = (
            f"The model predicts a <strong>{change_pct:.2f}% drop</strong> tomorrow. "
            f"RSI is {float(df['rsi'].iloc[-1]):.1f} and downward momentum detected. "
            f"Consider exiting or avoiding new positions today."
        )
    else:
        signal = "HOLD"
        signal_reason = (
            f"Predicted change of <strong>{change_pct:.2f}%</strong> falls within the neutral "
            f"&plusmn;1% band. RSI is {float(df['rsi'].iloc[-1]):.1f}. "
            f"Insufficient edge to justify a trade today."
        )

    sig_color = "#4ade80" if signal == "BUY" else "#f87171" if signal == "SELL" else "#60a5fa"

    # ── Chart data: plain Python lists, ISO date strings
    PLOT_N  = min(60, len(df))
    plot_df = df.tail(PLOT_N)

    dates   = [d.strftime("%Y-%m-%d") for d in plot_df.index]
    open_v  = [round(float(v), 4) for v in plot_df["Open"]]
    high_v  = [round(float(v), 4) for v in plot_df["High"]]
    low_v   = [round(float(v), 4) for v in plot_df["Low"]]
    close_v = [round(float(v), 4) for v in plot_df["Close"]]
    ma5_v   = [round(float(v), 4) for v in plot_df["ma5"]]
    ma10_v  = [round(float(v), 4) for v in plot_df["ma10"]]
    ma20_v  = [round(float(v), 4) for v in plot_df["ma20"]]
    mom_v   = [round(float(v) * 100, 4) for v in plot_df["momentum_5d"]]
    rsi_v   = [round(float(v), 2) for v in plot_df["rsi"]]

    pred_series    = pd.Series(pred_all, index=df.index)
    test_idx_set   = set(df.index[split:])
    plot_pred_rows = [(d.strftime("%Y-%m-%d"), round(float(pred_series[d]), 4))
                      for d in plot_df.index if d in test_idx_set]
    pred_dates = [r[0] for r in plot_pred_rows]
    pred_vals  = [r[1] for r in plot_pred_rows]

    last_date_str = plot_df.index[-1].strftime("%Y-%m-%d")
    next_date_str = (plot_df.index[-1] + pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")

    price_min = round(min(low_v  + ma5_v + ma10_v + ma20_v + [predicted_close]) * 0.97, 2)
    price_max = round(max(high_v + ma5_v + ma10_v + ma20_v + [predicted_close]) * 1.03, 2)

    # ── Plotly figure — 3 subplots: price, momentum, RSI
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
        subplot_titles=("Price & Moving Averages", "5-Day Momentum (%)", "RSI (14)")
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=open_v, high=high_v, low=low_v, close=close_v,
        name="OHLC",
        increasing=dict(line=dict(color="#4ade80", width=1), fillcolor="rgba(74,222,128,0.25)"),
        decreasing=dict(line=dict(color="#f87171", width=1), fillcolor="rgba(248,113,113,0.25)"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma5_v, name="MA5",
        line=dict(color="#4ade80", width=1.2, dash="dot"), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma10_v, name="MA10",
        line=dict(color="#fbbf24", width=1.2, dash="dot"), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma20_v, name="MA20",
        line=dict(color="#f87171", width=1.2, dash="dot"), mode="lines"), row=1, col=1)

    if pred_vals:
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_vals, name="XGB prediction",
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

    # RSI with overbought/oversold bands
    fig.add_trace(go.Scatter(x=dates, y=rsi_v,
        line=dict(color="#a78bfa", width=1.5),
        name="RSI", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_color="#f87171", line_width=1, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_color="#4ade80", line_width=1, line_dash="dot", row=3, col=1)

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
    fig.update_yaxes(ticksuffix="%", gridcolor="#1e293b",
                     zerolinecolor="#334155", tickfont=dict(size=10), row=2, col=1)
    fig.update_yaxes(range=[0, 100], gridcolor="#1e293b",
                     zerolinecolor="#334155", tickfont=dict(size=10), row=3, col=1)
    fig.update_xaxes(tickformat="%b %d", tickangle=-30, row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    chart_json = json.loads(fig.to_json())

    # Feature importance from XGBoost (score = gain)
    importance = model.get_booster().get_score(importance_type="gain")
    total_gain  = sum(importance.values()) or 1
    feat_rows = []
    last = df.iloc[-1]
    for i, feat in enumerate(FEATURES):
        raw_val = float(last[feat])
        # Format value sensibly
        if feat == "Close":
            val_str = f"${raw_val:.2f}"
        elif feat == "rsi":
            val_str = f"{raw_val:.1f}"
        elif feat in ("macd", "macd_signal", "macd_hist"):
            val_str = f"{raw_val:.3f}"
        else:
            val_str = f"{raw_val * 100:.2f}%"
        imp_key = f"f{i}"
        imp_pct = round(importance.get(imp_key, 0) / total_gain * 100, 1)
        feat_rows.append({
            "name":       FEATURE_LABELS.get(feat, feat),
            "value":      val_str,
            "importance": imp_pct,
        })
    # Sort by importance descending
    feat_rows.sort(key=lambda r: r["importance"], reverse=True)

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
        "rsi":             round(float(last["rsi"]), 1),
        "macd_hist":       round(float(last["macd_hist"]), 4),
        "momentum_5d":     round(float(last["momentum_5d"]) * 100, 2),
        "volatility_10d":  round(float(last["volatility_10d"]) * 100, 2),
        "date_range":      f"{df.index[0].date()} → {df.index[-1].date()}",
        "n_sessions":      len(df),
        "model":           "XGBoost",
        "features":        feat_rows,
        "chart":           chart_json,
        "backtest":        bt,
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
