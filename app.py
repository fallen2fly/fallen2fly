"""
Stock Analyzer - app.py
FastAPI application that fetches stock data (via yfinance), computes simple technical indicators
(SMA, EMA, RSI) and combines them with fundamental/company information to produce
an overall buy/sell/hold recommendation and an explanation.

Usage:
  1. Install dependencies:
     pip install fastapi uvicorn yfinance pandas numpy requests jinja2

  2. Run locally:
     uvicorn stock_analyzer_app:app --reload

Endpoints:
  - GET /              -> simple HTML form for quick testing
  - GET /analyze?symbol=SYMBOL&period=1y&interval=1d
                       -> JSON analysis for the given ticker symbol

Notes & extensions:
  - You can plug in premium data sources (Alpha Vantage, IEX Cloud, Finnhub) to
    improve fundamentals and ownership data.
  - This is intentionally self-contained and conservative about external APIs.
"""

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import datetime

app = FastAPI(title="Stock Analyzer")
templates = Jinja2Templates(directory=".")  # simple; only used for the root form


class AnalysisResult(BaseModel):
    symbol: str
    date: str
    technical_score: float
    fundamental_score: float
    combined_score: float
    recommendation: str
    reasons: list
    technical: dict
    fundamental: dict


# ----------------------- Technical indicator helpers -----------------------

def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------- Scoring helpers ------------------------------

def score_technical(df: pd.DataFrame) -> (float, dict):
    """Return a technical score [0-100] and a details dict"""
    close = df["Close"]
    sma_short = simple_moving_average(close, 20)
    sma_long = simple_moving_average(close, 50)
    ema_short = exponential_moving_average(close, 12)
    ema_long = exponential_moving_average(close, 26)
    rsi = compute_rsi(close)

    latest = {
        "close": float(close.iloc[-1]),
        "sma20": float(sma_short.iloc[-1]) if not np.isnan(sma_short.iloc[-1]) else None,
        "sma50": float(sma_long.iloc[-1]) if not np.isnan(sma_long.iloc[-1]) else None,
        "ema12": float(ema_short.iloc[-1]) if not np.isnan(ema_short.iloc[-1]) else None,
        "ema26": float(ema_long.iloc[-1]) if not np.isnan(ema_long.iloc[-1]) else None,
        "rsi14": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
    }

    score = 50.0  # neutral baseline
    reasons = []

    # Trend: SMA cross
    if latest["sma20"] and latest["sma50"]:
        if latest["sma20"] > latest["sma50"]:
            score += 15
            reasons.append("Short-term SMA above long-term SMA (bullish trend)")
        else:
            score -= 15
            reasons.append("Short-term SMA below long-term SMA (bearish trend)")

    # Momentum: EMA cross
    if latest["ema12"] and latest["ema26"]:
        if latest["ema12"] > latest["ema26"]:
            score += 10
            reasons.append("Short EMAs above long EMAs (positive momentum)")
        else:
            score -= 10
            reasons.append("Short EMAs below long EMAs (negative momentum)")

    # RSI
    if latest["rsi14"] is not None:
        if latest["rsi14"] < 30:
            score += 8
            reasons.append("RSI indicates oversold (possible buy opportunity)")
        elif latest["rsi14"] > 70:
            score -= 8
            reasons.append("RSI indicates overbought (caution for buyers)")
        else:
            reasons.append("RSI neutral")

    # Price vs EMA50
    if latest["ema26"]:
        diff_pct = (latest["close"] - latest["ema26"]) / latest["ema26"]
        if diff_pct > 0.10:
            score -= 5
            reasons.append("Price significantly above long EMA (risk of pullback)")
        elif diff_pct < -0.10:
            score += 5
            reasons.append("Price significantly below long EMA (potential value)")

    # Clamp
    score = max(0, min(100, score))

    technical = {
        "latest": latest,
        "sma20_series_last_5": sma_short.dropna().iloc[-5:].tolist() if len(sma_short.dropna()) >= 5 else sma_short.dropna().tolist(),
        "rsi14_series_last_5": rsi.dropna().iloc[-5:].tolist() if len(rsi.dropna()) >= 5 else rsi.dropna().tolist(),
    }

    return score, {"score": score, "reasons": reasons, "technical": technical}


def score_fundamental(info: dict) -> (float, dict):
    """Return a fundamental score [0-100] and details dict. Uses fields from yfinance Ticker.info when available."""
    score = 50.0
    reasons = []
    details = {}

    # Market cap
    market_cap = info.get("marketCap")
    if market_cap:
        details["marketCap"] = market_cap
        if market_cap > 50_000_000_000:
            score += 8
            reasons.append("Large market cap (established company)")
        elif market_cap < 500_000_000:
            score -= 5
            reasons.append("Small market cap (higher risk)")

    # Profitability
    pr
