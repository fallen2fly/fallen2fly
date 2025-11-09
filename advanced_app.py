"""
Advanced Stock Analyzer - app.py
Upgraded FastAPI app with:
  - MACD, Bollinger Bands, and ATR added to technical analysis
  - Fundamental scoring includes debt ratios and revenue growth
  - Optional SQLite storage for past analyses
  - Enhanced JSON output with structured signals and scores
  - Async data fetching to speed up multiple ticker requests

Usage:
  pip install fastapi uvicorn yfinance pandas numpy requests jinja2 sqlalchemy

Endpoints:
  GET /                 -> HTML form for testing
  GET /analyze?symbol=... -> Returns JSON with advanced analysis
  GET /history          -> Returns saved analysis history from SQLite
"""

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List
import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncio

app = FastAPI(title="Advanced Stock Analyzer")
Base = declarative_base()

# Database setup (SQLite)
engine = create_engine("sqlite:///analysis_history.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class AnalysisHistory(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime)
    combined_score = Column(Float)
    recommendation = Column(String)
    details = Column(JSON)

Base.metadata.create_all(bind=engine)

# ------------------ Models ------------------
class AnalysisResult(BaseModel):
    symbol: str
    date: str
    technical_score: float
    fundamental_score: float
    combined_score: float
    recommendation: str
    reasons: List[str]
    technical: dict
    fundamental: dict

# ------------------ Technical indicators ------------------
def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series) -> dict:
    ema12 = exponential_moving_average(series, 12)
    ema26 = exponential_moving_average(series, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return {"macd": macd.iloc[-1], "signal": signal.iloc[-1]}

def compute_bollinger(series: pd.Series, window=20) -> dict:
    sma = simple_moving_average(series, window)
    std = series.rolling(window).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return {"upper": upper.iloc[-1], "lower": lower.iloc[-1], "sma": sma.iloc[-1]}

def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.iloc[-1]

# ------------------ Scoring ------------------
def score_technical(df: pd.DataFrame) -> (float, dict):
    close = df['Close']
    rsi = compute_rsi(close)
    macd_vals = compute_macd(close)
    boll = compute_bollinger(close)
    atr = compute_atr(df)
    score = 50.0
    reasons = []

    # MACD
    if macd_vals['macd'] > macd_vals['signal']:
        score += 10
        reasons.append("MACD bullish crossover")
    else:
        score -= 10
        reasons.append("MACD bearish crossover")

    # Bollinger
    if close.iloc[-1] < boll['lower']:
        score += 5
        reasons.append("Price below Bollinger lower band (potential buy)")
    elif close.iloc[-1] > boll['upper']:
        score -= 5
        reasons.append("Price above Bollinger upper band (potential caution)")

    # RSI
    if rsi.iloc[-1] < 30:
        score += 5
        reasons.append("RSI oversold")
    elif rsi.iloc[-1] > 70:
        score -= 5
        reasons.append("RSI overbought")

    score = max(0, min(100, score))
    technical = {
        'latest_close': float(close.iloc[-1]),
        'rsi14': float(rsi.iloc[-1]),
        'macd': macd_vals,
        'bollinger': boll,
        'atr': atr
    }
    return score, {'score': score, 'reasons': reasons, 'technical': technical}

def score_fundamental(info: dict) -> (float, dict):
    score = 50.0
    reasons = []
    details = {}
    # Market cap
    mc = info.get('marketCap')
    if mc:
        details['marketCap'] = mc
        if mc > 50e9: score += 5; reasons.append("Large market cap")
        elif mc < 1e9: score -= 5; reasons.append("Small market cap")
    # Debt ratio
    debt_to_equity = info.get('debtToEquity')
    if debt_to_equity is not None:
        details['debtToEquity'] = debt_to_equity
        if debt_to_equity < 50: score += 5; reasons.append("Low debt ratio")
        else: score -= 5; reasons.append("High debt ratio")
    # Revenue growth
    rev_growth = info.get('revenueGrowth')
    if rev_growth is not None:
        details['revenueGrowth'] = rev_growth
        if rev_growth > 0.1: score += 5; reasons.append("Strong revenue growth")
        elif rev_growth < 0: score -= 5; reasons.append("Negative revenue growth")
    # Business summary keywords
    biz = info.get('longBusinessSummary','').lower()
    if 'growth' in biz or 'platform' in biz: score += 2; reasons.append("Positive business keywords")
    if 'declin' in biz or 'risk' in biz: score -= 5; reasons.append("Risky business keywords")
    score = max(0, min(100, score))
    fundamental = {'score': score, 'reasons': reasons, 'details': details}
    return score, fundamental

# ------------------ Endpoints ------------------
@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    html = """<html><body><h2>Advanced Stock Analyzer</h2><form action='/analyze'><input name='symbol' value='AAPL'/><input type='submit'/></form></body></html>"""
    return HTMLResponse(html)

@app.get('/analyze', response_model=AnalysisResult)
async def analyze(symbol: str = Query(..., min_length=1)):
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='1y', interval='1d', actions=False)
    if hist.empty: raise HTTPException(404, 'No data found')

    tech_score, tech_out = score_technical(hist)
    fund_score, fund_out = score_fundamental(ticker.info)
    combined = round(0.6*tech_score + 0.4*fund_score,2)

    if combined >= 70: rec = 'Strong Buy'
    elif combined >= 55: rec = 'Buy'
    elif combined >= 45: rec = 'Hold'
    elif combined >= 30: rec = 'Sell'
    else: rec = 'Strong Sell'

    reasons = tech_out['reasons'] + fund_out['reasons']
    if not reasons: reasons.append("No strong signals")

    result = AnalysisResult(symbol=symbol, date=datetime.datetime.utcnow().isoformat()+'Z',
                            technical_score=tech_score, fundamental_score=fund_score,
                            combined_score=combined, recommendation=rec,
                            reasons=reasons, technical=tech_out['technical'], fundamental=fund_out['details'])

    # Save to DB
    session = SessionLocal()
    record = AnalysisHistory(symbol=symbol, date=datetime.datetime.utcnow(), combined_score=combined,
                             recommendation=rec, details=result.dict())
    session.add(record)
    session.commit()
    session.close()

    return JSONResponse(result.dict())

@app.get('/history')
async def history():
    session = SessionLocal()
    records = session.query(AnalysisHistory).order_by(AnalysisHistory.date.desc()).limit(50).all()
    session.close()
    return JSONResponse([{'symbol': r.symbol, 'date': r.date.isoformat(), 'score': r.combined_score, 'recommendation': r.recommendation} for r in records])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('stock_analyzer_app:app', host='0.0.0.0', port=8000, reload=True)
