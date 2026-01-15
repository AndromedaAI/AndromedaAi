# AD.py – main (LIVE ES/NQ via yfinance – 100% GRATIS, geen API key)
from Rules import is_killzone_allowed, determine_bias, shared_steps_complete
from Confluences import *
from Holidays_and_news import is_market_closed, is_early_close, is_high_risk_news_day
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import json

print("AD – Andromeda Trader [LIVE MODE – yfinance ES/NQ]")

# Live data laden via yfinance (ES=F = ES futures, NQ=F = NQ futures)
def load_live_data(symbol, interval="1m", period="5d"):  # period="5d" voor genoeg minute data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df = df.rename(columns={
        'Datetime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

# Resample functie voor hogere timeframes
def resample_aggs(df, minutes):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df.set_index('timestamp')
    rule = f'{minutes}min'
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled = df.resample(rule).apply(ohlc).dropna()
    return resampled.reset_index()

# Dynamisch: gebruik recente data (valt terug op vorige dagen als markt dicht)
df_es_1min = load_live_data("ES=F", interval="1m")
df_nq_1min = load_live_data("NQ=F", interval="1m")

if df_es_1min.empty or df_nq_1min.empty:
    print("Geen recente bars → probeer 2m interval")
    df_es_1min = load_live_data("ES=F", interval="2m")
    df_nq_1min = load_live_data("NQ=F", interval="2m")

# Voor compatibiliteit
df_es = df_es_1min
df_nq = df_nq_1min

# Resample hogere timeframes
df_es_2min = resample_aggs(df_es_1min, 2)
df_es_3min = resample_aggs(df_es_1min, 3)
df_es_4min = resample_aggs(df_es_1min, 4)
df_es_5min = resample_aggs(df_es_1min, 5)

df_nq_2min = resample_aggs(df_nq_1min, 2)
df_nq_3min = resample_aggs(df_nq_1min, 3)
df_nq_4min = resample_aggs(df_nq_1min, 4)
df_nq_5min = resample_aggs(df_nq_1min, 5)

# News & markt checks
news, event = is_high_risk_news_day()
if news:
    print(f"⚠️ HOOG RISICO – {event} vandaag – reduceer risk 50%")

if is_market_closed():
    print("🛑 MARKT GESLOTEN – geen trades vandaag")
    exit()

if is_early_close():
    print("Early close vandaag – reduceer risk 50%")

if df_es.empty or df_nq.empty:
    print("Geen data beschikbaar → geen trade")
    exit()

zone, mult = is_killzone_allowed(df_es['timestamp'].iloc[-1])
if not zone:
    print("Buiten killzone → geen trade")
    exit()

bias, reason = determine_bias(df_es, df_nq)
if not bias:
    print("Geen bias → geen trade")
    exit()

print(f"Bias: {bias.upper()} – {reason}")

# Timeframes dict
tf_es = {1: df_es, 2: df_es_2min, 3: df_es_3min, 4: df_es_4min, 5: df_es_5min}
tf_nq = {1: df_nq, 2: df_nq_2min, 3: df_nq_3min, 4: df_nq_4min, 5: df_nq_5min}

if shared_steps_complete(df_es, df_nq, tf_es, tf_nq, df_es, df_nq):
    print(f"\n{'='*60}")
    print(f"🚀 TRADE: {bias.upper()} – ALLE STAPPEN VOLTOOID!")
    print(f"{'='*60}")
else:
    print("Stappen niet compleet → geen trade")

import time

RUN = True

print("AD – Andromeda Trader [LIVE MODE] gestart")

while RUN:
    try:
        print("\n--- Nieuwe cyclus ---", datetime.now())

        # ✅ JE BESTAANDE CODE HIER
        df_es_1min = load_live_data("ES=F", interval="1m")
        df_nq_1min = load_live_data("NQ=F", interval="1m")

        if df_es_1min.empty or df_nq_1min.empty:
            print("Geen data → retry")
            time.sleep(30)
            continue

        # ... rest van je logica ...
        # determine_bias, killzone, etc.

    except KeyboardInterrupt:
        print("\n🛑 Trader handmatig gestopt (CTRL+C)")
        RUN = False

    except Exception as e:
        print(f"⚠️ FOUT: {e}")
        time.sleep(30)

    time.sleep(60)  # ⏱️ elke 60 seconden nieuwe check
