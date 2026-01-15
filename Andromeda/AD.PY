# AD.py – main (VOLLEDIG LIVE – rate limit proof – geen warnings)
from Rules import is_killzone_allowed, determine_bias, shared_steps_complete
from Confluences import *
from Holidays_and_news import is_market_closed, is_early_close, is_high_risk_news_day
from massive import RESTClient
import pandas as pd
from datetime import datetime, timedelta

# Jouw Polygon API key
POLYGON_API_KEY = "z8vloB7z1PXwB_oNWoa_wig2KdtzkBEu"

# Live data laden
def load_live_data(symbol, multiplier=1, timespan="minute", from_date=None, to_date=None):
    client = RESTClient(api_key=POLYGON_API_KEY)
    aggs = client.get_aggs(symbol, multiplier, timespan, from_date, to_date, limit=50000)
    df = pd.DataFrame(aggs)
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

# Resample functie – zonder FutureWarning
def resample_aggs(df, minutes):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df.set_index('timestamp')
    rule = f'{minutes}min'   # ← 'min' in plaats van 'T' → geen warning meer
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled = df.resample(rule).apply(ohlc).dropna()
    return resampled.reset_index()

print("AD – Andromeda Trader [LIVE MODE]")

# === DYNAMISCHE LIVE DATES (vandaag, met fallback naar gisteren) ===
today = datetime.now().strftime("%Y-%m-%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

from_date = today
to_date = today

# Laad data van vandaag
df_es_1min = load_live_data("ES", 1, "minute", from_date, to_date)
df_nq_1min = load_live_data("NQ", 1, "minute", from_date, to_date)

# Als er nog geen bars zijn vandaag (markt nog niet open of gesloten) → gebruik gisteren
if df_es_1min.empty or df_nq_1min.empty:
    print("Geen bars vandaag → fallback naar gisteren")
    from_date = yesterday
    to_date = yesterday
    df_es_1min = load_live_data("ES", 1, "minute", from_date, to_date)
    df_nq_1min = load_live_data("NQ", 1, "minute", from_date, to_date)

# Nu zeker data hebben
df_es = df_es_1min
df_nq = df_nq_1min

# Resample hogere timeframes lokaal
df_es_2min = resample_aggs(df_es_1min, 2)
df_es_3min = resample_aggs(df_es_1min, 3)
df_es_4min = resample_aggs(df_es_1min, 4)
df_es_5min = resample_aggs(df_es_1min, 5)

df_nq_2min = resample_aggs(df_nq_1min, 2)
df_nq_3min = resample_aggs(df_nq_1min, 3)
df_nq_4min = resample_aggs(df_nq_1min, 4)
df_nq_5min = resample_aggs(df_nq_1min, 5)

# News altijd tonen
news, event = is_high_risk_news_day()
if news:
    print(f"⚠️ HOOG RISICO – {event} vandaag – reduceer risk 50%")

if is_market_closed():
    print("🛑 MARKT GESLOTEN – geen trades vandaag")
    exit()

if is_early_close():
    print(" Early close vandaag – reduceer risk 50%")

zone, mult = is_killzone_allowed(df_es['timestamp'].iloc[-1])
if not zone:
    print(f"Buiten killzone ({zone}) → geen trade")
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
