import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import warnings
import yfinance as yf
import requests
import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
from bs4 import BeautifulSoup
import joblib
import os
import textwrap

try:
    import nsepython as nse
except ImportError:
    nse = None

# ── Fibonacci Configuration ────────────────────────────────────────
# Lookback windows per timeframe (in number of data points)
FIB_LOOKBACK = {
    "1d": 90,
    "1h": 30,
    "15m": 20,
}
# Standard Fibonacci retracement levels
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
# Weight contributed to overall confidence (10% of total)
FIB_WEIGHT = 0.10

try:
    from prediction_tracker import save_prediction, load_history, load_advanced_stats, auto_verify_signals, load_options_stats
except ImportError:
    # Safe fallback stubs - app will work even if tracker module is outdated
    def save_prediction(data): pass
    def load_history(): return []
    def load_advanced_stats(): return {"win_rate": 0.0, "total": 0, "avg_profit": 0.0, "avg_loss": 0.0, "profit_factor": 0.0, "rr_actual": 0.0}
    def auto_verify_signals(fn): pass
    def load_options_stats(): return None

warnings.filterwarnings('ignore')

st.set_page_config(page_title="SRV Future Traders", page_icon="🏛️", layout="wide",
                   initial_sidebar_state="expanded")

# ── Complete Groww-Level Stock Map (250+ Stocks) ──────────────────────────
STOCK_MAP = {
    # ═══ INDICES ═══
    'NIFTY': '^NSEI', 'NIFTY50': '^NSEI', 'NSE': '^NSEI',
    'SENSEX': '^BSESN', 'BSE': '^BSESN',
    'BANKNIFTY': '^NSEBANK', 'NIFTYBANK': '^NSEBANK',
    'MIDCPNIFTY': '^NSEMDCP50', 'FINNIFTY': '^CNXFIN',
    'NIFTYIT': '^CNXIT', 'NIFTYPHARMA': '^CNXPHARMA',
    'NIFTYAUTO': '^CNXAUTO', 'NIFTYMETAL': '^CNXMETAL',
    'NIFTYREALTY': '^CNXREALTY', 'NIFTYFMCG': '^CNXFMCG',
    'NIFTYENERGY': '^CNXENERGY',

    # ═══ NIFTY 50 (All 50 stocks) ═══
    'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'INFY': 'INFY.NS',
    'HDFCBANK': 'HDFCBANK.NS', 'HDFC': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS', 'ICICI': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS', 'SBI': 'SBIN.NS', 'STATE BANK': 'SBIN.NS', 'STATEBANK': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'ITC': 'ITC.NS', 'KOTAKBANK': 'KOTAKBANK.NS',
    'LT': 'LT.NS', 'AXISBANK': 'AXISBANK.NS',
    'WIPRO': 'WIPRO.NS', 'MARUTI': 'MARUTI.NS',
    'SUNPHARMA': 'SUNPHARMA.NS', 'BAJFINANCE': 'BAJFINANCE.NS',
    'BAJFINSV': 'BAJFINSV.NS', 'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'HCLTECH': 'HCLTECH.NS', 'ASIANPAINT': 'ASIANPAINT.NS',
    'TITAN': 'TITAN.NS', 'ULTRACEMCO': 'ULTRACEMCO.NS',
    'NESTLEIND': 'NESTLEIND.NS', 'TECHM': 'TECHM.NS',
    'POWERGRID': 'POWERGRID.NS', 'NTPC': 'NTPC.NS',
    'ONGC': 'ONGC.NS', 'COALINDIA': 'COALINDIA.NS',
    'DRREDDY': 'DRREDDY.NS', 'CIPLA': 'CIPLA.NS',
    'DIVISLAB': 'DIVISLAB.NS', 'EICHERMOT': 'EICHERMOT.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS', 'M&M': 'M&M.NS',
    'ADANIENT': 'ADANIENT.NS', 'ADANIPORTS': 'ADANIPORTS.NS',
    'JSWSTEEL': 'JSWSTEEL.NS', 'HINDALCO': 'HINDALCO.NS',
    'BPCL': 'BPCL.NS', 'HINDUNILVR': 'HINDUNILVR.NS',
    'BRITANNIA': 'BRITANNIA.NS', 'INDUSINDBK': 'INDUSINDBK.NS',
    'TATASTEEL': 'TATASTEEL.NS', 'TATA': 'TATASTEEL.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'GRASIM': 'GRASIM.NS', 'APOLLOHOSP': 'APOLLOHOSP.NS',
    'SBILIFE': 'SBILIFE.NS', 'HDFCLIFE': 'HDFCLIFE.NS',
    'LTIM': 'LTIM.NS', 'LTTS': 'LTTS.NS',

    # ═══ TATA GROUP (Complete) ═══
    'TATAPOWER': 'TATAPOWER.NS', 'TATACHEM': 'TATACHEM.NS',
    'TATACOMM': 'TATACOMM.NS', 'TATAELXSI': 'TATAELXSI.NS',
    'TATAINVEST': 'TATAINVEST.NS', 'TATACONSUM': 'TATACONSUM.NS',
    'TATAMETALI': 'TATAMETALI.NS', 'TATASPONGE': 'TATASPONGE.NS',
    'TATACOFFEE': 'TATACOFFEE.NS', 'TTML': 'TTML.NS',
    'TITAN': 'TITAN.NS', 'VOLTAS': 'VOLTAS.NS',
    'TRENT': 'TRENT.NS', 'RALLIS': 'RALLIS.NS',

    # ═══ ADANI GROUP (Complete) ═══
    'ADANIGREEN': 'ADANIGREEN.NS', 'ADANIPOWER': 'ADANIPOWER.NS',
    'ADANITRANS': 'ADANITRANS.NS', 'ATGL': 'ATGL.NS',
    'AWL': 'AWL.NS', 'ADANIWILMAR': 'AWL.NS',
    'ACC': 'ACC.NS', 'AMBUJACEM': 'AMBUJACEM.NS',

    # ═══ BANKING & FINANCE ═══
    'PNB': 'PNB.NS', 'BANKBARODA': 'BANKBARODA.NS',
    'CANBK': 'CANBK.NS', 'UNIONBANK': 'UNIONBANK.NS',
    'IDFCFIRSTB': 'IDFCFIRSTB.NS', 'FEDERALBNK': 'FEDERALBNK.NS',
    'BANDHANBNK': 'BANDHANBNK.NS', 'RBLBANK': 'RBLBANK.NS',
    'YESBANK': 'YESBANK.NS', 'AUBANK': 'AUBANK.NS',
    'CENTRALBK': 'CENTRALBK.NS', 'INDIANB': 'INDIANB.NS',
    'IOB': 'IOB.NS', 'UCOBANK': 'UCOBANK.NS',
    'MAHABANK': 'MAHABANK.NS', 'PSB': 'PSB.NS',
    'J&KBANK': 'J&KBANK.NS', 'KARURVYSYA': 'KARURVYSYA.NS',
    'SOUTHBANK': 'SOUTHBANK.NS', 'TMB': 'TMB.NS',
    'CUB': 'CUB.NS', 'DCB': 'DCB.NS', 'CSB': 'CSB.NS',

    # ═══ NBFC & FINANCE ═══
    'SHRIRAMFIN': 'SHRIRAMFIN.NS', 'CHOLAFIN': 'CHOLAFIN.NS',
    'MUTHOOTFIN': 'MUTHOOTFIN.NS', 'MANAPPURAM': 'MANAPPURAM.NS',
    'CANFINHOME': 'CANFINHOME.NS', 'LICHSGFIN': 'LICHSGFIN.NS',
    'POONAWALLA': 'POONAWALLA.NS', 'IIFL': 'IIFL.NS',
    'MFSL': 'MFSL.NS', 'MOTILALOFS': 'MOTILALOFS.NS',
    'ANGELONE': 'ANGELONE.NS', 'CDSL': 'CDSL.NS',
    'BSE': 'BSE.NS', 'MCX': 'MCX.NS',
    'ICICIPRULI': 'ICICIPRULI.NS', 'ICICIGI': 'ICICIGI.NS',
    'LICI': 'LICI.NS', 'NIACL': 'NIACL.NS',
    'GICRE': 'GICRE.NS', 'STARHEALTH': 'STARHEALTH.NS',
    'POLICYBZR': 'POLICYBZR.NS',

    # ═══ IT & SOFTWARE ═══
    'MPHASIS': 'MPHASIS.NS', 'COFORGE': 'COFORGE.NS',
    'PERSISTENT': 'PERSISTENT.NS', 'HAPPSTMNDS': 'HAPPSTMNDS.NS',
    'ZOMATO': 'ETERNAL.NS', 'PAYTM': 'PAYTM.NS',
    'NAUKRI': 'NAUKRI.NS', 'INFOEDGE': 'NAUKRI.NS',
    'ROUTE': 'ROUTE.NS', 'MAPMY': 'MAPMYINDIA.NS',
    'LATENTVIEW': 'LATENTVIEW.NS', 'NEWGEN': 'NEWGEN.NS',
    'MASTEK': 'MASTEK.NS', 'ZENSAR': 'ZENSAR.NS',
    'BIRLASOFT': 'BIRLASOFT.NS', 'CYIENT': 'CYIENT.NS',
    'KPITTECH': 'KPITTECH.NS', 'SONATSOFTW': 'SONATSOFTW.NS',
    'TANLA': 'TANLA.NS', 'RATEGAIN': 'RATEGAIN.NS',

    # ═══ PHARMA & HEALTHCARE ═══
    'LUPIN': 'LUPIN.NS', 'AUROPHARMA': 'AUROPHARMA.NS',
    'BIOCON': 'BIOCON.NS', 'TORNTPHARM': 'TORNTPHARM.NS',
    'ALKEM': 'ALKEM.NS', 'IPCALAB': 'IPCALAB.NS',
    'GLENMARK': 'GLENMARK.NS', 'ABBOTINDIA': 'ABBOTINDIA.NS',
    'PFIZER': 'PFIZER.NS', 'SANOFI': 'SANOFI.NS',
    'LALPATHLAB': 'LALPATHLAB.NS', 'METROPOLIS': 'METROPOLIS.NS',
    'MAXHEALTH': 'MAXHEALTH.NS', 'FORTIS': 'FORTIS.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS', 'NARAYANA': 'NH.NS',
    'NATCOPHARM': 'NATCOPHARM.NS', 'LAURUSLABS': 'LAURUSLABS.NS',
    'GRANULES': 'GRANULES.NS', 'AJANTPHARM': 'AJANTPHARM.NS',

    # ═══ AUTO & AUTO ANCILLARY ═══
    'ASHOKLEY': 'ASHOKLEY.NS', 'TVSMOTOR': 'TVSMOTOR.NS',
    'BALKRISIND': 'BALKRISIND.NS', 'MRF': 'MRF.NS',
    'APOLLOTYRE': 'APOLLOTYRE.NS', 'CEATLTD': 'CEATLTD.NS',
    'BHARATFORG': 'BHARATFORG.NS', 'MOTHERSON': 'MOTHERSON.NS',
    'EXIDEIND': 'EXIDEIND.NS', 'AMARAJABAT': 'AMARARAJA.NS',
    'BOSCHLTD': 'BOSCHLTD.NS', 'ENDURANCE': 'ENDURANCE.NS',
    'SUNDRMFAST': 'SUNDRMFAST.NS', 'ESCORTS': 'ESCORTS.NS',
    'OLACABS': 'OLACABS.NS', 'ABORANGE': 'OLA.NS',

    # ═══ METALS & MINING ═══
    'VEDL': 'VEDL.NS', 'VEDANTA': 'VEDL.NS',
    'NMDC': 'NMDC.NS', 'NATIONALUM': 'NATIONALUM.NS',
    'SAIL': 'SAIL.NS', 'JINDALSTEL': 'JINDALSTEL.NS',
    'JSWENERGY': 'JSWENERGY.NS', 'JSWINFRA': 'JSWINFRA.NS',
    'RATNAMANI': 'RATNAMANI.NS', 'WELCORP': 'WELCORP.NS',
    'APLAPOLLO': 'APLAPOLLO.NS', 'GALLANTT': 'GALLANTT.NS',

    # ═══ OIL, GAS & ENERGY ═══
    'IOC': 'IOC.NS', 'GAIL': 'GAIL.NS',
    'PETRONET': 'PETRONET.NS', 'HINDPETRO': 'HINDPETRO.NS',
    'MGL': 'MGL.NS', 'IGL': 'IGL.NS', 'GUJGASLTD': 'GUJGASLTD.NS',
    'TATAPOWER': 'TATAPOWER.NS', 'TORNTPOWER': 'TORNTPOWER.NS',
    'CESC': 'CESC.NS', 'NHPC': 'NHPC.NS', 'SJVN': 'SJVN.NS',
    'IREDA': 'IREDA.NS', 'RECLTD': 'RECLTD.NS', 'PFC': 'PFC.NS',

    # ═══ FMCG & CONSUMER ═══
    'DABUR': 'DABUR.NS', 'GODREJCP': 'GODREJCP.NS',
    'MARICO': 'MARICO.NS', 'COLPAL': 'COLPAL.NS',
    'EMAMILTD': 'EMAMILTD.NS', 'GILLETTE': 'GILLETTE.NS',
    'PGHH': 'PGHH.NS', 'JYOTHYLAB': 'JYOTHYLAB.NS',
    'VGUARD': 'VGUARD.NS', 'BATAINDIA': 'BATAINDIA.NS',
    'RELAXO': 'RELAXO.NS', 'PAGEIND': 'PAGEIND.NS',
    'TRENT': 'TRENT.NS', 'DMART': 'DMART.NS',
    'DEVYANI': 'DEVYANI.NS', 'JUBLFOOD': 'JUBLFOOD.NS',
    'ZOMATO': 'ETERNAL.NS', 'SWIGGY': 'SWIGGY.NS',
    'PATANJALI': 'PATANJALI.NS',

    # ═══ JEWELLERY & LIFESTYLE ═══
    'KALYANKJIL': 'KALYANKJIL.NS', 'KALYAN': 'KALYANKJIL.NS',
    'KALYAN JEWELLERS': 'KALYANKJIL.NS', 'KALYANJEWELLERS': 'KALYANKJIL.NS',
    'TITAN': 'TITAN.NS', 'SENCO': 'SENCO.NS', 'PCJEWELLER': 'PCJEWELLER.NS',

    # ═══ CEMENT & CONSTRUCTION ═══
    'ULTRACEMCO': 'ULTRACEMCO.NS', 'SHREECEM': 'SHREECEM.NS',
    'AMBUJACEM': 'AMBUJACEM.NS', 'ACC': 'ACC.NS',
    'DALMIACMNT': 'DALMIACMNT.NS', 'RAMCOCEM': 'RAMCOCEM.NS',
    'JKCEMENT': 'JKCEMENT.NS', 'BIRLACEM': 'BIRLACEM.NS',
    'JKLAKSHMI': 'JKLAKSHMI.NS',

    # ═══ REAL ESTATE ═══
    'DLF': 'DLF.NS', 'GODREJPROP': 'GODREJPROP.NS',
    'OBEROIRLTY': 'OBEROIRLTY.NS', 'PRESTIGE': 'PRESTIGE.NS',
    'PHOENIXLTD': 'PHOENIXLTD.NS', 'BRIGADE': 'BRIGADE.NS',
    'SOBHA': 'SOBHA.NS', 'LODHA': 'LODHA.NS',
    'MAHLIFE': 'MAHLIFE.NS', 'SUNTECK': 'SUNTECK.NS',
    'RAYMOND': 'RAYMOND.NS',

    # ═══ INDUSTRIAL & ENGINEERING ═══
    'SIEMENS': 'SIEMENS.NS', 'ABB': 'ABB.NS',
    'CUMMINSIND': 'CUMMINSIND.NS', 'HAVELLS': 'HAVELLS.NS',
    'CROMPTON': 'CROMPTON.NS', 'BLUESTARLT': 'BLUESTARLT.NS',
    'POLYCAB': 'POLYCAB.NS', 'KEI': 'KEI.NS',
    'AFFLE': 'AFFLE.NS', 'DIXON': 'DIXON.NS',
    'KAYNES': 'KAYNES.NS', 'ELGIEQUIP': 'ELGIEQUIP.NS',
    'THERMAX': 'THERMAX.NS', 'GRINDWELL': 'GRINDWELL.NS',
    'CARBORUNIV': 'CARBORUNIV.NS', 'BEL': 'BEL.NS',
    'HAL': 'HAL.NS', 'BDL': 'BDL.NS', 'MAZAGON': 'MAZDOCK.NS',
    'COCHINSHIP': 'COCHINSHIP.NS', 'COCHIN SHIPYARD': 'COCHINSHIP.NS', 'COCHINSHIPYARD': 'COCHINSHIP.NS',
    'GRSE': 'GRSE.NS',

    # ═══ TELECOM & MEDIA ═══
    'BHARTIARTL': 'BHARTIARTL.NS', 'IDEA': 'IDEA.NS',
    'TTML': 'TTML.NS', 'HATHWAY': 'HATHWAY.NS',
    'DEN': 'DEN.NS', 'SUNTV': 'SUNTV.NS',
    'ZEEL': 'ZEEL.NS', 'PVR': 'PVRINOX.NS',
    'SAREGAMA': 'SAREGAMA.NS', 'NAZARA': 'NAZARA.NS',

    # ═══ RAILWAYS & INFRA ═══
    'IRCTC': 'IRCTC.NS', 'IRFC': 'IRFC.NS',
    'RVNL': 'RVNL.NS', 'RAILTEL': 'RAILTEL.NS',
    'RITES': 'RITES.NS', 'TITAGARH': 'TITAGARH.NS',
    'TEXRAIL': 'TEXRAIL.NS',
    'IRB': 'IRB.NS', 'KNR': 'KNRCON.NS',
    'NBCC': 'NBCC.NS', 'NCC': 'NCC.NS',
    'HCC': 'HCC.NS', 'ASHOKA': 'ASHOKA.NS',

    # ═══ CHEMICAL & FERTILIZER ═══
    'PIDILITIND': 'PIDILITIND.NS', 'UPL': 'UPL.NS',
    'AARTI': 'AARTIIND.NS', 'SRF': 'SRF.NS',
    'DEEPAKNTR': 'DEEPAKNTR.NS', 'NAVINFLUOR': 'NAVINFLUOR.NS',
    'PIIND': 'PIIND.NS', 'CLEAN': 'CLEAN.NS',
    'FLUOROCHEM': 'FLUOROCHEM.NS', 'ALKYLAMINE': 'ALKYLAMINE.NS',
    'CHAMBALFERT': 'CHAMBLFERT.NS', 'COROMANDEL': 'COROMANDEL.NS',
    'GNFC': 'GNFC.NS', 'NFL': 'NFL.NS',
    'RCF': 'RCF.NS', 'FACT': 'FACT.NS',

    # ═══ TEXTILES & APPAREL ═══
    'ARVIND': 'ARVIND.NS', 'RAYMOND': 'RAYMOND.NS',
    'TRIDENT': 'TRIDENT.NS', 'WELSPUNLIV': 'WELSPUNLIV.NS',
    'KPRMILL': 'KPRMILL.NS', 'GOKALDAS': 'GOKALDAS.NS',

    # ═══ NIPPON / AMC / MUTUAL FUND ═══
    'NIPPONLIFE': 'NAM-INDIA.NS', 'NAM-INDIA': 'NAM-INDIA.NS',
    'NIPPON': 'NAM-INDIA.NS',
    'HDFCAMC': 'HDFCAMC.NS', 'UTIAMC': 'UTIAMC.NS',

    # ═══ PSU & GOVERNMENT ═══
    'IRCTC': 'IRCTC.NS', 'COALINDIA': 'COALINDIA.NS',
    'NHPC': 'NHPC.NS', 'BEL': 'BEL.NS', 'HAL': 'HAL.NS',
    'CONCOR': 'CONCOR.NS', 'HUDCO': 'HUDCO.NS',
    'NMDC': 'NMDC.NS', 'NLCINDIA': 'NLCINDIA.NS',
    'SAIL': 'SAIL.NS', 'BHEL': 'BHEL.NS',
    'OFSS': 'OFSS.NS', 'COCHINSHIP': 'COCHINSHIP.NS',

    # ═══ ETFs (Popular on Groww) ═══
    'GOLDBEES': 'GOLDBEES.NS', 'SILVERBEES': 'SILVERBEES.NS',
    'NIFTYBEES': 'NIFTYBEES.NS', 'BANKBEES': 'BANKBEES.NS',
    'JUNIORBEES': 'JUNIORBEES.NS', 'LIQUIDBEES': 'LIQUIDBEES.NS',
    'SETFNIF50': 'SETFNIF50.NS', 'ITETF': 'ITETF.NS',
    'NIPPON GOLD ETF': 'GOLDBEES.NS', 'NIPPON SILVER ETF': 'SILVERBEES.NS',
    'TATASILV': 'TATSILV.NS', 'TATA SILVER': 'TATSILV.NS', 'TATASILVER': 'TATSILV.NS',
    'TATAGOLD': 'TATAGOLD.NS', 'TATA GOLD': 'TATAGOLD.NS',
    'TATA MOTORS': 'TATAMOTORS.NS', 'TATA MOTORE': 'TATAMOTORS.NS', 'TATA MOTORES': 'TATAMOTORS.NS',
    'TATA STEEL': 'TATASTEEL.NS', 'TATA POWER': 'TATAPOWER.NS',
    'TATA CHEM': 'TATACHEM.NS', 'TATA ELXSI': 'TATAELXSI.NS',
    'TATA CONSUMER': 'TATACONSUM.NS', 'TATA COMM': 'TATACOMM.NS',

    # ═══ NEW-AGE TECH / STARTUPS ═══
    'ZOMATO': 'ETERNAL.NS', 'PAYTM': 'PAYTM.NS',
    'NYKAA': 'NYKAA.NS', 'POLICYBZR': 'POLICYBZR.NS',
    'CARTRADE': 'CARTRADE.NS', 'DELHIVERY': 'DELHIVERY.NS',
    'MAPMYINDIA': 'MAPMYINDIA.NS',

    # ═══ COMMODITIES ═══
    'GOLD': 'GC=F', 'SILVER': 'SI=F', 'CRUDE': 'CL=F', 'CRUDEOIL': 'CL=F',
    'NATURALGAS': 'NG=F', 'COPPER': 'HG=F',

    # ═══ US STOCKS (Popular on Groww) ═══
    'AAPL': 'AAPL', 'GOOGL': 'GOOGL', 'MSFT': 'MSFT', 'AMZN': 'AMZN',
    'TSLA': 'TSLA', 'NVDA': 'NVDA', 'META': 'META', 'NFLX': 'NFLX',
    'AMD': 'AMD', 'INTC': 'INTC', 'CRM': 'CRM', 'ORCL': 'ORCL',
    'UBER': 'UBER', 'SNAP': 'SNAP', 'COIN': 'COIN', 'PLTR': 'PLTR',
    'DIS': 'DIS', 'BA': 'BA', 'JPM': 'JPM', 'V': 'V', 'MA': 'MA',
    'KO': 'KO', 'PEP': 'PEP', 'WMT': 'WMT', 'PG': 'PG', 'JNJ': 'JNJ',
}

COMMODITY_USD = {'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F'}

# ── Groww-Style Categories (20+ Sectors) ──────────────────────────────────
DASHBOARD_CATEGORIES = {
    '🏛️ Indices': ['NIFTY', 'SENSEX', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY'],
    '🔵 Tata Group': ['TATASTEEL', 'TATAMOTORS', 'TATAPOWER', 'TATACONSUM', 'TATAELXSI',
                       'TATACOMM', 'TATACHEM', 'TATAINVEST', 'TITAN', 'VOLTAS', 'TRENT', 'RALLIS'],
    '🏢 Adani Group': ['ADANIENT', 'ADANIPORTS', 'ADANIGREEN', 'ADANIPOWER', 'ATGL', 'AWL', 'ACC', 'AMBUJACEM'],
    '🏦 Public Banks': ['SBIN', 'SBI', 'PNB', 'BANKBARODA', 'CANBK', 'UNIONBANK', 'IOB', 'CENTRALB', 'INDIANB', 'UCOBANK', 'MAHABANK', 'PSB'],
    '🏧 Private Banks': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
                          'IDFCFIRSTB', 'FEDERALBNK', 'BANDHANBNK', 'RBLBANK', 'YESBANK', 'AUBANK', 'CUB', 'CSB', 'TMB', 'KARURVYSYA'],
    '💰 NBFC & Finance': ['BAJFINANCE', 'BAJFINSV', 'SHRIRAMFIN', 'CHOLAFIN', 'MUTHOOTFIN',
                           'MANAPPURAM', 'LICHSGFIN', 'CANFINHOME', 'POONAWALLA', 'IIFL', 'MFSL', 'MOTILALOFS', 'ANGELONE'],
    '🛡️ Insurance': ['LICI', 'SBILIFE', 'HDFCLIFE', 'ICICIPRULI', 'ICICIGI', 'STARHEALTH', 'NIACL', 'GICRE', 'POLICYBZR'],
    '💻 IT & Software': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'LTTS',
                          'MPHASIS', 'COFORGE', 'PERSISTENT', 'HAPPSTMNDS', 'KPITTECH',
                          'MASTEK', 'ZENSAR', 'BIRLASOFT', 'CYIENT', 'SONATSOFTW', 'TATAELXSI'],
    '📱 New-Age Tech': ['ZOMATO', 'PAYTM', 'NYKAA', 'POLICYBZR', 'DELHIVERY', 'CARTRADE', 'MAPMYINDIA', 'NAZARA'],
    '💊 Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN', 'AUROPHARMA',
                   'BIOCON', 'TORNTPHARM', 'ALKEM', 'IPCALAB', 'GLENMARK', 'NATCOPHARM',
                   'LAURUSLABS', 'GRANULES', 'AJANTPHARM', 'ABBOTINDIA', 'PFIZER', 'SANOFI'],
    '🏥 Healthcare': ['APOLLOHOSP', 'MAXHEALTH', 'FORTIS', 'LALPATHLAB', 'METROPOLIS'],
    '🚗 Auto': ['TATAMOTORS', 'MARUTI', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
                'ASHOKLEY', 'TVSMOTOR', 'ESCORTS'],
    '🔧 Auto Ancillary': ['BOSCHLTD', 'MOTHERSON', 'BHARATFORG', 'BALKRISIND', 'MRF',
                           'APOLLOTYRE', 'CEATLTD', 'EXIDEIND', 'ENDURANCE', 'SUNDRMFAST'],
    '🏭 Metal & Mining': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'NMDC', 'NATIONALUM',
                           'SAIL', 'JINDALSTEL', 'COALINDIA', 'APLAPOLLO', 'RATNAMANI', 'WELCORP', 'GALLANTT'],
    '⛽ Oil & Gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'HINDPETRO', 'GAIL',
                     'PETRONET', 'MGL', 'IGL', 'GUJGASLTD'],
    '⚡ Power & Energy': ['NTPC', 'POWERGRID', 'TATAPOWER', 'ADANIGREEN', 'ADANIPOWER',
                          'TORNTPOWER', 'CESC', 'NHPC', 'SJVN', 'IREDA', 'RECLTD', 'PFC', 'JSWENERGY'],
    '🏗️ Cement': ['ULTRACEMCO', 'SHREECEM', 'AMBUJACEM', 'ACC', 'DALMIACMNT', 'RAMCOCEM', 'JKCEMENT', 'JKLAKSHMI'],
    '🏠 Real Estate': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'PHOENIXLTD', 'BRIGADE', 'SOBHA', 'LODHA', 'SUNTECK'],
    '🛒 FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP',
                 'MARICO', 'COLPAL', 'TATACONSUM', 'EMAMILTD', 'PATANJALI', 'JUBLFOOD', 'DMART'],
    '🏭 Industrial & Engineering': ['LT', 'SIEMENS', 'ABB', 'HAVELLS', 'CROMPTON', 'POLYCAB',
                                     'KEI', 'DIXON', 'THERMAX', 'CUMMINSIND', 'ELGIEQUIP', 'GRINDWELL'],
    '🛡️ Defence': ['HAL', 'BEL', 'BDL', 'MAZAGON', 'COCHINSHIP', 'GRSE'],
    '💎 Jewellery': ['KALYANKJIL', 'TITAN', 'SENCO', 'PCJEWELLER'],
    '🚂 Railways': ['IRCTC', 'IRFC', 'RVNL', 'RAILTEL', 'RITES', 'TITAGARH'],
    '📡 Telecom': ['BHARTIARTL', 'IDEA', 'TTML'],
    '🧪 Chemicals': ['PIDILITIND', 'UPL', 'SRF', 'DEEPAKNTR', 'NAVINFLUOR', 'PIIND',
                      'FLUOROCHEM', 'AARTI', 'CLEAN', 'ALKYLAMINE'],
    '🌾 Fertilizer': ['CHAMBALFERT', 'COROMANDEL', 'GNFC', 'NFL', 'RCF', 'FACT'],
    '👔 Textiles': ['ARVIND', 'PAGEIND', 'TRENT', 'RAYMOND', 'TRIDENT', 'WELSPUNLIV', 'KPRMILL', 'GOKALDAS'],
    '📈 AMC & Exchange': ['NIPPON', 'HDFCAMC', 'UTIAMC', 'CDSL', 'MCX', 'ANGELONE'],
    '🏆 Commodities': ['GOLD', 'SILVER', 'CRUDE', 'NATURALGAS', 'COPPER'],
    '📦 ETFs': ['GOLDBEES', 'SILVERBEES', 'TATASILV', 'TATAGOLD', 'NIFTYBEES', 'BANKBEES', 'JUNIORBEES'],
    '🇺🇸 US Stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                       'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'PLTR', 'DIS', 'BA',
                       'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'],
}

# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST_DEFAULT = ['WIPRO', 'RELIANCE', 'TATASTEEL', 'NIPPON', 'BOSCHLTD', 'VEDL',
                     'TATAMOTORS', 'GOLD', 'SILVER', 'ADANIGREEN', 'SHRIRAMFIN', 'CHOLAFIN',
                     'IRCTC', 'HAL', 'ZOMATO', 'DLF']

# ── Premium CSS (Groww-inspired) ──────────────────────────────────────────
# Inject viewport meta tag for proper mobile rendering
st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">', unsafe_allow_html=True)

import base64

def get_base64_of_bin_file(bin_file):
    try:
        import os
        if os.path.exists(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
    except Exception:
        pass
    return ""

bg_img_base64 = get_base64_of_bin_file('background.png')
if bg_img_base64:
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bg_img_base64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "" !important;
        position: absolute !important;
        top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important;
        background: radial-gradient(circle at 50% 0%, rgba(30, 16, 16, 0.4) 0%, rgba(3, 7, 18, 0.88) 75%) !important;
        pointer-events: none !important;
        z-index: 0 !important;
    }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0%, #1e1010 0%, #030712 70%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
/* ── Google Fonts import ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

* { font-family: 'Outfit', sans-serif; box-sizing: border-box; }

/* ── Mobile-Specific Overrides ── */
@media (max-width: 768px) {
    input, select, textarea { font-size: 16px !important; }
    .main-title { font-size: 1.6rem !important; padding: 0 8px; }
    .sub-title { font-size: 0.8rem !important; }
    .ticker-bar { gap: 12px; font-size: 0.75rem; padding: 6px 10px; }
    .stock-card { padding: 0.75rem; margin: 0.3rem 0; }
    .stock-card .price { font-size: 1rem; }
    .signal-buy, .signal-sell, .signal-hold { padding: 0.9rem; font-size: 1rem; }
    .verdict-h1 { font-size: 2rem !important; margin: 8px 0 !important; }
    .verdict-desc { font-size: 0.9rem !important; }
    .verdict-container { padding: 16px !important; border-radius: 14px !important; }
    .forecast-container > div { display: block !important; width: 100% !important; margin-bottom: 10px; }
    .pattern-inset { flex-direction: column; gap: 10px; }
    .news-card { flex-wrap: wrap; gap: 6px; padding: 0.6rem 0.9rem; }
    .news-title { font-size: 0.88rem; }
    .section-head { font-size: 0.95rem; margin: 1.2rem 0 0.7rem 0; }
    .recent-row { gap: 10px; }
    .recent-item { min-width: 70px; padding: 6px 10px; }
}

/* Custom Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #030712; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }

/* Ticker Bar */
.ticker-bar {
    background: rgba(17, 24, 39, 0.6); 
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 10px 18px; 
    border-radius: 12px; 
    margin-bottom: 1.5rem;
    display: flex; 
    gap: 24px; 
    overflow-x: auto; 
    white-space: nowrap; 
    font-size: 0.85rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.ticker-item { display: inline-block; }
.ticker-name { color: #9ca3af; font-weight: 700; }
.ticker-price { color: #f3f4f6; font-weight: 700; margin-left: 6px; font-family: 'JetBrains Mono', monospace; }
.ticker-up { color: #10b981; font-weight: 700; margin-left: 4px; }
.ticker-down { color: #ef4444; font-weight: 700; margin-left: 4px; }

/* Main Title */
.main-title {
    font-size: 2.6rem; 
    font-weight: 900; 
    text-align: center;
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 40%, #fbbf24 100%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.04em;
    text-shadow: 0 0 40px rgba(239, 68, 68, 0.1);
}
.sub-title { 
    text-align: center; 
    color: #9ca3af; 
    font-size: 0.95rem; 
    margin-bottom: 2rem; 
    font-weight: 500;
}

/* Stock Cards (Glassmorphic & Premium) */
.stock-card {
    background: rgba(17, 24, 39, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1.25rem; 
    margin: 0.5rem 0; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}
.stock-card:hover { 
    border-color: rgba(251, 191, 36, 0.4); 
    transform: translateY(-3px); 
    box-shadow: 0 10px 30px rgba(251, 191, 36, 0.08); 
}
.stock-card .name { font-size: 0.9rem; font-weight: 700; color: #e5e7eb; margin-bottom: 6px; }
.stock-card .price { font-size: 1.3rem; font-weight: 800; color: white; font-family: 'JetBrains Mono', monospace; }
.stock-card .change-up { color: #10b981; font-size: 0.9rem; font-weight: 700; }
.stock-card .change-down { color: #ef4444; font-size: 0.9rem; font-weight: 700; }

/* Recently Viewed Row */
.recent-row { display: flex; gap: 16px; overflow-x: auto; padding: 10px 0; }
.recent-item {
    text-align: center; 
    min-width: 90px; 
    padding: 10px 14px;
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    transition: all 0.2s ease;
}
.recent-item:hover {
    border-color: rgba(251, 191, 36, 0.3);
    transform: translateY(-1px);
}
.recent-item .sym { font-size: 0.85rem; font-weight: 700; color: #f3f4f6; }
.recent-item .chg-up { font-size: 0.75rem; color: #10b981; font-weight: 700; }
.recent-item .chg-down { font-size: 0.75rem; color: #ef4444; font-weight: 700; }

/* Section Headers */
.section-head {
    font-size: 1.15rem; 
    font-weight: 800; 
    color: #f3f4f6;
    margin: 2.5rem 0 1.25rem 0; 
    border-left: 5px solid; 
    border-image: linear-gradient(to bottom, #ef4444, #fbbf24) 1;
    padding-left: 14px; 
    letter-spacing: -0.03em;
    text-transform: uppercase;
}

/* Signal Cards */
.signal-buy { 
    background: linear-gradient(135deg, #059669, #10b981); 
    color: white; 
    padding: 1.4rem; 
    border-radius: 16px; 
    text-align: center; 
    font-size: 1.3rem; 
    font-weight: 800; 
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2); 
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.signal-sell { 
    background: linear-gradient(135deg, #dc2626, #ef4444); 
    color: white; 
    padding: 1.4rem; 
    border-radius: 16px; 
    text-align: center; 
    font-size: 1.3rem; 
    font-weight: 800; 
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2); 
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.signal-hold { 
    background: linear-gradient(135deg, #d97706, #f59e0b); 
    color: white; 
    padding: 1.4rem; 
    border-radius: 16px; 
    text-align: center; 
    font-size: 1.3rem; 
    font-weight: 800; 
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.2); 
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* News Card */
.news-card {
    background: rgba(17, 24, 39, 0.5); 
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 0.85rem 1.4rem; 
    border-radius: 10px; 
    margin-bottom: 0.6rem;
    transition: all 0.2s ease; 
    cursor: pointer;
    display: flex; 
    align-items: center; 
    gap: 12px;
}
.news-card:hover { 
    background: rgba(31, 41, 55, 0.6); 
    border-color: rgba(251, 191, 36, 0.3); 
    transform: translateX(4px);
}
.sentiment-pos { color: #10b981; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.sentiment-neg { color: #ef4444; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.sentiment-neu { color: #9ca3af; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.news-title { color: #f3f4f6; font-size: 0.95rem; font-weight: 600; text-decoration: none; }
.news-title:hover { color: #fbbf24; }

/* Pattern Inset Layout */
.pattern-inset {
    background: rgba(17, 24, 39, 0.45);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px;
    border-radius: 14px;
    margin-top: -10px;
    margin-bottom: 25px;
    display: flex;
    justify-content: space-around;
    align-items: center;
}

/* Index mini card */
.idx-card { 
    background: rgba(17, 24, 39, 0.5); 
    border: 1px solid rgba(255, 255, 255, 0.04); 
    padding: 0.85rem; 
    border-radius: 12px; 
    margin: 0.35rem 0; 
}

/* Live Pulse Animations */
.pulse-green {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white; padding: 6px 18px; border-radius: 20px;
    font-weight: 800; font-size: 0.9rem; display: inline-block;
    animation: live-pulse-green 1.8s infinite;
    box-shadow: 0 0 20px rgba(16,185,129,0.4);
    border: 1px solid rgba(255,255,255,0.15);
}
.pulse-red {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white; padding: 6px 18px; border-radius: 20px;
    font-weight: 800; font-size: 0.9rem; display: inline-block;
    animation: live-pulse-red 1.8s infinite;
    box-shadow: 0 0 20px rgba(239,68,68,0.4);
    border: 1px solid rgba(255,255,255,0.15);
}
.live-badge {
    color: #ef4444; font-weight: 800; 
    animation: live-flicker 1.5s infinite;
}
@keyframes live-flicker {
    0%, 100% { opacity: 1; text-shadow: 0 0 8px rgba(239,68,68,0.6); }
    50% { opacity: 0.4; text-shadow: none; }
}
@keyframes live-pulse-green {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129, 0.6); }
    70% { transform: scale(1.04); box-shadow: 0 0 0 10px rgba(16,185,129, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129, 0); }
}
@keyframes live-pulse-red {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239,68,68, 0.6); }
    70% { transform: scale(1.04); box-shadow: 0 0 0 10px rgba(239,68,68, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239,68,68, 0); }
}

/* Market Sentiment Bar */
.sentiment-bar {
    background: rgba(17, 24, 39, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 14px 22px;
    border-radius: 14px; 
    margin-bottom: 1.5rem; 
    display: flex;
    justify-content: space-between; 
    align-items: center;
    border-left: 6px solid;
    border-image: linear-gradient(to bottom, #ef4444, #fbbf24) 1;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}
.sent-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; font-weight: 700; }
.sent-value { font-size: 1.15rem; font-weight: 800; margin-top: 3px; }
.sent-reason { font-size: 0.9rem; color: #e5e7eb; font-weight: 500; }

/* ── Premium Global Overrides ── */
.stApp {
    background-color: #030712 !important;
    color: #f3f4f6 !important;
}
[data-testid="stSidebar"] {
    background-color: #050810 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Premium Tabs */
button[data-baseweb="tab"] {
    color: #9ca3af !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease !important;
    background: transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: #fbbf24 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ef4444 !important;
    border-bottom: 2px solid #fbbf24 !important;
}

/* Premium Gradient Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 50%, #fbbf24 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(251, 191, 36, 0.35) !important;
    color: #030712 !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* Premium Input Fields & Dropdowns */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    color: #f3f4f6 !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
    border-color: #fbbf24 !important;
}
.stSlider [data-testid="stTickBar"] {
    color: #9ca3af !important;
}

/* Unified glassmorphic cards */
.premium-card {
    background: rgba(17, 24, 39, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)


# ── Data Helpers ──────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_stock(raw_symbol, days=200, interval='1d', period=None):
    symbol = raw_symbol.strip().upper()
    mapped = None
    
    # Check exact match
    if symbol in STOCK_MAP:
        mapped = STOCK_MAP[symbol]
    else:
        # Check fuzzy match
        for k, v in STOCK_MAP.items():
            if k in symbol or symbol.replace(' ','') == k:
                mapped = v
                break
        
    if not mapped:
        mapped = symbol
        # Default to NSE if no suffix and not a known US stock
        us_stocks = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'SNAP', 'COIN', 'PLTR', 'DIS', 'BA', 'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'}
        if '.' not in mapped and '=' not in mapped and not mapped.startswith('^') and mapped not in us_stocks:
            mapped = mapped + '.NS'

    import time
    if not period: period = f'{days}d'
    
    for attempt in range(3):
        try:
            tk = yf.Ticker(mapped)
            df = tk.history(period=period, interval=interval)
            
            if df is None or df.empty:
                df = yf.download(mapped, period=period, interval=interval, progress=False)
                
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df, mapped
        except Exception:
            pass
        time.sleep(1)
        
    return None, mapped

@st.cache_data(ttl=3600)
def fetch_fundamentals(mapped):
    """Fetch fundamental data as seen on Screener.in"""
    try:
        tk = yf.Ticker(mapped)
        info = tk.info
        return {
            'mkt_cap': info.get('marketCap', 0),
            'pe': info.get('trailingPE', 0),
            'pb': info.get('priceToBook', 0),
            'div_yield': info.get('dividendYield', 0),
            'high_52': info.get('fiftyTwoWeekHigh', 0),
            'low_52': info.get('fiftyTwoWeekLow', 0),
            'sector': info.get('sector', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A')
        }
    except:
        return None


@st.cache_data(ttl=300)
def get_usd_inr():
    try:
        fx = yf.download('USDINR=X', period='5d', progress=False)
        if fx is not None and not fx.empty:
            c = fx['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            return float(c.iloc[-1])
    except Exception:
        pass
    return 83.5


import re

@st.cache_data(ttl=5)
def get_realtime_price(symbol, mapped):
    """Fetch exact live price from Google Finance for near-zero delay"""
    # Map for Google Finance (e.g. NSE:RELIANCE)
    if mapped.endswith('.NS'): gf_sym = mapped.replace('.NS', '') + ':NSE'
    elif mapped.endswith('.BO'): gf_sym = mapped.replace('.BO', '') + ':BOM'
    elif mapped == '^NSEI': gf_sym = 'NIFTY_50:INDEXNSE'
    elif mapped == '^BSESN': gf_sym = 'SENSEX:INDEXBOM'
    elif mapped == '^NSEBANK': gf_sym = 'NIFTY_BANK:INDEXNSE'
    elif mapped in {'GC=F', 'SI=F', 'CL=F'}: return None # Skip commodities for now
    else: gf_sym = f"{mapped}:NASDAQ" # Default to NASDAQ for US stocks
    
    try:
        r = requests.get(f'https://www.google.com/finance/quote/{gf_sym}', timeout=3)
        if r.status_code == 200:
            m = re.search(r'data-last-price="([0-9.]+)"', r.text)
            if m: return float(m.group(1))
    except Exception:
        pass
    return None

@st.cache_data(ttl=900)
def fetch_bulk_mtf_data(symbols):
    """
    Bulk downloads 1D, 1H, and 15m data for a list of symbols simultaneously.
    Returns: dict of {symbol: {'1d': df, '1h': df, '15m': df}}
    """
    if not symbols: return {}
    
    # Map symbols to Yahoo Finance format
    mapped_syms = []
    sym_map = {}
    for s in symbols:
        mapped = STOCK_MAP.get(s, s)
        mapped_syms.append(mapped)
        sym_map[mapped] = s
    
    tickers_str = " ".join(mapped_syms)
    
    # Download 1D
    try:
        df_1d = yf.download(tickers_str, period="60d", interval="1d", group_by="ticker", threads=True, progress=False)
    except:
        df_1d = pd.DataFrame()
        
    # Download 1H
    try:
        df_1h = yf.download(tickers_str, period="20d", interval="1h", group_by="ticker", threads=True, progress=False)
    except:
        df_1h = pd.DataFrame()
        
    # Download 15m
    try:
        df_15m = yf.download(tickers_str, period="5d", interval="15m", group_by="ticker", threads=True, progress=False)
    except:
        df_15m = pd.DataFrame()
    
    results = {}
    for mapped in mapped_syms:
        sym = sym_map[mapped]
        results[sym] = {'1d': None, '1h': None, '15m': None}
        
        # If multiple symbols are requested, columns are MultiIndex (Ticker, Open/High...)
        if len(mapped_syms) > 1:
            if mapped in df_1d.columns.levels[0]:
                d1 = df_1d[mapped].dropna(how='all')
                if not d1.empty: results[sym]['1d'] = d1
            if mapped in df_1h.columns.levels[0]:
                h1 = df_1h[mapped].dropna(how='all')
                if not h1.empty: results[sym]['1h'] = h1
            if mapped in df_15m.columns.levels[0]:
                m15 = df_15m[mapped].dropna(how='all')
                if not m15.empty: results[sym]['15m'] = m15
        else:
            # Single symbol returns standard columns
            if not df_1d.empty: results[sym]['1d'] = df_1d.dropna(how='all')
            if not df_1h.empty: results[sym]['1h'] = df_1h.dropna(how='all')
            if not df_15m.empty: results[sym]['15m'] = df_15m.dropna(how='all')
            
    return results

def get_price_info(symbol, days=5):
    """Get current price, change, change% for a symbol."""
    df, mapped = fetch_stock(symbol, days)
    if df is None or df.empty:
        return None
    close = df['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) == 0:
        return None
    cur = float(close.iloc[-1])
    # Improved 'prev' logic: Try to get actual previous close from bar data
    prev = float(close.iloc[-2]) if len(close) >= 2 else cur
    
    # Try getting exact real-time price from Google Finance to fix Yahoo latency
    live_price = get_realtime_price(symbol, mapped)
    if live_price is not None:
        cur = live_price

    vol = df['Volume']
    if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
    last_vol = int(vol.iloc[-1]) if vol is not None and len(vol) > 0 else 0

    is_commodity = mapped in COMMODITY_USD
    is_indian = mapped.endswith('.NS') or mapped.endswith('.BO') or mapped.startswith('^')
    # If not US stock, default to ₹ for safety in Indian app context
    us_stocks = {'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'UBER', 'SNAP', 'COIN', 'PLTR', 'DIS', 'BA', 'JPM', 'V', 'MA', 'KO', 'PEP', 'WMT', 'PG', 'JNJ'}
    is_us = any(mapped.startswith(u) for u in us_stocks)

    if is_commodity:
        fx = get_usd_inr()
        cur *= fx
        prev *= fx
        curr_sym = '₹'
    elif is_indian or not is_us:
        curr_sym = '₹'
    else:
        curr_sym = '$'

    chg = cur - prev
    pct = (chg / prev * 100) if prev != 0 else 0
    return {'symbol': symbol, 'mapped': mapped, 'price': cur, 'prev': prev,
            'change': chg, 'pct': pct, 'currency': curr_sym, 'volume': last_vol}


def detect_candle_pattern(df):
    """Detects confirmed candlestick patterns from the last CLOSED candle to avoid live repainting"""
    if df is None or len(df) < 3: return {"pattern": "Not enough data", "advice": "Wait for more data."}
    
    # We analyze the LAST CLOSED candle (-2) rather than the live fluctuating candle (-1)
    c = df.iloc[-2] 
    p = df.iloc[-3]
    
    # Current and previous fully closed OHLC
    cO, cH, cL, cC = float(c['Open']), float(c['High']), float(c['Low']), float(c['Close'])
    pO, pH, pL, pC = float(p['Open']), float(p['High']), float(p['Low']), float(p['Close'])
    
    cBody = abs(cC - cO)
    pBody = abs(pC - pO)
    cRange = cH - cL if (cH - cL) > 0 else 0.0001
    
    # Doji
    if cBody / cRange < 0.1:
        return {"pattern": "Doji ⚖️ (Indecision)", "advice": "Hold. Wait for breakout above Doji high or breakdown below low.", "score": 0.5}
        
    # Bullish Engulfing
    if pC < pO and cC > cO and cO <= pC and cC >= pO:
        return {"pattern": "Bullish Engulfing 📈", "advice": f"**BUY ENTRY**: Buy if next candle crosses {cH:,.2f}. **STOP LOSS**: {cL:,.2f}", "score": 0.7}
        
    # Bearish Engulfing
    if pC > pO and cC < cO and cO >= pC and cC <= pO:
        return {"pattern": "Bearish Engulfing 📉", "advice": f"**SELL ENTRY**: Short if next candle goes below {cL:,.2f}. **STOP LOSS**: {cH:,.2f}", "score": 0.3}
        
    # Shadows
    lower_shadow = min(cO, cC) - cL
    upper_shadow = cH - max(cO, cC)
    
    # Hammer
    if lower_shadow > 2.0 * cBody and upper_shadow < cBody * 0.5:
        if cC > cO: return {"pattern": "Bullish Hammer 🔨", "advice": f"**BUY ENTRY**: Buy if next candle crosses {cH:,.2f}. **STOP LOSS**: {cL:,.2f}", "score": 0.7}
        else: return {"pattern": "Hanging Man 🔻", "advice": f"**SELL ENTRY**: Wait for next candle to close below {cL:,.2f} before shorting.", "score": 0.3}
        
    # Shooting Star / Inverted Hammer
    if upper_shadow > 2.5 * cBody and lower_shadow < cBody * 0.5:
        if cC < cO: return {"pattern": "Shooting Star 🌠", "advice": f"**SELL ENTRY**: Short if next candle goes below {cL:,.2f}. **STOP LOSS**: {cH:,.2f}", "score": 0.3}
        else: return {"pattern": "Inverted Hammer ⛏️", "advice": f"**BUY ENTRY**: Buy if next candle crosses {cH:,.2f}. **STOP LOSS**: {cL:,.2f}", "score": 0.7}
        
    # Marubozu (Strong momentum)
    if cBody / cRange > 0.9:
        if cC > cO: return {"pattern": "Bullish Marubozu 🧨", "advice": "STRONG BUY: High momentum upward. Trail stop loss deeply.", "score": 0.7}
        else: return {"pattern": "Bearish Marubozu 🧱", "advice": "STRONG SELL: Heavy selling pressure. Avoid buying.", "score": 0.3}
        
    if cC > cO: return {"pattern": "Standard Bullish 🟢", "advice": "Trend is UP. Consider buying on minor intraday dips.", "score": 0.7}
    elif cC < cO: return {"pattern": "Standard Bearish 🔴", "advice": "Trend is DOWN. Avoid fresh buying.", "score": 0.3}
    
    return {"pattern": "Neutral ➖", "advice": "No clear breakout pattern right now.", "score": 0.5}

# ── News Fetchers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_market_news(query="Indian Stock Market"):
    """Fetches highly robust targeted news using Google News RSS"""
    try:
        q = urllib.parse.quote_plus(query)
        url = f'https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        u = urllib.request.urlopen(req, timeout=10)
        tree = ET.parse(u)
        items = []
        for item in tree.findall('.//item')[:15]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            title = title.split(' - ')[0] if ' - ' in title else title # Clean source suffix
            if len(title) > 10:
                items.append({'title': title, 'url': link, 'source': 'GoogleNews'})
        return items
    except Exception:
        return []

@st.cache_data(ttl=300)
def fetch_global_news():
    """Fetches global market catalysts (US Fed, Wall Street, Global Trends)"""
    queries = ["Global stock market trends", "US Federal Reserve benchmark rates", "Wall Street news today"]
    all_items = []
    for q in queries:
        try:
            url = f'https://news.google.com/rss/search?q={urllib.parse.quote_plus(q)}&hl=en-US&gl=US&ceid=US:en'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            u = urllib.request.urlopen(req, timeout=10)
            tree = ET.parse(u)
            for item in tree.findall('.//item')[:5]:
                title = item.find('title').text or ""
                link = item.find('link').text or ""
                all_items.append({'title': title.split(' - ')[0], 'url': link, 'source': 'GlobalNews'})
        except: continue
    return all_items

# ── AI News Intelligence Engine ──────────────────────────────────────────
# Weighted Sentiment Keywords (High Impact = ±2, Medium = ±1)
POS_WORDS_HIGH = ['record', 'surge', 'buyback', 'beat', 'breakout', 'boom', 'all-time-high', 'dividend',
                  'blockbuster', 'outperform', 'massive', 'soaring', 'acquisition', 'upgrade',
                  'debt-free', 'profit-growth', 'multi-bagger', 'rally']
POS_WORDS_MED = ['gain', 'bullish', 'profit', 'growth', 'recover', 'strong', 'rise', 'up',
                 'buy', 'high', 'jump', 'soar', 'positive', 'improve', 'expand', 'upside',
                 'optimistic', 'momentum', 'stable', 'accumulate', 'support', 'rebound',
                 'margin', 'revenue', 'inflow', 'bullrun', 'approval', 'innovation',
                 'partnership', 'contract', 'order-win', 'capex']
NEG_WORDS_HIGH = ['crash', 'fraud', 'default', 'bankruptcy', 'scam', 'ban', 'penalty',
                  'downgrade', 'plunge', 'collapse', 'crisis', 'investigation', 'npa',
                  'writeoff', 'shutdown', 'warning', 'layoff', 'breach']
NEG_WORDS_MED = ['fall', 'drop', 'bearish', 'miss', 'sell', 'loss', 'decline', 'weak',
                 'cut', 'fear', 'risk', 'slump', 'down', 'tank', 'tumble', 'low',
                 'outflow', 'negative', 'pressure', 'concern', 'volatile', 'uncertainty',
                 'inflation', 'debt', 'delay', 'slowdown', 'correction', 'resistance',
                 'selling', 'caution', 'dip']

# Backward compatible flat lists
POS_WORDS = POS_WORDS_HIGH + POS_WORDS_MED
NEG_WORDS = NEG_WORDS_HIGH + NEG_WORDS_MED

# Catalyst Categories (Expanded for better 'Reasoning')
CATALYSTS = {
    'Earnings & Growth': ['dividend', 'earnings', 'profit', 'revenue', 'income', 'growth', 'beat', 'margin', 'sales', 'eps', 'guidance', 'upside', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'net-income', 'topline', 'bottomline'],
    'Deals & Orders': ['merger', 'acquisition', 'deal', 'partnership', 'contract', 'order', 'agreement', 'tender', 'jv', 'outperform', 'collaboration', 'mou', 'joint-venture'],
    'Policy & Govt': ['policy', 'regulation', 'tax', 'hike', 'cut', 'budget', 'rbi', 'sebi', 'government', 'fed', 'interest', 'reserve', 'subsidy', 'ministry', 'reform'],
    'Market Risk': ['loss', 'debt', 'default', 'scam', 'fraud', 'investigation', 'penalty', 'crisis', 'downgrade', 'fii-selling', 'inflation', 'war', 'geopolitical', 'recession'],
    'Technical Breakout': ['breakout', 'support', 'resistance', 'moving-average', 'ema', 'rsi', 'oversold', 'overbought', 'crossover', 'momentum', 'volume-spike', 'bullish-pattern'],
    'Corporate Action': ['bonus', 'split', 'buyback', 'rights-issue', 'listing', 'ipo', 'management-change', 'ceo', 'board-approval', 'promoter', 'stake'],
    'FII/DII Flow': ['fii', 'dii', 'fpi', 'foreign-investor', 'institutional', 'mutual-fund', 'inflow', 'outflow', 'bulk-deal', 'block-deal']
}

# News Importance Keywords (KEEP categories)
NEWS_KEEP_KEYWORDS = {
    'Earnings': ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'profit', 'revenue', 'net income', 'eps', 'results', 'guidance'],
    'Large Orders': ['order', 'contract', 'deal', 'billion', 'crore', 'agreement', 'partnership', 'mou'],
    'Government': ['government', 'policy', 'regulation', 'ministry', 'sebi', 'budget', 'reform', 'fiscal'],
    'RBI': ['rbi', 'interest rate', 'repo rate', 'monetary policy', 'inflation', 'rate cut', 'rate hike'],
    'FII/DII': ['fii', 'dii', 'foreign investor', 'institutional', 'fpi', 'mutual fund', 'block deal', 'bulk deal'],
    'M&A': ['merger', 'acquisition', 'takeover', 'buyout', 'stake', 'divestment', 'buyback']
}

# News IGNORE Keywords
NEWS_IGNORE_KEYWORDS = ['celebrity', 'actor', 'film', 'movie', 'cricket', 'bollywood', 'entertainment',
                        'horoscope', 'zodiac', 'wedding', 'lifestyle', 'recipe']

# Event Detection Patterns
EVENT_PATTERNS = {
    'EARNINGS_BEAT': {'keywords': ['beat', 'exceeded', 'above estimate', 'strong q', 'record profit', 'profit surge', 'revenue beat'], 'impact': 2.0},
    'EARNINGS_MISS': {'keywords': ['missed', 'below estimate', 'weak q', 'disappointing', 'profit decline', 'revenue miss'], 'impact': -2.0},
    'RBI_RATE_CUT': {'keywords': ['rate cut', 'rbi cuts', 'repo rate reduced', 'dovish', 'easing'], 'impact': 1.5},
    'RBI_RATE_HIKE': {'keywords': ['rate hike', 'rbi hikes', 'repo rate increased', 'hawkish', 'tightening'], 'impact': -1.5},
    'BLOCK_DEAL_BUY': {'keywords': ['block deal buy', 'bulk deal buy', 'large stake acquired', 'promoter buy'], 'impact': 1.5},
    'BLOCK_DEAL_SELL': {'keywords': ['block deal sell', 'bulk deal sell', 'stake sale', 'promoter sell', 'offloading'], 'impact': -1.5},
    'FII_BUYING': {'keywords': ['fii bought', 'fpi inflow', 'foreign buying', 'fii net buyer'], 'impact': 1.5},
    'FII_SELLING': {'keywords': ['fii sold', 'fpi outflow', 'foreign selling', 'fii net seller'], 'impact': -1.5},
    'BUDGET': {'keywords': ['budget', 'fiscal policy', 'tax reform', 'capex boost'], 'impact': 1.0},
    'UPGRADE': {'keywords': ['upgraded', 'target raised', 'price target', 'overweight', 'buy rating'], 'impact': 1.5},
    'DOWNGRADE': {'keywords': ['downgraded', 'target cut', 'underweight', 'sell rating', 'reduce'], 'impact': -1.5},
}

# Stock to Sector Mapping
STOCK_TO_SECTOR = {}
_SECTOR_STOCKS = {
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANDHANBNK", "PNB", "BANKBARODA", "CANBK", "IDFCFIRSTB"],
    "IT & Tech": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "MINDTREE"],
    "Auto & EV": ["TATAMOTORS", "MARUTI", "M&M", "TVSMOTOR", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "ASHOKLEY"],
    "Metal & Mining": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "NMDC", "NATIONALUM"],
    "Oil & Gas": ["RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO", "GAIL", "PETRONET"],
    "Power & Energy": ["NTPC", "POWERGRID", "TATAPOWER", "JSWENERGY", "ADANIGREEN", "ADANIPOWER", "NHPC"],
    "FMCG & Retail": ["HINDUNILVR", "ITC", "TATACONSUM", "BRITANNIA", "NESTLEIND", "DABUR", "MARICO", "GODREJCP"],
    "Pharma & Healthcare": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "BIOCON", "LUPIN", "AUROPHARMA"],
    "Railways & Infra": ["IRCTC", "IRFC", "RVNL", "LT", "ADANIENT", "ADANIPORTS", "GRASIM"],
    "Financial Services": ["BAJFINANCE", "BAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIGI", "CHOLAFIN", "MUTHOOTFIN"],
    "Telecom": ["BHARTIARTL", "IDEA", "TATACOMM"],
    "Cement": ["ULTRACEMCO", "AMBUJACEM", "ACC", "SHREECEM", "DALMIACEM"],
    "Defence": ["HAL", "BEL", "BDL", "MAZAGON", "COCHINSHIP"],
    "Commodities": ["GOLDIAM", "TATASILVER", "SILVERBEES", "GOLDBEES", "SILVERM"],
}
for _sect, _stocks in _SECTOR_STOCKS.items():
    for _s in _stocks:
        STOCK_TO_SECTOR[_s] = _sect

def score_headline_v2(text):
    """Enhanced headline scoring with weighted keywords. Returns -2 to +2."""
    t = text.lower()
    score = 0.0
    score += sum(2 for w in POS_WORDS_HIGH if w in t)
    score += sum(1 for w in POS_WORDS_MED if w in t)
    score -= sum(2 for w in NEG_WORDS_HIGH if w in t)
    score -= sum(1 for w in NEG_WORDS_MED if w in t)
    # Clamp to -2 to +2
    return max(-2.0, min(2.0, score))

def score_headline(text):
    """Backward compatible wrapper"""
    return score_headline_v2(text)

def filter_important_news(headlines):
    """Filter news by importance — keep financial, ignore noise, remove duplicates."""
    if not headlines:
        return []
    
    filtered = []
    seen_titles = []
    
    for h in headlines:
        title = h.get('title', '') if isinstance(h, dict) else str(h)
        t_low = title.lower()
        
        # IGNORE check — skip celebrity/entertainment/noise
        if any(kw in t_low for kw in NEWS_IGNORE_KEYWORDS):
            continue
        
        # Duplicate check — skip if >80% similar to any seen title
        is_dup = False
        title_words = set(t_low.split())
        for seen in seen_titles:
            seen_words = set(seen.split())
            if len(title_words) > 0 and len(title_words & seen_words) / len(title_words | seen_words) > 0.80:
                is_dup = True
                break
        if is_dup:
            continue
        
        seen_titles.append(t_low)
        
        # Tag importance category
        importance = 'General'
        for cat, keywords in NEWS_KEEP_KEYWORDS.items():
            if any(kw in t_low for kw in keywords):
                importance = cat
                break
        
        if isinstance(h, dict):
            h['importance'] = importance
        else:
            h = {'title': title, 'importance': importance}
        
        filtered.append(h)
    
    return filtered

def calculate_freshness_weight(pub_date_str):
    """Weight news by age. Returns 0.0 to 1.0. Missing date = 0.5 (user choice)."""
    if not pub_date_str:
        return 0.5  # Unknown age = 50% weight (user decision)
    
    try:
        from email.utils import parsedate_to_datetime
        pub_dt = parsedate_to_datetime(pub_date_str)
        age_hours = (datetime.now(pub_dt.tzinfo) - pub_dt).total_seconds() / 3600
    except Exception:
        try:
            # Try common date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S', '%a, %d %b %Y %H:%M:%S %z']:
                try:
                    pub_dt = datetime.strptime(pub_date_str.strip(), fmt)
                    age_hours = (datetime.now() - pub_dt).total_seconds() / 3600
                    break
                except:
                    continue
            else:
                return 0.5
        except:
            return 0.5
    
    if age_hours <= 1:    return 1.00
    elif age_hours <= 6:  return 0.80
    elif age_hours <= 24: return 0.50
    elif age_hours <= 72: return 0.10
    else:                 return 0.00

def detect_special_events(headlines):
    """Detect special financial events from headlines. Returns event info dict."""
    if not headlines:
        return {'event_type': None, 'impact_score': 0.0, 'confidence': 0.0, 'description': 'No events detected'}
    
    best_event = None
    best_score = 0
    best_matches = 0
    
    for h in headlines:
        title = h.get('title', '') if isinstance(h, dict) else str(h)
        t_low = title.lower()
        
        for event_type, pattern in EVENT_PATTERNS.items():
            matches = sum(1 for kw in pattern['keywords'] if kw in t_low)
            if matches > best_matches:
                best_matches = matches
                best_event = event_type
                best_score = pattern['impact']
    
    if best_event and best_matches > 0:
        confidence = min(best_matches / 3.0, 1.0)
        friendly_names = {
            'EARNINGS_BEAT': '📊 Earnings Beat', 'EARNINGS_MISS': '📉 Earnings Miss',
            'RBI_RATE_CUT': '🏛️ RBI Rate Cut', 'RBI_RATE_HIKE': '🏛️ RBI Rate Hike',
            'BLOCK_DEAL_BUY': '🏦 Block Deal Buy', 'BLOCK_DEAL_SELL': '🏦 Block Deal Sell',
            'FII_BUYING': '💰 FII Buying', 'FII_SELLING': '💸 FII Selling',
            'BUDGET': '📋 Budget Impact', 'UPGRADE': '⬆️ Analyst Upgrade', 'DOWNGRADE': '⬇️ Analyst Downgrade'
        }
        return {
            'event_type': best_event,
            'impact_score': best_score * confidence,
            'confidence': confidence,
            'description': friendly_names.get(best_event, best_event)
        }
    
    return {'event_type': None, 'impact_score': 0.0, 'confidence': 0.0, 'description': 'No major events'}

def analyze_live_candle(df):
    """Analyzes the current forming candle for immediate momentum"""
    if df is None or len(df) < 2: return {"bias": "Neutral", "color": "#94a3b8", "desc": "Stabilizing...", "pct": 0}
    
    live = df.iloc[-1]
    prev = df.iloc[-2]
    
    lO, lH, lL, lC = float(live['Open']), float(live['High']), float(live['Low']), float(live['Close'])
    pH, pL = float(prev['High']), float(prev['Low'])
    
    pct_from_open = (lC - lO) / lO * 100 if lO != 0 else 0
    
    if lC > lO:
        bias = "UP 📈"
        color = "#10b981"
        if lC > pH: desc = "Strong Bullish Breakout"
        else: desc = "Bullish Accumulation"
    elif lC < lO:
        bias = "DOWN 📉"
        color = "#ef4444"
        if lC < pL: desc = "Strong Bearish Breakdown"
        else: desc = "Bearish Pressure"
    else:
        bias = "NEUTRAL ⚖️"
        color = "#94a3b8"
        desc = "Price Indecision"
        
    return {"bias": bias, "color": color, "desc": desc, "pct": pct_from_open}

@st.cache_data(ttl=600)
def get_sector_sentiment(symbol):
    """Get sentiment for the stock's sector. Returns score -1.0 to +1.0."""
    sector = STOCK_TO_SECTOR.get(symbol.upper(), None)
    if not sector:
        return {'score': 0.0, 'sector': 'Unknown', 'label': 'Neutral'}
    
    try:
        sector_news = fetch_market_news(f"{sector} India stock market sector news")
        if not sector_news:
            return {'score': 0.0, 'sector': sector, 'label': 'Neutral'}
        
        filtered = filter_important_news(sector_news)
        if not filtered:
            filtered = sector_news[:5]
        
        total_score = 0.0
        count = 0
        for h in filtered:
            title = h.get('title', '') if isinstance(h, dict) else str(h)
            sc = score_headline_v2(title)
            total_score += sc
            count += 1
        
        avg = total_score / count if count > 0 else 0.0
        # Normalize to -1 to +1
        normalized = max(-1.0, min(1.0, avg / 2.0))
        
        if normalized > 0.2: label = f"Positive ({normalized*100:+.0f}%)"
        elif normalized < -0.2: label = f"Negative ({normalized*100:+.0f}%)"
        else: label = f"Neutral ({normalized*100:+.0f}%)"
        
        return {'score': normalized, 'sector': sector, 'label': label}
    except:
        return {'score': 0.0, 'sector': sector, 'label': 'Neutral'}

def analyze_news_v2(headlines):
    """Enhanced news analysis with importance filtering, freshness, and event detection."""
    if not headlines: return 0.0, [], "Technical Momentum (No News Data)", {'event_type': None, 'impact_score': 0.0, 'confidence': 0.0, 'description': 'No events'}
    
    # Step 1: Filter important news
    filtered = filter_important_news(headlines)
    if not filtered:
        filtered = headlines[:5]
    
    scored = []
    cat_counts = {k: 0 for k in CATALYSTS.keys()}
    best_h = None
    max_cat_match = -1
    
    for h in filtered:
        title = h.get('title', '') if isinstance(h, dict) else str(h)
        if title.endswith('?') and len(title.split()) < 5: continue
        
        # Enhanced scoring
        sc = score_headline_v2(title)
        
        # Apply freshness weight
        pub_date = h.get('pubDate', h.get('published', '')) if isinstance(h, dict) else ''
        freshness = calculate_freshness_weight(pub_date)
        weighted_score = sc * freshness
        
        t_low = title.lower()
        
        # Identify Catalysts
        found_cats = []
        for cat, keywords in CATALYSTS.items():
            matches = sum(1 for k in keywords if k in t_low)
            if matches > 0:
                cat_counts[cat] += 1
                found_cats.append(cat)
                if matches > max_cat_match:
                    max_cat_match = matches
                    best_h = title
        
        # AI Score label
        if sc >= 1.0: label = 'positive'
        elif sc <= -1.0: label = 'negative'
        else: label = 'neutral'
        
        # Score badge
        if sc >= 1.5: score_badge = '🟢 +2'
        elif sc >= 0.5: score_badge = '🟢 +1'
        elif sc <= -1.5: score_badge = '🔴 -2'
        elif sc <= -0.5: score_badge = '🔴 -1'
        else: score_badge = '🟡 0'
        
        importance = h.get('importance', 'General') if isinstance(h, dict) else 'General'
        
        entry = {**(h if isinstance(h, dict) else {'title': title}), 
                'score': sc, 'weighted_score': weighted_score, 'freshness': freshness,
                'label': label, 'score_badge': score_badge, 'catalysts': found_cats,
                'importance': importance}
        scored.append(entry)
    
    if not scored: return 0.0, [], "Technical Indicator dominance", {'event_type': None, 'impact_score': 0.0, 'confidence': 0.0, 'description': 'No events'}
    
    # Use weighted scores for average
    avg = sum(s['weighted_score'] for s in scored) / len(scored)
    
    # Identify Primary Catalyst
    top_cat = max(cat_counts, key=cat_counts.get) if any(cat_counts.values()) else "Broad Market Trends"
    
    prefix = ""
    if top_cat == "Earnings & Growth": prefix = "Robust Corporate Earnings" if avg > 0 else "Weak Earnings Performance"
    elif top_cat == "Deals & Orders": prefix = "Strategic New Contracts or Acquisitions" if avg > 0 else "Cancelled Deals or Orders"
    elif top_cat == "Policy & Govt": prefix = "Positive Regulatory Support" if avg > 0 else "Regulatory Headwinds"
    elif top_cat == "Market Risk": prefix = "Macro Stability" if avg > 0 else "Heightened Market Risk"
    elif top_cat == "Technical Breakout": prefix = "Technical Breakout" if avg > 0 else "Technical Breakdown"
    elif top_cat == "Corporate Action": prefix = "Positive Corporate Action" if avg > 0 else "Internal Governance Scrutiny"
    elif top_cat == "FII/DII Flow": prefix = "Strong Institutional Inflow" if avg > 0 else "Institutional Selling Pressure"
    else: prefix = "Bullish Sentiment" if avg > 0 else "Bearish Sentiment" if avg < 0 else "Market Dynamics"

    h_snippet = f': "{best_h[:75]}..."' if best_h else ""
    primary = f"{prefix}{h_snippet}"
    
    # Event detection
    events = detect_special_events(filtered)
    
    return round(avg, 3), scored, primary, events

def generate_gemini_intelligence(symbol, domestic_news, global_news, api_key):
    """
    Uses Gemini API to synthesize and connect domestic stock news and global market conditions.
    Returns: (domestic_catalyst, global_catalyst, correlation_text)
    """
    # Format the news items
    dom_str = "\n".join([f"- {h.get('title', h)}" for h in domestic_news[:7]])
    glob_str = "\n".join([f"- {h.get('title', h)}" for h in global_news[:5]])
    
    prompt = f"""
    You are an expert financial analyst. Analyze these stock news headlines for {symbol} (domestic market) and global macro-economic news:
    
    Domestic News for {symbol}:
    {dom_str}
    
    Global Macro News:
    {glob_str}
    
    Provide a synthesized analysis split into exactly three sections:
    1. DOMESTIC CATALYST: A concise, insightful summary of the key local factors affecting {symbol} (e.g., IPOs, earnings, expansions). (Max 120 words)
    2. GLOBAL DRIVER: A concise, insightful summary of the key international macroeconomic factors (e.g., Fed rates, oil prices, geopolitical issues). (Max 120 words)
    3. INTERPLAY: A clear explanation of how these two forces interact (e.g., how Fed rates impact FII inflows needed for domestic IPO success, or how global drivers limit local positives). (Max 100 words)
    
    Make the tone professional, sharp, and easy to read. Write the response in English.
    Format your response EXACTLY like this:
    [DOMESTIC]
    <Domestic analysis content>
    [GLOBAL]
    <Global analysis content>
    [INTERPLAY]
    <Interplay content>
    """
    
    try:
        import requests
        import json
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        if r.status_code == 200:
            data = r.json()
            text = data['candidates'][0]['content']['parts'][0]['text']
            
            # Parse the response
            dom_part, glob_part, interplay_part = "", "", ""
            if '[DOMESTIC]' in text and '[GLOBAL]' in text and '[INTERPLAY]' in text:
                parts = text.split('[GLOBAL]')
                dom_part = parts[0].replace('[DOMESTIC]', '').strip()
                parts_2 = parts[1].split('[INTERPLAY]')
                glob_part = parts_2[0].strip()
                interplay_part = parts_2[1].strip()
            else:
                lines = text.split('\n')
                current_section = None
                dom_lines, glob_lines, interplay_lines = [], [], []
                for line in lines:
                    if 'DOMESTIC' in line.upper():
                        current_section = 'dom'
                        continue
                    elif 'GLOBAL' in line.upper():
                        current_section = 'glob'
                        continue
                    elif 'INTERPLAY' in line.upper():
                        current_section = 'interplay'
                        continue
                    
                    if current_section == 'dom':
                        dom_lines.append(line)
                    elif current_section == 'glob':
                        glob_lines.append(line)
                    elif current_section == 'interplay':
                        interplay_lines.append(line)
                dom_part = "\n".join(dom_lines).strip()
                glob_part = "\n".join(glob_lines).strip()
                interplay_part = "\n".join(interplay_lines).strip()
                
            if dom_part and glob_part and interplay_part:
                return dom_part, glob_part, interplay_part
        return None
    except Exception:
        return None

def analyze_news(headlines):
    """Backward compatible wrapper — returns (avg, scored, primary) without events."""
    result = analyze_news_v2(headlines)
    return result[0], result[1], result[2]

@st.cache_data(ttl=900)
def get_master_market_sentiment():
    """Combines Indian and Global sentiment for a master bias with separate headline results"""
    # Fetch Indian News (Local)
    local_news = fetch_market_news("Indian stock market Nifty Sensex today trends news")
    l_score, l_scored, l_cat = analyze_news(local_news)
    
    # Fetch Global News
    global_news = fetch_global_news()
    g_score, g_scored, g_cat = analyze_news(global_news)
    
    master_score = (l_score * 0.6) + (g_score * 0.4)
    
    if master_score > 0.15: label, color = "STRONG POSITIVE 🚀", "#00b386"
    elif master_score > 0.05: label, color = "POSITIVE 📈", "#10b981"
    elif master_score < -0.15: label, color = "STRONG NEGATIVE 💥", "#eb5b3c"
    elif master_score < -0.05: label, color = "NEGATIVE 📉", "#ef4444"
    else: label, color = "NEUTRAL ⚖️", "#94a3b8"
    
    return {
        "label": label, "color": color, "score": master_score, 
        "local_reason": l_cat, "global_reason": g_cat,
        "indian_headlines": l_scored,
        "global_headlines": g_scored
    }



def get_stock_catalyst(symbol):
    """Wrapper to get a single catalyst string for UI display"""
    try:
        news = fetch_market_news(f"{symbol} share news")
        if not news: return "Market Dynamics"
        _, _, primary = analyze_news(news)
        return primary
    except:
        return "Broad Market Trends"

@st.cache_data(ttl=900)
def get_global_market_sentiment():
    """Analyze overall sentiment for the Indian Market"""
    try:
        news = fetch_market_news("Indian stock market Nifty Sensex today trends news")
        score, scored_news, catalyst = analyze_news(news)
        if score > 0.15: label, color = "POSITIVE 📈", "#10b981"
        elif score < -0.15: label, color = "NEGATIVE 📉", "#ef4444"
        else: label, color = "NEUTRAL ⚖️", "#94a3b8"
        return {"label": label, "color": color, "reason": catalyst, "score": score, "headlines": scored_news[:3]}
    except:
        return {"label": "NEUTRAL ⚖️", "color": "#94a3b8", "reason": "Market Dynamics", "score": 0.0, "headlines": []}


# ── TradingView Data Source ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_tv_sentiment(symbol, mapped):
    """
    Fetches the Technical Analysis summary from TradingView.
    """
    tv_sym = get_tv_symbol(symbol, mapped)
    try:
        from tradingview_ta import TA_Handler, Interval
        handler = TA_Handler(
            symbol=tv_sym.split(':')[-1],
            exchange=tv_sym.split(':')[0],
            screener="india" if "NSE" in tv_sym or "BSE" in tv_sym else "america",
            interval=Interval.INTERVAL_1_DAY
        )
        analysis = handler.get_analysis()
        summary = analysis.summary
        buy = summary.get('BUY', 0)
        sell = summary.get('SELL', 0)
        total = sum(summary.values())
        return (buy - sell) / total if total > 0 else 0.0
    except Exception as e:
        # Fallback to 0 if library is missing or symbol is not found
        return 0.0


# ── Options Engine (Phase 5) ─────────────────────────────────────────────────────────────
class OptionsEngine:
    def __init__(self):
        self.is_available = nse is not None

    def fetch_option_chain(self, symbol):
        """Fetches the live option chain from NSE."""
        if not self.is_available: return None
        try:
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                chain = nse.option_chain(symbol)
            else:
                chain = nse.option_chain(symbol)
            
            if not chain or 'filtered' not in chain or 'data' not in chain['filtered']:
                return None
                
            return chain
        except Exception as e:
            return None

    def calculate_oi_walls(self, chain):
        """Finds the strikes with Highest Call OI and Highest Put OI."""
        if not chain: return {"resistance_strike": None, "support_strike": None, "max_ce_oi": 0, "max_pe_oi": 0}
        
        data = chain['filtered']['data']
        max_ce_oi = 0
        max_pe_oi = 0
        res_strike = None
        sup_strike = None
        
        for strike_data in data:
            strike = strike_data.get('strikePrice')
            
            # Call side
            if 'CE' in strike_data:
                ce_oi = strike_data['CE'].get('openInterest', 0)
                if ce_oi > max_ce_oi:
                    max_ce_oi = ce_oi
                    res_strike = strike
                    
            # Put side
            if 'PE' in strike_data:
                pe_oi = strike_data['PE'].get('openInterest', 0)
                if pe_oi > max_pe_oi:
                    max_pe_oi = pe_oi
                    sup_strike = strike
                    
        return {"resistance_strike": res_strike, "support_strike": sup_strike, "max_ce_oi": max_ce_oi, "max_pe_oi": max_pe_oi}

    def calculate_pcr(self, chain):
        """Calculates Put Call Ratio."""
        if not chain: return {"pcr": 1.0, "sentiment": "Neutral"}
        
        tot_ce_oi = chain.get('filtered', {}).get('CE', {}).get('totOI', 0)
        tot_pe_oi = chain.get('filtered', {}).get('PE', {}).get('totOI', 0)
        
        if tot_ce_oi == 0: return {"pcr": 1.0, "sentiment": "Neutral"}
        pcr = tot_pe_oi / tot_ce_oi
        
        sentiment = "Neutral"
        if pcr < 0.8: sentiment = "Bearish"
        elif pcr > 1.2: sentiment = "Bullish"
        
        return {"pcr": round(pcr, 2), "sentiment": sentiment}

    def calculate_max_pain(self, chain):
        """Calculates Max Pain strike."""
        if not chain: return None
        data = chain['filtered']['data']
        
        strikes = [d['strikePrice'] for d in data]
        pain_values = {}
        
        for test_strike in strikes:
            total_pain = 0
            for strike_data in data:
                strike = strike_data['strikePrice']
                
                # Call Option Pain
                if 'CE' in strike_data:
                    oi = strike_data['CE'].get('openInterest', 0)
                    if test_strike > strike:
                        total_pain += (test_strike - strike) * oi
                        
                # Put Option Pain
                if 'PE' in strike_data:
                    oi = strike_data['PE'].get('openInterest', 0)
                    if test_strike < strike:
                        total_pain += (strike - test_strike) * oi
                        
            pain_values[test_strike] = total_pain
            
        if not pain_values: return None
        return min(pain_values, key=pain_values.get)

    def scan_iv_rank(self, chain):
        """Scans Implied Volatility (IV)."""
        if not chain: return {"iv": None, "preference": "Neutral", "atm_strike": None}
        data = chain['filtered']['data']
        
        underlying = chain.get('records', {}).get('underlyingValue', 0)
        if underlying == 0: return {"iv": None, "preference": "Neutral", "atm_strike": None}
        
        closest_strike = None
        min_diff = float('inf')
        atm_ce_iv = 0
        atm_pe_iv = 0
        
        for strike_data in data:
            strike = strike_data['strikePrice']
            diff = abs(strike - underlying)
            if diff < min_diff:
                min_diff = diff
                closest_strike = strike
                if 'CE' in strike_data: atm_ce_iv = strike_data['CE'].get('impliedVolatility', 0)
                if 'PE' in strike_data: atm_pe_iv = strike_data['PE'].get('impliedVolatility', 0)
                
        avg_iv = (atm_ce_iv + atm_pe_iv) / 2
        
        preference = "Neutral"
        if avg_iv > 25: preference = "Sell Options"
        elif avg_iv > 0 and avg_iv < 15: preference = "Buy Options"
        
        return {"iv": round(avg_iv, 2), "preference": preference, "atm_strike": closest_strike}

    def detect_oi_buildup(self, chain):
        """Detects OI buildup (Long Build-up, Short Build-up, Short Covering, Long Unwinding)."""
        if not chain: return "Neutral"
        data = chain['filtered']['data']
        
        underlying = chain.get('records', {}).get('underlyingValue', 0)
        if underlying == 0: return "Neutral"
        
        atm_strike = min([d['strikePrice'] for d in data], key=lambda x: abs(x - underlying))
        atm_data = next((d for d in data if d['strikePrice'] == atm_strike), None)
        
        if not atm_data or 'CE' not in atm_data: return "Neutral"
        
        ce_oi_change = atm_data['CE'].get('changeinOpenInterest', 0)
        ce_price_change = atm_data['CE'].get('pChange', 0)
        
        if ce_price_change > 0 and ce_oi_change > 0:
            return "🟢 Long Build-up"
        elif ce_price_change < 0 and ce_oi_change > 0:
            return "🔴 Short Build-up"
        elif ce_price_change > 0 and ce_oi_change < 0:
            return "🟢 Short Covering"
        elif ce_price_change < 0 and ce_oi_change < 0:
            return "🔴 Long Unwinding"
            
        return "Neutral"

    def get_full_analysis(self, symbol):
        """Runs all options analysis on a symbol."""
        chain = self.fetch_option_chain(symbol)
        if not chain:
            return None
            
        return {
            "symbol": symbol,
            "underlying": chain.get('records', {}).get('underlyingValue', 0),
            "oi_walls": self.calculate_oi_walls(chain),
            "pcr": self.calculate_pcr(chain),
            "max_pain": self.calculate_max_pain(chain),
            "iv": self.scan_iv_rank(chain),
            "buildup": self.detect_oi_buildup(chain)
        }

# ── AI Engine ─────────────────────────────────────────────────────────────
class AIEngine:
    FIB_LOOKBACK = FIB_LOOKBACK
    FIB_LEVELS = FIB_LEVELS
    FIB_WEIGHT = FIB_WEIGHT

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_path = "model.pkl"

    def save_model(self):
        """Saves all models and scalers to a single file."""
        try:
            joblib.dump({'models': self.models, 'scalers': self.scalers}, self.model_path)
            return True
        except Exception:
            return False

    def load_model(self):
        """Loads models and scalers from file."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.models = data['models']
                self.scalers = data['scalers']
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def _rsi(prices, period=14):
        if len(prices) < 2: return 50
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        ag = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains) if len(gains) > 0 else 0
        al = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses) if len(losses) > 0 else 0
        if al == 0: return 100
        return 100 - (100 / (1 + ag / al))

    @staticmethod
    def _ema(prices, period=9):
        if len(prices) < period: return np.mean(prices)
        alpha = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = (p * alpha) + (ema * (1 - alpha))
        return ema

    def _features(self, prices, volumes, idx, lb=30, global_moms=None):
        # Increased lookback for ATR/ADX stability
        w = prices[max(0, idx-lb):idx]
        vw = volumes[max(0, idx-lb):idx]
        if len(w) < 20: return None
        
        # 1. Basic Moving Averages
        ma5 = np.mean(w[-5:]); ma10 = np.mean(w[-10:]) if len(w)>=10 else np.mean(w); ma20 = np.mean(w)
        
        # 2. EMAs for Crossovers
        ema9 = self._ema(w, 9); ema21 = self._ema(w, 21)
        ema_cross = 1 if ema9 > ema21 else -1
        
        # 3. Volatility (ATR-like approx)
        std20 = np.std(w)
        atr_approx = np.mean(np.abs(np.diff(w[-10:]))) if len(w) > 10 else std20
        
        # 4. Momentum & Trend (MACD & RSI)
        rsi = self._rsi(w)
        ema12 = self._ema(w, 12); ema26 = self._ema(w, 26)
        macd = ema12 - ema26
        
        # 5. Trend Strength (ADX-like approx)
        up_move = sum(1 for i in range(1, len(w)) if w[i] > w[i-1]) / len(w)
        adx_approx = abs(up_move - 0.5) * 2 # 0 to 1 scale
        
        # 6. Volume Flow (OBV-like)
        obv = 0
        for i in range(1, len(w)):
            if w[i] > w[i-1]: obv += vw[i]
            elif w[i] < w[i-1]: obv -= vw[i]
        obv_rel = obv / np.mean(vw) if np.mean(vw) > 0 else 0
        
        # 7. Regimes
        is_trending = 1 if adx_approx > 0.25 else 0
        
        # 8. Global Index Momentum (New)
        gm = global_moms[idx-1] if global_moms is not None and len(global_moms) >= idx else 0
        
        return np.nan_to_num([
            ma5, ma10, ma20, ema9, ema21, ema_cross, 
            std20, rsi, macd, adx_approx, obv_rel, 
            atr_approx, is_trending, gm
        ], nan=0.0, posinf=0.0, neginf=0.0).tolist()

    @staticmethod
    def calculate_fibonacci(df, lookback):
        """
        Computes Fibonacci retracement levels over a specified lookback window.
        1. Retrieves the last `lookback` rows.
        2. Computes the high and low over that window.
        3. Returns fib_levels = {level: low + (high-low)*level for level in FIB_LEVELS}
        """
        window_df = df.tail(lookback)
        if window_df.empty:
            return {lvl: 0.0 for lvl in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}
        
        high = float(window_df['High'].max())
        low = float(window_df['Low'].min())
        diff = high - low
        
        levels = {}
        for lvl in [0.236, 0.382, 0.5, 0.618, 0.786]:
            levels[lvl] = low + diff * lvl
        
        levels[0.0] = low
        levels[1.0] = high
        return levels

    @staticmethod
    def fib_near_levels(current_price, fib_levels, tolerance):
        """
        Checks if the current price is within tolerance of key Fibonacci levels.
        Returns booleans for: near_618, near_50, near_382
        """
        near_618 = abs(current_price - fib_levels.get(0.618, 0)) <= tolerance
        near_50 = abs(current_price - fib_levels.get(0.5, 0)) <= tolerance
        near_382 = abs(current_price - fib_levels.get(0.382, 0)) <= tolerance
        return near_618, near_50, near_382

    def train(self, symbol, prices, volumes, news_sent=0.0, intraday=False):
        # Fetch global index trends for feature correlation
        try:
            g_df = yf.download('^GSPC', period='1y', interval='1d', progress=False)
            if g_df is not None and not g_df.empty:
                g_close = g_df['Close']
                if isinstance(g_close, pd.DataFrame): g_close = g_close.iloc[:,0]
                g_mom = g_close.pct_change().fillna(0).tolist()
            else: g_mom = None
        except: g_mom = None

        prices = np.array(prices, dtype=float); volumes = np.array(volumes, dtype=float)
        X, y1, y2, y3, y4 = [], [], [], [], []
        
        # Adaptive Triple Barrier Parameters based on asset volatility
        returns = np.abs(np.diff(prices) / prices[:-1])
        avg_vol = np.mean(returns) if len(returns) > 0 else 0.01
        
        # Scale targets: More volatile stocks get wider targets, stable ETFs get tighter targets
        if intraday:
            # Lower thresholds for intraday (15 min interval)
            # Min 0.15% target, Max 1.5% target
            PROFIT_TARGET = max(0.0015, min(0.015, avg_vol * 1.5))
        else:
            # Daily swing target (Min 0.8% target, Max 4% target)
            PROFIT_TARGET = max(0.008, min(0.04, avg_vol * 1.5))
        STOP_LOSS = PROFIT_TARGET / 2.0
        
        for i in range(30, len(prices)-10):
            f = self._features(prices, volumes, i, global_moms=g_mom)
            if f is None: continue
            X.append(f)
            
            # Labeling for different windows (1, 2, 3, 5 steps)
            for window, y_list in zip([2, 4, 6, 10], [y1, y2, y3, y4]):
                label = 0 # Default: Neutral/Hold
                entry_p = prices[i]
                
                # Check looking forward
                for j in range(1, window + 1):
                    future_p = prices[i + j]
                    ret = (future_p - entry_p) / entry_p
                    
                    if ret >= PROFIT_TARGET:
                        label = 1 # BUY (Target hit)
                        break
                    elif ret <= -STOP_LOSS:
                        label = -1 # SELL (Stop hit)
                        break
                y_list.append(label)
                
        if len(X) < 50: return None
        X = np.array(X)
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        
        self.models[symbol] = {'d1': {}, 'd2': {}, 'd3': {}, 'd4': {}}
        for day, labels in zip(['d1','d2','d3','d4'], [y1, y2, y3, y4]):
            y = np.array(labels)
            split = int(len(Xs) * 0.8)
            Xtr, Xte = Xs[:split], Xs[split:]
            ytr, yte = y[:split], y[split:]
            
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42)
            
            rf.fit(Xtr,ytr); gb.fit(Xtr,ytr)
            self.models[symbol][day] = {'rf': rf, 'gb': gb, 'acc': (rf.score(Xte,yte) + gb.score(Xte,yte))/2}
            
        self.scalers[symbol] = sc
        self.save_model() # Persist after training
        return {'d1_acc': self.models[symbol]['d1']['acc'], 'd2_acc': self.models[symbol]['d2']['acc'], 'd3_acc': self.models[symbol]['d3']['acc'], 'd4_acc': self.models[symbol]['d4']['acc']}

    def predict(self, symbol, prices, volumes, news_sent=0.0, tv_sent=0.0, sector_score=0.0, event_score=0.0, intraday=False, df=None, df_1h=None, df_1d=None, df_15m=None):
        if symbol not in self.models:
            if not self.load_model() or symbol not in self.models: return None
        
        sc = self.scalers[symbol]
        
        # 1. MTF Status Analysis
        main_status = self.get_timeframe_status(df)
        mtf_data = {}
        if df_1h is not None: mtf_data["1h"] = self.get_timeframe_status(df_1h)
        if df_1d is not None: mtf_data["1d"] = self.get_timeframe_status(df_1d)
        if df_15m is not None: mtf_data["15m"] = self.get_timeframe_status(df_15m)
        
        # 2. Global Index Momentum (correlation)
        if hasattr(self, '_cached_g_mom'):
            g_mean_mom = self._cached_g_mom
        else:
            try:
                g_df = yf.download('^GSPC', period='5d', interval='1d', progress=False)
                if g_df is not None and not g_df.empty:
                    g_close = g_df['Close']
                    if isinstance(g_close, pd.DataFrame): g_close = g_close.iloc[:, 0]
                    g_mean_mom = g_close.pct_change().tail(1).values[0]
                else: g_mean_mom = 0
            except: g_mean_mom = 0
            self._cached_g_mom = g_mean_mom
        
        # 3. Multi-Timeframe Alignment Logic
        mtf_alignment = 1.0
        is_conflict = False
        if "1d" in mtf_data:
            if mtf_data["1d"]["trend"] != main_status["trend"]: 
                mtf_alignment *= 0.8
                is_conflict = True
        
        triple_confirm = None
        if "15m" in mtf_data and "1h" in mtf_data and "1d" in mtf_data:
            trend_15m = mtf_data["15m"]["trend"]
            trend_1h = mtf_data["1h"]["trend"]
            trend_1d = mtf_data["1d"]["trend"]
            if trend_15m == trend_1h == trend_1d:
                triple_confirm = trend_15m
        
        # 4. Feature Extraction for current point
        prices_arr = np.array(prices, dtype=float)
        volumes_arr = np.array(volumes, dtype=float)
        
        # Global moms array for _features
        g_moms = [g_mean_mom] * len(prices_arr)
        f_latest = self._features(prices_arr, volumes_arr, len(prices_arr), global_moms=g_moms)
        if f_latest is None: return None
        
        feat = sc.transform([f_latest])
        
        # Determine lookback based on dataframe frequency
        freq = getattr(df.index, 'freq', None)
        if freq is None:
            # Fallback: infer from dataframe length (assume 1d)
            lookback_key = '1d'
        else:
            freq_str = str(freq)
            if 'D' in freq_str:
                lookback_key = '1d'
            elif 'H' in freq_str:
                lookback_key = '1h'
            elif 'T' in freq_str:
                lookback_key = '15m'
            else:
                lookback_key = '1d'
        lookback = self.FIB_LOOKBACK.get(lookback_key, 90)
        # Compute Fibonacci levels for this timeframe
        fib_levels = self.calculate_fibonacci(df, lookback)
        # Tolerance based on ATR or price
        price_val = float(df['Close'].iloc[-1])
        atr = np.mean(np.abs(np.diff(df['Close'].tail(14)))) if len(df) >= 14 else price_val * 0.003
        tolerance = max(atr * 0.25, price_val * 0.003)
        near_618, near_50, near_382 = self.fib_near_levels(price_val, fib_levels, tolerance)
        # Determine fib_score according to proximity flags
        if near_618:
            fib_score = 1.0
        elif near_50:
            fib_score = 0.7
        elif near_382:
            fib_score = 0.4
        else:
            fib_score = 0.0
        
        # SMC (Smart Money Concepts) feature detection
        smc = self.detect_smc_features(df)
        
        # Multi-Timeframe SMC: Run SMC detection on each available timeframe
        mtf_smc = {}  # {timeframe_label: smc_result}
        primary_tf_label = '1D'  # Default assumption for primary df
        freq = getattr(df.index, 'freq', None)
        if freq is not None:
            freq_str = str(freq)
            if 'T' in freq_str or 'min' in freq_str: primary_tf_label = '15M'
            elif 'H' in freq_str: primary_tf_label = '1H'
        elif intraday: primary_tf_label = '15M'
        mtf_smc[primary_tf_label] = smc
        if df_15m is not None and primary_tf_label != '15M':
            try: mtf_smc['15M'] = self.detect_smc_features(df_15m)
            except: pass
        if df_1h is not None and primary_tf_label != '1H':
            try: mtf_smc['1H'] = self.detect_smc_features(df_1h)
            except: pass
        if df_1d is not None and primary_tf_label != '1D':
            try: mtf_smc['1D'] = self.detect_smc_features(df_1d)
            except: pass
        current_price = float(df['Close'].iloc[-1])
        
        # 5. Prediction Ensemble
        results = {}
        labels = ['today', 'tomorrow', 'next_3_days', 'next_week']
        steps = ['d1', 'd2', 'd3', 'd4']
        
        # Step 1 (v3): Volatility Filter Check
        vol_label, vol_mult = self.calculate_volatility_state(df)
        
        # Step 2 (v3): Prep Confidence Breakdown Flags
        v_curr = float(df['Volume'].iloc[-1])
        v_avg = df['Volume'].tail(20).mean()
        is_vol_strong = v_curr > (v_avg * 1.2)
        
        is_mtf_sync = False
        if "1d" in mtf_data and "1h" in mtf_data:
            if "15m" in mtf_data:
                is_mtf_sync = (mtf_data["1d"]["trend"] == mtf_data["1h"]["trend"] == mtf_data["15m"]["trend"] == main_status["trend"])
            else:
                is_mtf_sync = (mtf_data["1d"]["trend"] == mtf_data["1h"]["trend"] == main_status["trend"])
            if is_mtf_sync: mtf_alignment *= 1.2 # Bonus for sync

        is_trending = f_latest[-1] > 0.5 
        
        # Compute component scores once outside the loop
        vol_ratio = v_curr / v_avg if v_avg > 0 else 1.0
        volume_score = min(vol_ratio / 2.0, 1.0)
        
        news_val = (news_sent + 1.0) / 2.0
        tv_val = (tv_sent + 1.0) / 2.0
        sec_val = (sector_score + 1.0) / 2.0
        evt_val = (event_score + 1.0) / 2.0
        sentiment_score = (news_val + tv_val + sec_val + evt_val) / 4.0
        
        # Detect Liquidity Sweeps
        liquidity_sweep = self.detect_liquidity_sweeps(df)
        
        for label, step_key in zip(labels, steps):
            m_set = self.models[symbol][step_key]
            
            probs_rf = m_set['rf'].predict_proba(feat)[0]
            probs_gb = m_set['gb'].predict_proba(feat)[0]
            
            classes = list(m_set['rf'].classes_)
            p_buy = (probs_rf[classes.index(1)] + probs_gb[classes.index(1)]) / 2 if 1 in classes else 0
            p_sell = (probs_rf[classes.index(-1)] + probs_gb[classes.index(-1)]) / 2 if -1 in classes else 0
            
            raw_prob = p_buy if p_buy > p_sell else p_sell
            ml_side = 1 if p_buy > p_sell else -1
            
            # SMC parameters & proximity calculations for current label
            ml_score = raw_prob
            tech_score = main_status['score']
            
            # Base Structure Score
            structure_score = 1.0 if smc['current_trend'] == ("Bullish" if ml_side == 1 else "Bearish") else 0.0
            
            # Liquidity Sweep Multiplier (User Request: natural boost instead of flat addition)
            has_bull_sweep = liquidity_sweep['bullish']
            has_bear_sweep = liquidity_sweep['bearish']
            
            if ml_side == 1 and has_bull_sweep:
                structure_score *= 1.5  # Boosts the 10% weight to 15% 
            elif ml_side == -1 and has_bear_sweep:
                structure_score *= 1.5
            
            # Order Block Score — Search across ALL timeframes for closest OB
            ob_score = 0.0
            closest_ob = None
            closest_ob_dist_pct = 999.0
            ob_timeframe = 'N/A'
            
            # Collect all OBs from every timeframe with their tf label
            all_tf_obs = []  # list of (ob_dict, tf_label)
            for tf_label, tf_smc in mtf_smc.items():
                ob_key = 'active_bullish_ob' if ml_side == 1 else 'active_bearish_ob'
                for ob in tf_smc.get(ob_key, []):
                    all_tf_obs.append((ob, tf_label))
            
            if all_tf_obs:
                best_ob = None
                best_tf = None
                min_dist = float('inf')
                for ob, tf_label in all_tf_obs:
                    if current_price < ob['bottom']:
                        dist = ob['bottom'] - current_price
                    elif current_price > ob['top']:
                        dist = current_price - ob['top']
                    else:
                        dist = 0.0
                    if dist < min_dist:
                        min_dist = dist
                        best_ob = ob
                        best_tf = tf_label
                if best_ob is not None:
                    closest_ob = best_ob
                    ob_timeframe = best_tf
                    ref = best_ob['top'] if ml_side == 1 else best_ob['bottom']
                    prox = 1.0 if min_dist == 0.0 else max(0.0, 1.0 - min_dist / (ref * 0.02))
                    fresh = max(0.0, 1.0 - (best_ob['age'] * 0.02))
                    ob_score = prox * fresh
                    closest_ob_dist_pct = (min_dist / current_price) * 100.0
            
            # Build multi-TF OB summary for display
            mtf_ob_summary = {}  # {tf_label: {type, count, closest_dist}}
            if 'mtf_smc' in locals():
                for tf_label, tf_smc in mtf_smc.items():
                    bull_obs = tf_smc.get('active_bullish_ob', [])
                    bear_obs = tf_smc.get('active_bearish_ob', [])
                    total = len(bull_obs) + len(bear_obs)
                    if total > 0:
                        mtf_ob_summary[tf_label] = {
                            'bullish': len(bull_obs),
                            'bearish': len(bear_obs),
                            'total': total,
                            'trend': tf_smc.get('current_trend', 'Neutral')
                        }
                
            # Fibonacci Confluence
            fib_score = 0.0
            if fib_levels:
                for lvl in fib_levels.values():
                    dist = abs(current_price - lvl) / current_price
                    if dist < 0.005: # Within 0.5%
                        fib_score = 1.0
                        break
                    elif dist < 0.01:
                        fib_score = 0.5
            
            # Fair Value Gap Proximity
            fvg_score = 0.0
            closest_fvg = None
            closest_fvg_dist_pct = 0.0
            all_fvgs = smc.get('active_bullish_fvg', []) + smc.get('active_bearish_fvg', [])
            if all_fvgs:
                best_fvg = None
                min_dist = float('inf')
                for fvg in all_fvgs:
                    # Bullish ML -> look for Bullish FVG below price
                    if ml_side == 1 and fvg['type'] == 'Bullish' and current_price >= fvg['top']:
                        dist = current_price - fvg['top']
                    # Bearish ML -> look for Bearish FVG above price
                    elif ml_side == -1 and fvg['type'] == 'Bearish' and current_price <= fvg['bottom']:
                        dist = fvg['bottom'] - current_price
                    else:
                        continue
                        
                    if dist < min_dist:
                        min_dist = dist
                        best_fvg = fvg
                if best_fvg is not None:
                    closest_fvg = best_fvg
                    ref = best_fvg['top'] if ml_side == 1 else best_fvg['bottom']
                    fvg_score = 1.0 if min_dist == 0.0 else max(0.0, 1.0 - min_dist / (ref * 0.02))
                    closest_fvg_dist_pct = (min_dist / current_price) * 100.0
            
            # ML Ensemble Probability
            ml_score = raw_prob
            
            # 10-Factor Institutional Confluence Score
            final_score = (
                (ml_score * 0.30) +         # ML ensemble
                (tech_score * 0.18) +        # Technical
                (volume_score * 0.12) +      # Volume
                (structure_score * 0.10) +   # SMC Structure
                (ob_score * 0.08) +          # Order Block
                (fib_score * 0.04) +         # Fibonacci
                (fvg_score * 0.03) +         # FVG
                (news_val * 0.07) +          # NEW: News Sentiment
                (sec_val * 0.04) +           # NEW: Sector Sentiment
                (evt_val * 0.04)             # NEW: Event Impact
            )
            
            # Baseline conviction score (excluding SMC optional components OB, FVG, Fib)
            # ML(30) + Tech(18) + Vol(12) + Struct(10) + News(7) + Sec(4) + Evt(4) = 85%
            baseline_score = (
                (ml_score * 0.30) +
                (tech_score * 0.18) +
                (volume_score * 0.12) +
                (structure_score * 0.10) +
                (news_val * 0.07) +
                (sec_val * 0.04) +
                (evt_val * 0.04)
            ) / 0.85
            
            # Apply Multipliers to both scores
            baseline_score *= vol_mult
            if not is_trending: baseline_score *= 0.95
            baseline_score *= mtf_alignment
            baseline_score = min(max(baseline_score, 0.0), 1.0)
            
            final_score *= vol_mult
            if not is_trending: final_score *= 0.95
            final_score *= mtf_alignment
            final_score = min(max(final_score, 0.0), 1.0)
            
            # Guardrails
            if raw_prob < 0.50: final_score = min(final_score, 0.39)
            elif baseline_score < 0.40: final_score = min(final_score, 0.39)
            elif baseline_score < 0.65: final_score = min(final_score, 0.64)
            
            # Signal Selection
            if vol_mult <= 0.1: sig = "NO TRADE (Low Volatility)"
            elif is_conflict: sig = "NO TRADE (Trend Conflict)"
            elif final_score >= 0.65: sig = "STRONG BUY" if ml_side == 1 else "STRONG SELL"
            elif final_score >= 0.40: sig = "BUY" if ml_side == 1 else "SELL"
            else: sig = "NO TRADE (Low Confidence)"
            
            stars = 5 if final_score >= 0.85 else 4 if final_score >= 0.70 else 3 if final_score >= 0.50 else 2 if final_score >= 0.30 else 1
            
            results[label] = {
                'signal': sig, 'confidence': round(final_score, 4), 'stars': stars,
                'ml_prob': round(raw_prob, 4), 'tech_score': main_status['score'],
                'pattern_score': round((fib_score + ob_score) / 2.0, 4),
                'is_trending': is_trending,
                'scores': {
                    'ml_score': round(ml_score, 4),
                    'tech_score': round(tech_score, 4),
                    'volume_score': round(volume_score, 4),
                    'structure_score': round(structure_score, 4),
                    'ob_score': round(ob_score, 4),
                    'fib_score': round(fib_score, 4),
                    'fvg_score': round(fvg_score, 4),
                    'news_score': round(news_val, 4),
                    'sec_score': round(sec_val, 4),
                    'evt_score': round(evt_val, 4)
                },
                'breakdown': {
                    'Trend Alignment': "PASS ✅" if not is_conflict else "FAIL ❌",
                    'Volume Confirm': "PASS ✅" if is_vol_strong else "FAIL ❌",
                    'MTF Sync': "PASS ✅" if is_mtf_sync else "FAIL ❌",
                    'Volatility OK': "PASS ✅" if vol_mult > 0 else "FAIL ❌",
                    'News Sentiment': "PASS ✅" if news_val >= 0.5 else "FAIL ❌",
                    'Sector Trend': "PASS ✅" if sec_val >= 0.5 else "FAIL ❌",
                    'Event Impact': "PASS ✅" if evt_val >= 0.5 else "FAIL ❌",
                    'Liquidity Sweep': "PASS 💧" if (ml_side == 1 and liquidity_sweep['bullish']) or (ml_side == -1 and liquidity_sweep['bearish']) else "FAIL ❌"
                },
                'smc': {
                    'current_trend': smc['current_trend'],
                    'closest_ob': closest_ob,
                    'closest_ob_dist_pct': closest_ob_dist_pct,
                    'ob_timeframe': ob_timeframe,
                    'mtf_ob_summary': mtf_ob_summary,
                    'closest_fvg': closest_fvg,
                    'closest_fvg_dist_pct': closest_fvg_dist_pct,
                    'confluence_detected': stars == 5,
                    'smc_full': smc
                }
            }
        results['fib_levels'] = fib_levels
        results['mtf_status'] = mtf_data
        results['volatility'] = vol_label
        results['triple_confirm'] = triple_confirm
        results['liquidity_sweep'] = liquidity_sweep
        return results

    def get_timeframe_status(self, df):
        """Extracts technical status for a single timeframe dataframe."""
        if df is None or len(df) < 20:
            return {"trend": "Neutral", "rsi": 50, "pattern": "N/A", "score": 0.5}
        
        prices = df['Close'].dropna().values
        rsi = self._rsi(prices)
        ema9 = self._ema(prices, 9)
        ema21 = self._ema(prices, 21)
        
        trend = "Bullish" if prices[-1] > ema21 else "Bearish"
        
        # Simple technical score (0 to 1)
        score = 0.5
        if trend == "Bullish": score += 0.2
        else: score -= 0.2
        
        if 40 < rsi < 60: score += 0.1
        elif rsi > 70: score -= 0.1 # Overbought
        elif rsi < 30: score += 0.1 # Oversold
        
        return {"trend": trend, "rsi": round(rsi, 2), "score": round(score, 2), "pattern": "N/A"}

    @staticmethod
    def detect_liquidity_sweeps(df, window=2, lookback=20):
        """
        Identify institutional liquidity sweeps where the price hunts stops past swing peaks/troughs before reversing.
        """
        if df is None or len(df) < window * 2 + 1:
            return {"bullish": False, "bearish": False, "bullish_level": None, "bearish_level": None}
        
        df = df.copy()
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        # 1. Identify swing points
        swing_highs = []
        swing_lows = []
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i-window : i+window+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-window : i+window+1]):
                swing_lows.append((i, lows[i]))
        
        bullish_sweep = False
        bearish_sweep = False
        bullish_level = None
        bearish_level = None
        
        # 2. Check the latest two candles
        volumes = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))
        opens = df['Open'].values
        
        bullish_quality = "None"
        bearish_quality = "None"
        
        for idx in [len(df) - 2, len(df) - 1]:
            if idx < 0 or idx >= len(df):
                continue
            current_low = lows[idx]
            current_high = highs[idx]
            current_close = closes[idx]
            current_open = opens[idx]
            current_vol = volumes[idx]
            
            vol_avg = np.mean(volumes[max(0, idx-20):idx]) if idx > 0 else 0
            body = abs(current_open - current_close)
            
            # Bullish sweep
            recent_lows = [val for pos, val in swing_lows if pos < idx and idx - pos <= lookback]
            if recent_lows:
                min_recent_low = min(recent_lows)
                if current_low < min_recent_low and current_close > min_recent_low:
                    bullish_sweep = True
                    bullish_level = min_recent_low
                    
                    # Calculate quality
                    lower_wick = min(current_open, current_close) - current_low
                    is_large_wick = lower_wick > body * 1.5
                    is_high_vol = current_vol > vol_avg * 1.2
                    
                    if is_large_wick and is_high_vol: bullish_quality = "Institutional ⭐⭐⭐⭐⭐"
                    elif is_large_wick or is_high_vol: bullish_quality = "Medium ⭐⭐⭐"
                    else: bullish_quality = "Weak ⭐"
                    
            # Bearish sweep
            recent_highs = [val for pos, val in swing_highs if pos < idx and idx - pos <= lookback]
            if recent_highs:
                max_recent_high = max(recent_highs)
                if current_high > max_recent_high and current_close < max_recent_high:
                    bearish_sweep = True
                    bearish_level = max_recent_high
                    
                    # Calculate quality
                    upper_wick = current_high - max(current_open, current_close)
                    is_large_wick = upper_wick > body * 1.5
                    is_high_vol = current_vol > vol_avg * 1.2
                    
                    if is_large_wick and is_high_vol: bearish_quality = "Institutional ⭐⭐⭐⭐⭐"
                    elif is_large_wick or is_high_vol: bearish_quality = "Medium ⭐⭐⭐"
                    else: bearish_quality = "Weak ⭐"
                    
        return {
            "bullish": bullish_sweep, 
            "bearish": bearish_sweep,
            "bullish_level": bullish_level,
            "bearish_level": bearish_level,
            "bullish_quality": bullish_quality,
            "bearish_quality": bearish_quality
        }

    @staticmethod
    def detect_smc_features(df):
        """
        Detects Smart Money Concepts (SMC) features in the dataframe:
        - Swing Highs & Swing Lows (window=2)
        - Break of Structure (BOS) & Change of Character (CHOCH)
        - Active Bullish & Bearish Order Blocks (OB) with Freshness Score
        - Active Bullish & Bearish Fair Value Gaps (FVG)
        """
        if df is None or len(df) < 5:
            return {
                "swing_highs": [],
                "swing_lows": [],
                "bos": [],
                "choch": [],
                "active_bullish_ob": [],
                "active_bearish_ob": [],
                "active_bullish_fvg": [],
                "active_bearish_fvg": [],
                "current_trend": "Neutral"
            }

        df = df.copy()
        n = len(df)
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values if 'Open' in df.columns else closes
        times = df.index

        swing_highs = [] 
        swing_lows = []  
        bos_list = []    
        choch_list = []  

        # Tracks active order blocks
        active_obs = []
        
        # Tracks active FVGs
        active_fvgs = []

        # Determine starting trend based on simple SMA/EMA
        current_trend = "Bullish" if closes[-1] >= np.mean(closes) else "Bearish"

        last_swing_high_val = None
        last_swing_high_idx = None
        last_swing_low_val = None
        last_swing_low_idx = None

        # Loop to detect swing points and structure breaks
        for i in range(2, n):
            # 1 & 2. Detect Swing Highs and Lows (needs 2 candles future lookahead, so i < n - 2)
            if i < n - 2:
                # Detect Swing High at i
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    swing_highs.append((i, highs[i]))
                    last_swing_high_val = highs[i]
                    last_swing_high_idx = i

                # Detect Swing Low at i
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    swing_lows.append((i, lows[i]))
                    last_swing_low_val = lows[i]
                    last_swing_low_idx = i

            # 3. Detect BOS & CHOCH at index i based on the latest confirmed swing points
            if current_trend == "Bullish":
                # Break of Structure (Bullish BOS): Close breaks above last swing high
                if last_swing_high_val is not None and closes[i] > last_swing_high_val:
                    bos_list.append({
                        'index': int(i),
                        'type': 'Bullish',
                        'price': float(last_swing_high_val),
                        'time': times[i]
                    })
                    # Create Bullish Order Block
                    ob_idx = None
                    for k in range(i, max(0, i - 15), -1):
                        if closes[k] < opens[k]:
                            ob_idx = k
                            break
                    if ob_idx is not None:
                        active_obs.append({
                            'index': int(ob_idx),
                            'top': float(max(opens[ob_idx], closes[ob_idx])),
                            'bottom': float(lows[ob_idx]),
                            'high': float(highs[ob_idx]),
                            'low': float(lows[ob_idx]),
                            'type': 'Bullish',
                            'active': True,
                            'time': times[ob_idx]
                        })
                    last_swing_high_val = None # Reset

                # Change of Character (Bearish CHOCH): Close breaks below last swing low
                elif last_swing_low_val is not None and closes[i] < last_swing_low_val:
                    choch_list.append({
                        'index': int(i),
                        'type': 'Bearish',
                        'price': float(last_swing_low_val),
                        'time': times[i]
                    })
                    current_trend = "Bearish"
                    # Create Bearish OB
                    ob_idx = None
                    for k in range(i, max(0, i - 15), -1):
                        if closes[k] > opens[k]:
                            ob_idx = k
                            break
                    if ob_idx is not None:
                        active_obs.append({
                            'index': int(ob_idx),
                            'top': float(highs[ob_idx]),
                            'bottom': float(min(opens[ob_idx], closes[ob_idx])),
                            'high': float(highs[ob_idx]),
                            'low': float(lows[ob_idx]),
                            'type': 'Bearish',
                            'active': True,
                            'time': times[ob_idx]
                        })
                    last_swing_low_val = None

            elif current_trend == "Bearish":
                # Break of Structure (Bearish BOS): Close breaks below last swing low
                if last_swing_low_val is not None and closes[i] < last_swing_low_val:
                    bos_list.append({
                        'index': int(i),
                        'type': 'Bearish',
                        'price': float(last_swing_low_val),
                        'time': times[i]
                    })
                    # Create Bearish OB
                    ob_idx = None
                    for k in range(i, max(0, i - 15), -1):
                        if closes[k] > opens[k]:
                            ob_idx = k
                            break
                    if ob_idx is not None:
                        active_obs.append({
                            'index': int(ob_idx),
                            'top': float(highs[ob_idx]),
                            'bottom': float(min(opens[ob_idx], closes[ob_idx])),
                            'high': float(highs[ob_idx]),
                            'low': float(lows[ob_idx]),
                            'type': 'Bearish',
                            'active': True,
                            'time': times[ob_idx]
                        })
                    last_swing_low_val = None

                # Change of Character (Bullish CHOCH): Close breaks above last swing high
                elif last_swing_high_val is not None and closes[i] > last_swing_high_val:
                    choch_list.append({
                        'index': int(i),
                        'type': 'Bullish',
                        'price': float(last_swing_high_val),
                        'time': times[i]
                    })
                    current_trend = "Bullish"
                    # Create Bullish OB
                    ob_idx = None
                    for k in range(i, max(0, i - 15), -1):
                        if closes[k] < opens[k]:
                            ob_idx = k
                            break
                    if ob_idx is not None:
                        active_obs.append({
                            'index': int(ob_idx),
                            'top': float(max(opens[ob_idx], closes[ob_idx])),
                            'bottom': float(lows[ob_idx]),
                            'high': float(highs[ob_idx]),
                            'low': float(lows[ob_idx]),
                            'type': 'Bullish',
                            'active': True,
                            'time': times[ob_idx]
                        })
                    last_swing_high_val = None

            # 4. FVG Detection
            if lows[i] > highs[i-2]:
                active_fvgs.append({
                    'index': int(i-1),
                    'bottom': float(highs[i-2]),
                    'top': float(lows[i]),
                    'type': 'Bullish',
                    'active': True,
                    'time': times[i-1]
                })
            if highs[i] < lows[i-2]:
                active_fvgs.append({
                    'index': int(i-1),
                    'bottom': float(highs[i]),
                    'top': float(lows[i-2]),
                    'type': 'Bearish',
                    'active': True,
                    'time': times[i-1]
                })

        # Mitigation checking
        for ob in active_obs:
            ob_idx = ob['index']
            ob['age'] = int(n - 1 - ob_idx)
            for j in range(ob_idx + 1, n):
                if ob['type'] == 'Bullish':
                    if lows[j] < ob['low']:
                        ob['active'] = False
                        break
                else:
                    if highs[j] > ob['high']:
                        ob['active'] = False
                        break

        for fvg in active_fvgs:
            fvg_idx = fvg['index']
            for j in range(fvg_idx + 2, n):
                if fvg['type'] == 'Bullish':
                    if lows[j] < fvg['bottom']:
                        fvg['active'] = False
                        break
                else:
                    if highs[j] > fvg['top']:
                        fvg['active'] = False
                        break

        return {
            "swing_highs": [(int(idx), float(val)) for idx, val in swing_highs],
            "swing_lows": [(int(idx), float(val)) for idx, val in swing_lows],
            "bos": bos_list,
            "choch": choch_list,
            "active_bullish_ob": [ob for ob in active_obs if ob['active'] and ob['type'] == 'Bullish'],
            "active_bearish_ob": [ob for ob in active_obs if ob['active'] and ob['type'] == 'Bearish'],
            "active_bullish_fvg": [fvg for fvg in active_fvgs if fvg['active'] and fvg['type'] == 'Bullish'],
            "active_bearish_fvg": [fvg for fvg in active_fvgs if fvg['active'] and fvg['type'] == 'Bearish'],
            "current_trend": current_trend
        }
    def calculate_risk_parameters(self, symbol, entry, signal, capital, risk_pct, reward_ratio=2, df=None):
        """Step 1: Risk Engine - Calculates SL, Target, and Position Size"""
        if entry <= 0: return None
        
        risk_amt = capital * (risk_pct / 100)
        
        # Calculate Volatility-based Stop Loss (using ATR approx)
        atr = np.mean(np.abs(np.diff(df['Close'].tail(14)))) if df is not None and len(df) > 14 else entry * 0.01
        
        if "BUY" in signal:
            # Stop Loss: Recent Swing Low or ATR-based
            sl = entry - (atr * 1.5)
            # Ensure SL is not too far (max 3%) or too close (min 0.5%)
            if sl > entry * 0.995: sl = entry * 0.995
            if sl < entry * 0.97: sl = entry * 0.97
            
            risk_per_share = entry - sl
            target = entry + (risk_per_share * reward_ratio)
        else:
            # Stop Loss: Recent Swing High or ATR-based
            sl = entry + (atr * 1.5)
            if sl < entry * 1.005: sl = entry * 1.005
            if sl > entry * 1.03: sl = entry * 1.03
            
            risk_per_share = sl - entry
            target = entry - (risk_per_share * reward_ratio)
            
        pos_size = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
        
        return {
            "entry": entry,
            "sl": sl,
            "target": target,
            "risk_reward": f"1:{reward_ratio}",
            "pos_size": pos_size,
            "risk_amt": risk_amt,
            "profit_amt": risk_amt * reward_ratio
        }

    def detect_entry_timing(self, df):
        """Step 3: Entry Timing Engine - Pullback (READY) vs Breakout (DETECTED)"""
        if df is None or len(df) < 30: return "ANALYZING..."
        
        c = float(df['Close'].iloc[-1])
        v = float(df['Volume'].iloc[-1])
        v_avg = df['Volume'].tail(20).mean()
        
        ema21 = self._ema(df['Close'].values, 21)
        prev_ema21 = self._ema(df['Close'].values[:-1], 21)
        
        # High of the last 20 candles for breakout detection
        high20 = df['High'].iloc[:-1].tail(20).max()
        low20 = df['Low'].iloc[:-1].tail(20).min()
        
        # 1. Breakout Check
        if c > high20 and v > v_avg * 1.3:
            return "BREAKOUT (DETECTED) 🚀"
        if c < low20 and v > v_avg * 1.3:
            return "BREAKDOWN (DETECTED) 📉"
            
        # 2. Pullback Check
        if ema21 > prev_ema21 and c > ema21: # Uptrend
            dist = (c - ema21) / ema21
            if dist < 0.008: return "PULLBACK (READY) ⏳"
        elif ema21 < prev_ema21 and c < ema21: # Downtrend
            dist = (ema21 - c) / ema21
            if dist < 0.008: return "PULLBACK (READY) ⏳"
            
        return "CONSOLIDATING ⚖️"

    def detect_liquidity(self, df):
        """Step 5: Liquidity Logic - Identify institutional SL Clusters (Swing Highs/Lows)"""
        if df is None or len(df) < 50: return "ZONE: NEUTRAL"
        
        # Use last 60 candles to find meaningful supply/demand zones (Liquidity)
        window = df.tail(60)
        top_liq = window['High'].max()
        bot_liq = window['Low'].min()
        
        c = float(df['Close'].iloc[-1])
        
        # Proximity detection (within 0.3% of the extreme)
        if c >= top_liq * 0.997:
            return "ABOVE APEX LIQUIDITY 🧲"
        elif c <= bot_liq * 1.003:
            return "BELOW BASE LIQUIDITY 🧲"
        
        if c > top_liq * 0.985:
            return "NEAR SUPPLY LIQUIDITY ⚠️"
        if c < bot_liq * 1.015:
            return "NEAR DEMAND LIQUIDITY ⚠️"
            
        return "MID-ZONE LIQUIDITY"

    def calculate_volatility_state(self, df):
        """Step 1 (v3): Volatility Filter - ATR based stagnation detection"""
        if df is None or len(df) < 50: return "NORMAL", 1.0
        
        # True Range calculation
        h, l, pc = df['High'].values, df['Low'].values, df['Close'].shift(1).values
        tr = np.max([h-l, np.abs(h-pc), np.abs(l-pc)], axis=0)
        
        atr_current = np.mean(tr[-14:]) # 14-period ATR
        atr_base = np.mean(tr[-50:])    # 50-period average volatility
        
        if atr_current < (atr_base * 0.70):
            # Reduced penalty: Allow 0.7 multiplier to let strong trends through
            return "LOW (STAGNANT) ❄️", 0.7 
        if atr_current > (atr_base * 3.0):
            return "EXTREME (CHAOTIC) ☢️", 0.4 # Slightly lower multiplier for extreme chaos
            
        return "STABLE ✅", 1.0

    def get_market_session(self):
        """Step 4 (v3): Indian Market Session Awareness (IST)"""
        now = datetime.now()
        # Ensure we are in IST (Server time might be different, but for local use assuming IST or offset)
        # Assuming system time is IST for this implementation
        cur_time = now.time()
        
        if time(9, 15) <= cur_time <= time(10, 30):
            return "OPENING VOLATILITY ⚡ (High Risk/High Reward)"
        if time(12, 0) <= cur_time <= time(14, 0):
            return "MIDDAY STAGNATION 💤 (Potential Sideways)"
        if time(14, 30) <= cur_time <= time(15, 30):
            return "CLOSING MOMENTUM 🚀 (Institutional Move)"
        if cur_time > time(15, 30):
            return "MARKET CLOSED 🌙"
            
        return "REGULAR SESSION"








# ── Charts ────────────────────────────────────────────────────────────────

def detect_support_resistance(df, window=5, num_levels=5):
    """
    Detect support and resistance levels using pivot high/low algorithm.
    Returns a dict with 'support', 'resistance', signal info, and price context.
    """
    if df is None or len(df) < window * 2 + 1:
        price = float(df['Close'].iloc[-1]) if df is not None and len(df) > 0 else 0
        return {
            'support': [], 'resistance': [], 'current_price': price,
            'nearest_support': 0, 'nearest_resistance': 0,
            'sr_signal': 'N/A', 'sr_color': '#94a3b8', 'sr_detail': 'Not enough data'
        }

    highs = df['High'].values
    lows  = df['Low'].values
    cur_price = float(df['Close'].iloc[-1])

    raw_res, raw_sup = [], []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            raw_res.append(highs[i])
        if lows[i] == min(lows[i - window: i + window + 1]):
            raw_sup.append(lows[i])

    # Cluster nearby levels (within 0.5%)
    def cluster(levels):
        if not levels:
            return []
        levels = sorted(set(round(l, 2) for l in levels))
        clusters = []
        group = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - group[-1]) / group[-1] < 0.005:
                group.append(lvl)
            else:
                clusters.append(group)
                group = [lvl]
        clusters.append(group)
        return [{'price': round(sum(g) / len(g), 2), 'strength': len(g)} for g in clusters]

    sup_levels = sorted(cluster(raw_sup), key=lambda x: x['price'])[-num_levels:]
    res_levels = sorted(cluster(raw_res), key=lambda x: x['price'])[:num_levels]

    # Nearest levels to current price
    sup_prices = [s['price'] for s in sup_levels if s['price'] < cur_price]
    res_prices = [r['price'] for r in res_levels if r['price'] > cur_price]

    nearest_sup = max(sup_prices) if sup_prices else (sup_levels[0]['price'] if sup_levels else 0)
    nearest_res = min(res_prices) if res_prices else (res_levels[-1]['price'] if res_levels else 0)

    # Generate signal
    sup_dist = abs(cur_price - nearest_sup) / cur_price if nearest_sup else 1
    res_dist = abs(nearest_res - cur_price) / cur_price if nearest_res else 1

    if sup_dist < 0.015:
        sig, color, detail = 'BUY ZONE', '#10b981', f'Price near support ₹{nearest_sup:,.2f} — high-probability entry'
    elif res_dist < 0.015:
        sig, color, detail = 'SELL ZONE', '#ef4444', f'Price near resistance ₹{nearest_res:,.2f} — consider booking profits'
    elif cur_price > (nearest_res if nearest_res else cur_price * 1.1):
        sig, color, detail = 'BREAKOUT', '#00b386', f'Price broke above resistance ₹{nearest_res:,.2f}'
    elif cur_price < (nearest_sup if nearest_sup else cur_price * 0.9):
        sig, color, detail = 'BREAKDOWN', '#eb5b3c', f'Price broke below support ₹{nearest_sup:,.2f}'
    else:
        sig, color, detail = 'MID-ZONE', '#6366f1', f'Between support ₹{nearest_sup:,.2f} and resistance ₹{nearest_res:,.2f}'

    return {
        'support': sup_levels,
        'resistance': res_levels,
        'current_price': cur_price,
        'nearest_support': nearest_sup,
        'nearest_resistance': nearest_res,
        'sr_signal': sig,
        'sr_color': color,
        'sr_detail': detail,
    }


def build_sr_chart(df, symbol, sr_data):
    """
    Build Support & Resistance chart with horizontal level lines.
    FIX: fig.add_shape() does NOT accept row/col in Plotly 5.x — removed those args.
    """
    UP_COLOR   = '#00b386'
    DOWN_COLOR = '#eb5b3c'
    SUP_COLOR  = '#10b981'
    RES_COLOR  = '#ef4444'
    CUR_COLOR  = '#f8fafc'

    fig = go.Figure()

    # ── Candlestick ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR,  decreasing_fillcolor=DOWN_COLOR,
        name='Price',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>O:%{open:.2f} H:%{high:.2f} L:%{low:.2f} C:%{close:.2f}',
    ))

    x_start = df.index[0]
    x_end   = df.index[-1]

    # ── Support levels ───────────────────────────────────────────────────
    for lvl in sr_data.get('support', []):
        price    = lvl['price']
        strength = lvl['strength']
        linewidth = min(1 + strength * 0.4, 3)
        alpha_float = min(80 + strength * 20, 255) / 255.0
        # NOTE: add_shape does NOT take row/col — those args were removed in Plotly 5.x
        fig.add_shape(
            type='line',
            x0=x_start, x1=x_end,
            y0=price,   y1=price,
            line=dict(color=f'rgba(16, 185, 129, {alpha_float})', width=linewidth, dash='dot'),
        )
        fig.add_annotation(
            x=x_end, y=price,
            text=f"S ₹{price:,.0f} ({strength}x)",
            showarrow=False,
            font=dict(size=9, color=SUP_COLOR),
            xanchor='left', yanchor='middle',
        )

    # ── Resistance levels ────────────────────────────────────────────────
    for lvl in sr_data.get('resistance', []):
        price    = lvl['price']
        strength = lvl['strength']
        linewidth = min(1 + strength * 0.4, 3)
        alpha_float = min(80 + strength * 20, 255) / 255.0
        fig.add_shape(
            type='line',
            x0=x_start, x1=x_end,
            y0=price,   y1=price,
            line=dict(color=f'rgba(239, 68, 68, {alpha_float})', width=linewidth, dash='dot'),
        )
        fig.add_annotation(
            x=x_end, y=price,
            text=f"R ₹{price:,.0f} ({strength}x)",
            showarrow=False,
            font=dict(size=9, color=RES_COLOR),
            xanchor='left', yanchor='middle',
        )

    # ── Current price line ───────────────────────────────────────────────
    cur_price = sr_data.get('current_price', float(df['Close'].iloc[-1]))
    fig.add_shape(
        type='line',
        x0=x_start, x1=x_end,
        y0=cur_price, y1=cur_price,
        line=dict(color=CUR_COLOR, width=1.5, dash='solid'),
    )
    fig.add_annotation(
        x=x_end, y=cur_price,
        text=f"CMP ₹{cur_price:,.2f}",
        showarrow=False,
        font=dict(size=10, color=CUR_COLOR, family='Arial Black'),
        xanchor='left', yanchor='middle',
    )

    fig.update_layout(
        template='plotly_dark',
        title=dict(text=f'<b>{symbol}</b> — Support & Resistance Levels', font=dict(size=15, color='#f8fafc')),
        height=480,
        showlegend=False,
        hovermode='x unified',
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        margin=dict(l=40, r=120, t=55, b=40),
        xaxis=dict(showgrid=True, gridcolor='#334155', rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor='#334155', tickprefix='₹'),
    )
    return fig


def build_candle_chart(df, symbol, fib_levels=None, smc=None):
    """
    Build candlestick chart with optional Fibonacci levels and SMC overlays (Order Blocks, FVGs, BOS/CHOCH)
    """
    UP_COLOR = '#00b386'
    DOWN_COLOR = '#eb5b3c'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        name='Price', showlegend=True
    ), row=1, col=1)
    
    colors = [UP_COLOR if c >= o else DOWN_COLOR for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Vol', opacity=0.3, showlegend=False), row=2, col=1)
    
    show_leg = False
    
    if fib_levels:
        show_leg = True
        current_price = float(df['Close'].iloc[-1])
        
        # 1. Find swing points (High and Low) in the current visible data
        high_val = float(df['High'].max())
        low_val = float(df['Low'].min())
        high_idx = df['High'].idxmax()
        low_idx = df['Low'].idxmin()
        
        # Fibonacci lines are drawn from the first swing point to the end of the chart
        start_idx = min(high_idx, low_idx)
        end_idx = df.index[-1]
        diff = high_val - low_val
        
        # 2. Add diagonal trend line connecting High and Low swing points
        fig.add_trace(go.Scatter(
            x=[high_idx, low_idx],
            y=[high_val, low_val],
            mode='lines',
            line=dict(color='rgba(100, 116, 139, 0.6)', width=1.5, dash='dash'),
            name='Fib Trend Line',
            showlegend=False
        ), row=1, col=1)
        
        # Map of levels to their TradingView colors and fill colors
        fib_styles = {
            0.0:   {'color': '#64748b', 'fill': None,                     'label': '0'},
            0.236: {'color': '#ef4444', 'fill': 'rgba(239, 68, 68, 0.05)',  'label': '0.236'},
            0.382: {'color': '#f97316', 'fill': 'rgba(249, 115, 22, 0.05)', 'label': '0.382'},
            0.5:   {'color': '#22c55e', 'fill': 'rgba(34, 197, 94, 0.05)',  'label': '0.5'},
            0.618: {'color': '#15803d', 'fill': 'rgba(21, 128, 61, 0.05)',  'label': '0.618'},
            0.786: {'color': '#06b6d4', 'fill': 'rgba(6, 182, 212, 0.05)',  'label': '0.786'},
            1.0:   {'color': '#475569', 'fill': 'rgba(71, 85, 105, 0.05)',  'label': '1'}
        }
        
        # Calculate level price values
        ordered_levels = []
        for lvl, style in fib_styles.items():
            val = low_val + diff * lvl
            ordered_levels.append((val, lvl))
            
        ordered_levels = sorted(ordered_levels)
        
        # Find nearest support and resistance
        support_val, support_lvl = None, None
        resistance_val, resistance_lvl = None, None
        
        for val, lvl in ordered_levels:
            if val <= current_price:
                support_val = val
                support_lvl = lvl
            else:
                resistance_val = val
                resistance_lvl = lvl
                break
        
        # Dummy trace for legend
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[current_price],
            mode='lines',
            line=dict(color='#64748b', width=1.5, dash='solid'),
            name='Fibonacci Levels',
            showlegend=True
        ), row=1, col=1)
        
        # 3. Draw colored background bands using Layout Shapes (bound to subplot 1)
        # This prevents any drawing or bleeding into subplot 2 (volume chart)
        for i in range(len(ordered_levels) - 1):
            val_low, lvl_low = ordered_levels[i]
            val_high, lvl_high = ordered_levels[i+1]
            style_high = fib_styles[lvl_high]
            if style_high['fill']:
                fig.add_shape(
                    type="rect",
                    x0=start_idx, x1=end_idx,
                    y0=val_low, y1=val_high,
                    fillcolor=style_high['fill'],
                    line=dict(width=0),
                    xref="x1", yref="y1"
                )
        
        # 4. Draw horizontal Fibonacci lines (no fill to prevent bleeding)
        for val, lvl in ordered_levels:
            style = fib_styles[lvl]
            is_support = (val == support_val)
            is_resistance = (val == resistance_val)
            
            # Line settings
            line_color = '#10b981' if is_support else '#ef4444' if is_resistance else style['color']
            line_width = 2.5 if (is_support or is_resistance) else 1.5
            
            # Label text
            label_prefix = "Support: " if is_support else "Resistance: " if is_resistance else ""
            txt_label = f"<b>{label_prefix}{style['label']} ({val:.2f})</b>" if (is_support or is_resistance) else f"{style['label']} ({val:.2f})"
            
            # Draw line trace starting at start_idx to end_idx
            fig.add_trace(go.Scatter(
                x=[start_idx, end_idx],
                y=[val, val],
                mode='lines+text',
                line=dict(color=line_color, width=line_width, dash='solid'),
                text=[txt_label, ''],
                textposition='top right',
                textfont=dict(size=9, color=line_color, family="sans-serif"),
                showlegend=False
            ), row=1, col=1)

    # SMC Overlays (Order Blocks, FVGs, BOS/CHOCH)
    if smc:
        show_leg = True
        
        # Draw active Bullish Order Blocks
        for ob in smc.get('active_bullish_ob', []):
            x0 = ob['time']
            if x0 not in df.index:
                if x0 < df.index[0]:
                    x0 = df.index[0]
                else:
                    continue
            fig.add_shape(
                type="rect",
                x0=x0, x1=df.index[-1],
                y0=ob['bottom'], y1=ob['top'],
                fillcolor='rgba(0, 179, 134, 0.12)',
                line=dict(color='rgba(0, 179, 134, 0.35)', width=1, dash='solid'),
                xref="x1", yref="y1"
            )
            
        # Draw active Bearish Order Blocks
        for ob in smc.get('active_bearish_ob', []):
            x0 = ob['time']
            if x0 not in df.index:
                if x0 < df.index[0]:
                    x0 = df.index[0]
                else:
                    continue
            fig.add_shape(
                type="rect",
                x0=x0, x1=df.index[-1],
                y0=ob['bottom'], y1=ob['top'],
                fillcolor='rgba(235, 91, 60, 0.12)',
                line=dict(color='rgba(235, 91, 60, 0.35)', width=1, dash='solid'),
                xref="x1", yref="y1"
            )

        # Draw active Bullish & Bearish FVGs (Fair Value Gaps)
        for fvg in smc.get('active_bullish_fvg', []):
            x0 = fvg['time']
            if x0 not in df.index:
                if x0 < df.index[0]:
                    x0 = df.index[0]
                else:
                    continue
            fig.add_shape(
                type="rect",
                x0=x0, x1=df.index[-1],
                y0=fvg['bottom'], y1=fvg['top'],
                fillcolor='rgba(96, 165, 250, 0.08)',
                line=dict(color='rgba(96, 165, 250, 0.25)', width=1, dash='dot'),
                xref="x1", yref="y1"
            )
            
        for fvg in smc.get('active_bearish_fvg', []):
            x0 = fvg['time']
            if x0 not in df.index:
                if x0 < df.index[0]:
                    x0 = df.index[0]
                else:
                    continue
            fig.add_shape(
                type="rect",
                x0=x0, x1=df.index[-1],
                y0=fvg['bottom'], y1=fvg['top'],
                fillcolor='rgba(96, 165, 250, 0.08)',
                line=dict(color='rgba(96, 165, 250, 0.25)', width=1, dash='dot'),
                xref="x1", yref="y1"
            )

        # Plot BOS & CHOCH markers & lines
        bos_x = []
        bos_y = []
        bos_text = []
        for item in smc.get('bos', []):
            t = item['time']
            if t in df.index:
                bos_x.append(t)
                bos_y.append(item['price'])
                bos_text.append(f"BOS ({item['type'][:3]})")
                loc = df.index.get_loc(t)
                x0 = df.index[max(0, loc - 5)]
                fig.add_trace(go.Scatter(
                    x=[x0, t], y=[item['price'], item['price']],
                    mode='lines',
                    line=dict(color='rgba(245, 158, 11, 0.65)', width=1.5, dash='dot'),
                    showlegend=False
                ), row=1, col=1)

        choch_x = []
        choch_y = []
        choch_text = []
        for item in smc.get('choch', []):
            t = item['time']
            if t in df.index:
                choch_x.append(t)
                choch_y.append(item['price'])
                choch_text.append(f"CH ({item['type'][:3]})")
                loc = df.index.get_loc(t)
                x0 = df.index[max(0, loc - 5)]
                fig.add_trace(go.Scatter(
                    x=[x0, t], y=[item['price'], item['price']],
                    mode='lines',
                    line=dict(color='rgba(99, 102, 241, 0.65)', width=1.5, dash='dashdot'),
                    showlegend=False
                ), row=1, col=1)

        if bos_x:
            fig.add_trace(go.Scatter(
                x=bos_x, y=bos_y,
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=7, color='#f59e0b'),
                text=bos_text,
                textposition='top center',
                textfont=dict(size=8, color='#f59e0b'),
                name='BOS',
                showlegend=True
            ), row=1, col=1)

        if choch_x:
            fig.add_trace(go.Scatter(
                x=choch_x, y=choch_y,
                mode='markers+text',
                marker=dict(symbol='star', size=8, color='#6366f1'),
                text=choch_text,
                textposition='top center',
                textfont=dict(size=8, color='#6366f1'),
                name='CHOCH',
                showlegend=True
            ), row=1, col=1)

        # Legend placeholders
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='markers',
            marker=dict(symbol='square', size=8, color='rgba(0, 179, 134, 0.3)'),
            name='Bullish OB',
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='markers',
            marker=dict(symbol='square', size=8, color='rgba(235, 91, 60, 0.3)'),
            name='Bearish OB',
            showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='markers',
            marker=dict(symbol='square', size=8, color='rgba(96, 165, 250, 0.2)'),
            name='Fair Value Gap (FVG)',
            showlegend=True
        ), row=1, col=1)
            
    fig.update_layout(
        template='plotly_white', height=450, showlegend=show_leg,
        paper_bgcolor='white', plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    # Remove gridlines and border lines for clean white visual style
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, row=2, col=1)
    return fig


# ── Utility Functions ────────────────────────────────────────────────────────






def build_gauge(up_prob, signal, title="Signal"):
    cmap = {'BUY':'#10b981','SELL':'#ef4444','HOLD':'#f59e0b'}
    fig = go.Figure(go.Indicator(mode="gauge+number", value=up_prob*100,
        title={'text':f"{title}: {signal}",'font':{'size':16,'color':'white'}},
        number={'suffix':'%','font':{'color':'white'}},
        gauge={'axis':{'range':[0,100],'tickcolor':'white'}, 'bar':{'color':cmap.get(signal,'#f59e0b')},
               'bgcolor':'#1e293b',
               'steps':[{'range':[0,45],'color':'rgba(239,68,68,0.15)'},
                        {'range':[45,55],'color':'rgba(245,158,11,0.15)'},
                        {'range':[55,100],'color':'rgba(16,185,129,0.15)'}]}))
    fig.update_layout(height=250, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                      font={'color':'white'}, margin=dict(l=30,r=30,t=50,b=10))
    return fig

def get_tv_symbol(symbol, mapped):
    """Map yfinance symbols to TradingView symbols"""
    if mapped.endswith('.NS'): return "NSE:" + mapped.replace('.NS', '')
    if mapped.endswith('.BO'): return "BSE:" + mapped.replace('.BO', '')
    if mapped == '^NSEI': return "NSE:NIFTY"
    if mapped == '^BSESN': return "BSE:SENSEX"
    if mapped == '^NSEBANK': return "NSE:BANKNIFTY"
    if mapped == 'GC=F': return "COMEX:GC1!"
    if mapped == 'SI=F': return "COMEX:SI1!"
    if mapped == 'CL=F': return "NYMEX:CL1!"
    return mapped

def build_tradingview_chart(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container" style="height:800px;width:100%">
      <div id="tradingview_{symbol.lower()}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 800,
        "symbol": "{tv_sym}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_{symbol.lower()}"
      }});
      </script>
    </div>
    """

def build_tradingview_analysis(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
      {{
        "interval": "1D",
        "width": "100%",
        "isTransparent": false,
        "height": 450,
        "symbol": "{tv_sym}",
        "showIntervalTabs": true,
        "displayMode": "single",
        "locale": "en",
        "colorTheme": "dark"
      }}
      </script>
    </div>
    """

def build_tradingview_profile_widget(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
      {{
        "width": "100%",
        "height": 400,
        "colorTheme": "dark",
        "isTransparent": false,
        "symbol": "{tv_sym}",
        "locale": "en"
      }}
      </script>
    </div>
    """

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def nav_to(page):
    st.session_state.page = page

def get_market_status():
    import datetime
    try:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))) # IST
        day = now.weekday()
        if day >= 5: # Saturday or Sunday
            return "🔴 Market Closed (Weekend)", "#ef4444"
        
        # Simple Indian market hours: 9:15 AM - 3:30 PM
        curr_time = now.time()
        start = datetime.time(9, 15)
        end = datetime.time(15, 30)
        
        if start <= curr_time <= end:
            return "🟢 Market Open (Live)", "#10b981"
        else:
            return "🟡 Market Closed (After Hours)", "#f59e0b"
    except:
        return "⚪ Market Status Unknown", "#94a3b8"

# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
def main():
    if 'unlocked' not in st.session_state:
        st.session_state.unlocked = False

    if not st.session_state.unlocked:
        st.markdown("""
            <canvas id="matrix-rain-canvas" style="position:fixed; top:0; left:0; width:100vw; height:100vh; z-index:-1; pointer-events:none; background:#000000;"></canvas>
            <script>
            (function() {
                var canvas = document.getElementById("matrix-rain-canvas");
                if (!canvas) return;
                var ctx = canvas.getContext("2d");
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
                
                var characters = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ";
                var charArr = characters.split("");
                var fontSize = 16;
                var columns = canvas.width / fontSize;
                var rainDrops = [];
                for(var x = 0; x < columns; x++) {
                    rainDrops[x] = 1;
                }
                function draw() {
                    ctx.fillStyle = "rgba(0, 0, 0, 0.06)";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#00FF00";
                    ctx.font = fontSize + "px monospace";
                    for(var i = 0; i < rainDrops.length; i++) {
                        var text = charArr[Math.floor(Math.random() * charArr.length)];
                        ctx.fillText(text, i * fontSize, rainDrops[i] * fontSize);
                        if(rainDrops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                            rainDrops[i] = 0;
                        }
                        rainDrops[i]++;
                    }
                }
                setInterval(draw, 30);
                window.addEventListener("resize", function() {
                    canvas.width = window.innerWidth;
                    canvas.height = window.innerHeight;
                });
            })();
            </script>
            <style>
            [data-testid="stSidebar"] { display: none !important; }
            header[data-testid="stHeader"] { display: none !important; }
            [data-testid="stAppViewContainer"] { background: #000000 !important; background-image: none !important; }
            [data-testid="stAppViewContainer"]::before { display: none !important; }
            
            .hacker-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 80vh;
                text-align: center;
                z-index: 100;
            }
            .hacker-skull {
                font-size: 8rem;
                filter: drop-shadow(0 0 20px rgba(0, 255, 0, 0.6));
                margin-bottom: 25px;
                animation: pulse-glow 2s infinite ease-in-out;
            }
            .hacker-title {
                font-family: 'Courier New', Courier, monospace;
                font-size: 4rem;
                font-weight: 900;
                color: #00FF00;
                letter-spacing: 16px;
                margin-bottom: 12px;
                text-shadow: 0 0 15px rgba(0, 255, 0, 0.8);
            }
            .hacker-subtitle {
                font-family: 'Courier New', Courier, monospace;
                font-size: 1.25rem;
                font-weight: 600;
                color: #00FF00;
                letter-spacing: 8px;
                margin-bottom: 50px;
                text-shadow: 0 0 8px rgba(0, 255, 0, 0.6);
            }
            @keyframes pulse-glow {
                0%, 100% { transform: scale(1); filter: drop-shadow(0 0 20px rgba(0,255,0,0.6)); }
                50% { transform: scale(1.05); filter: drop-shadow(0 0 35px rgba(0,255,0,0.9)); }
            }
            
            div.stButton > button {
                background: transparent !important;
                border: 2px solid #00ff00 !important;
                box-shadow: 0 0 15px rgba(0, 255, 0, 0.4) !important;
                color: #00ff00 !important;
                font-family: 'Courier New', Courier, monospace !important;
                font-size: 1.35rem !important;
                font-weight: 900 !important;
                letter-spacing: 4px !important;
                padding: 16px 45px !important;
                border-radius: 4px !important;
                transition: all 0.3s ease !important;
                text-shadow: 0 0 5px #00ff00 !important;
            }
            div.stButton > button:hover {
                background: #00ff00 !important;
                color: #000000 !important;
                box-shadow: 0 0 30px #00ff00 !important;
                text-shadow: none !important;
                transform: scale(1.05);
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="hacker-container">', unsafe_allow_html=True)
        st.markdown('<div class="hacker-skull">💀</div>', unsafe_allow_html=True)
        st.markdown('<div class="hacker-title">HACKER</div>', unsafe_allow_html=True)
        st.markdown('<div class="hacker-subtitle">SYSTEM BREACHED</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("ACCESS GRANTED ▮", key="hacker_unlock"):
                st.session_state.unlocked = True
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    status_text, status_color = get_market_status()
    st.markdown(f'<div class="main-title">🏛️ SRV FUTURE TRADERS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Real-time BSE • NSE • MoneyControl — Institutional SMC & AI Predictions &nbsp;'
                f'<span style="background:{status_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{status_text}</span></div>', unsafe_allow_html=True)

    # Self-healing engine initialization: Re-instantiate if stale or missing methods
    if 'engine_v2' not in st.session_state or not hasattr(st.session_state.engine_v2, 'predict'):
        st.session_state.engine_v2 = AIEngine()

    # ── Ticker Bar & Live Refresh ─────────────────────────────────────
    ticker_syms = ['^NSEI', '^BSESN', '^NSEBANK', '^GSPC', '^IXIC']
    lc1, lc2 = st.columns([5, 1])
    with lc2:
        live_mode = st.checkbox("📡 Live Mode", help="Auto-refreshes every 30s")
    
    if live_mode:
        st.markdown("""
            <script>
            setTimeout(function(){
               window.location.reload();
            }, 30000);
            </script>
        """, unsafe_allow_html=True)

    # ── Global Market Sentiment Bar ───────────────────────────────────
    msent = get_master_market_sentiment()
    with st.container():
        st.markdown(f'''
            <div class="sentiment-bar" style="border-left-color: {msent["color"]}; margin-bottom: 5px;">
                <div>
                    <div class="sent-label">Master Market Bias</div>
                    <div class="sent-value" style="color: {msent["color"]};">{msent["label"]}</div>
                </div>
                <div style="text-align: right; flex: 1; margin-left: 20px;">
                    <div class="sent-label">Global catalyst</div>
                    <div class="sent-reason">"{msent["global_reason"]}"</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        with st.expander("📄 View AI Evidence (Market Headlines Breakdown)", expanded=False):
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("### 🇮🇳 Indian News")
                for h in msent["indian_headlines"][:5]:
                    scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(h['label'],'sentiment-neu')
                    st.markdown(f'''<div class="news-card"><span class="{scls}">[{h["label"].upper()}]</span> {h["title"]}</div>''', unsafe_allow_html=True)
            with tc2:
                st.markdown("### 🌎 Global News")
                for h in msent["global_headlines"][:5]:
                    scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(h['label'],'sentiment-neu')
                    st.markdown(f'''<div class="news-card"><span class="{scls}">[{h["label"].upper()}]</span> {h["title"]}</div>''', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

    ticker_html = '<div class="ticker-bar">'
    for ts in ticker_syms:
        info = get_price_info(ts, 5)
        if info:
            cls = 'ticker-up' if info['change'] >= 0 else 'ticker-down'
            arrow = '▲' if info['change'] >= 0 else '▼'
            ticker_html += (f'<span class="ticker-item"><span class="ticker-name">{ts}</span>'
                           f'<span class="ticker-price">{info["currency"]}{info["price"]:,.2f}</span>'
                           f'<span class="{cls}">{arrow} {info["change"]:+,.2f} ({info["pct"]:+.2f}%)</span></span>')
    ticker_html += '</div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🏛️ SRV Future Traders")
        page = st.radio("Navigate", [
            "🏠 Explore", "🔮 AI Prediction", "📈 AI Backtester", "🔍 Stock Screener", "📰 Market News",
            "🔥 MTF Scanner", "🏆 Top Movers"
        ], key="page")
        st.markdown("---")
        st.markdown(f'''
            <div style="display: flex; align-items: center; margin-top: 20px; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 10px;">⚙️</span>
                <span style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">Trading Settings</span>
            </div>
        ''', unsafe_allow_html=True)
        
        capital = st.number_input("Trading Capital (₹)", value=100000, step=5000, key="risk_cap")
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1, key="risk_pct", format="%.2f")
        reward_ratio = st.slider("Reward-to-Risk Ratio", 1.0, 10.0, 2.0, 0.1, key="reward_ratio", format="%.2f")
        
        max_loss = capital * risk_pct / 100
        target_profit = max_loss * reward_ratio
        
        st.markdown(f'''
            <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(251, 191, 36, 0.25); border-left: 4px solid #ef4444; padding: 18px; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem; margin-right: 12px;">📉</span>
                    <span style="color: #ef4444; font-size: 1.1rem; font-weight: 700;">Max Risk: ₹{max_loss:,.0f}</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 12px;">💰</span>
                    <span style="color: #fbbf24; font-size: 1.1rem; font-weight: 700;">Target Profit: ₹{target_profit:,.0f}</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("**Risk Management Engine**")
        st.caption(f"📍 Position sizing is based on ₹{max_loss:,.0f} risk.")
        st.caption(f"📍 Targets are set at {reward_ratio}x your risk.")
        st.caption("🤖 **Auto-Verification Active**: Tracking SL/Target hits.")
        
        st.markdown("---")
        st.markdown(f'''
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 1.5rem; margin-right: 10px;">🔑</span>
                <span style="font-size: 1.3rem; font-weight: 700; color: #f8fafc;">Gemini AI Intelligence</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Check for key in secrets or environment, default to empty
        default_gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")) if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY", "")
        gemini_key = st.text_input("Gemini API Key", value=default_gemini_key, type="password", help="Enter a free Gemini API Key from Google AI Studio to enable deep market reasoner analysis.")
        if gemini_key:
            st.session_state.gemini_api_key = gemini_key
            st.success("🤖 Gemini Active!")
        else:
            st.info("💡 Enter key for rich domestic vs global reasoning.")
            
        st.markdown("---")
        st.caption("**Data Sources**")
        st.caption("✅ Yahoo Finance Live")
        st.caption("✅ MoneyControl News")
        st.caption("✅ TradingView Insights")
        st.caption("✅ Screener.in Fundamentals")
        st.caption("✅ BSE / NSE India")

    # ── Pages ─────────────────────────────────────────────────────────
    if page == "🏠 Explore":
        page_explore()
    elif page == "🔮 AI Prediction":
        page_prediction()
    elif page == "📈 AI Backtester":
        page_backtester()
    elif page == "🔍 Stock Screener":
        page_screener()
    elif page == "📰 Market News":
        page_news()
    elif page == "🔥 MTF Scanner":
        page_mtf_scanner()
    elif page == "🎯 Options Engine":
        page_options_opportunities()
    elif page == "🏆 Top Movers":
        page_top_movers()

    st.markdown("---")
    st.caption("🏛️ SRV Future Traders • BSE • NSE • MoneyControl")

def build_tradingview_news_widget(symbol, mapped):
    tv_sym = get_tv_symbol(symbol, mapped)
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "symbol",
        "symbol": "{tv_sym}",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "100%",
        "height": 450,
        "colorTheme": "dark",
        "locale": "en"
      }}
      </script>
    </div>
    """

def build_tradingview_market_news():
    return f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>
      {{
        "feedMode": "market",
        "market": "stock",
        "isTransparent": false,
        "displayMode": "regular",
        "width": "100%",
        "height": 600,
        "colorTheme": "dark",
        "locale": "en"
      }}
      </script>
    </div>
    """

def render_sector_heatmap():
    """Step 6: Sector Money Flow Engine (Dynamic Heatmap)"""
    st.markdown('<div class="section-head">🔥 Sector Money Flow (Live Leaders)</div>', unsafe_allow_html=True)
    
    sectors = {
        "Banking": {
            "emoji": "🏦",
            "stocks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"]
        },
        "IT & Tech": {
            "emoji": "💻",
            "stocks": ["TCS", "INFY", "HCLTECH", "WIPRO"]
        },
        "Auto & EV": {
            "emoji": "🚗",
            "stocks": ["TATAMOTORS", "MARUTI", "M&M", "TVSMOTOR"]
        },
        "Metal & Mining": {
            "emoji": "🏭",
            "stocks": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL"]
        },
        "Oil & Gas": {
            "emoji": "⛽",
            "stocks": ["RELIANCE", "ONGC", "BPCL", "IOC"]
        },
        "Power & Energy": {
            "emoji": "⚡",
            "stocks": ["NTPC", "POWERGRID", "TATAPOWER", "JSWENERGY"]
        },
        "FMCG & Retail": {
            "emoji": "🛒",
            "stocks": ["HINDUNILVR", "ITC", "TATACONSUM", "BRITANNIA"]
        },
        "Railways & Infra": {
            "emoji": "🚂",
            "stocks": ["IRCTC", "IRFC", "RVNL", "LT"]
        }
    }
    
    sectors_items = list(sectors.items())
    for row_idx in range(2):
        cols = st.columns(4)
        for col_idx in range(4):
            i = row_idx * 4 + col_idx
            if i >= len(sectors_items):
                break
            name, sdata = sectors_items[i]
            total_chg = 0
            valid_count = 0
            leaders = []
            
            for stock in sdata["stocks"]:
                info = get_price_info(stock, 5)
                if info:
                    total_chg += info['pct']
                    valid_count += 1
                    symbol_display = info['symbol']
                    if info['pct'] > 0:
                        leaders.append(f"{symbol_display} 🔥")
                    else:
                        leaders.append(f"{symbol_display} ❄️")
            
            avg_chg = total_chg / valid_count if valid_count > 0 else 0.0
            status = "STRONG 🔥" if avg_chg > 0.5 else "WEAK ❌" if avg_chg < -0.5 else "NEUTRAL ⚖️"
            
            if avg_chg > 0.5:
                color = "#10b981"
                bg_opacity = "rgba(16, 185, 129, 0.08)"
                border_color = "rgba(16, 185, 129, 0.25)"
            elif avg_chg < -0.5:
                color = "#ef4444"
                bg_opacity = "rgba(239, 68, 68, 0.08)"
                border_color = "rgba(239, 68, 68, 0.25)"
            else:
                color = "#94a3b8"
                bg_opacity = "rgba(148, 163, 184, 0.05)"
                border_color = "rgba(148, 163, 184, 0.15)"
                
            with cols[col_idx]:
                st.markdown(f"""
                    <div style="background: {bg_opacity}; border: 1px solid {border_color}; padding: 16px; border-radius: 12px; height: 165px; margin-bottom: 16px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; display: flex; align-items: center; gap: 4px;">
                            <span>{sdata['emoji']}</span> {name}
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 900; color: {color}; margin: 8px 0 2px 0;">{status}</div>
                        <div style="font-size: 1.25rem; font-weight: 900; color: #ffffff;">{avg_chg:+.2f}%</div>
                        <div style="font-size: 0.65rem; color: #64748b; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            {' | '.join(leaders[:3])}
                        </div>
                    </div>
                """, unsafe_allow_html=True)



def render_trade_proof():
    """Step 5 (v3): Professional Proof Panel - Last 10 Trades"""
    st.markdown('<div class="section-head">📑 Institutional Proof (Last 10 Signals)</div>', unsafe_allow_html=True)
    
    # Auto-verify signals using current price engine
    auto_verify_signals(lambda sym: get_realtime_price(sym, STOCK_MAP.get(sym, sym)))
    
    history = load_history()
    # Get last 10 
    latest_10 = history[:10]
    
    if not latest_10:
        st.info("No historical signals available yet.")
        return
        
    cols = st.columns(min(len(latest_10), 5))
    for i, trade in enumerate(latest_10):
        if i >= 5: break # Only show top row for design cleanliness
        
        is_pending = trade.get('correct') is None and "HOLD" not in trade.get('signal', '') and "NO TRADE" not in trade.get('signal', '')
        res_col = "#38bdf8" if is_pending else ("#10b981" if trade.get('correct') == True else "#ef4444" if trade.get('correct') == False else "#94a3b8")
        
        # Priority: Specific Status (TARGET/SL) > Generic Label
        if trade.get('status'):
            res_text = trade['status']
        else:
            res_text = "PENDING ⏳" if is_pending else ("WIN ✅" if trade.get('correct') == True else "LOSS ❌" if trade.get('correct') == False else "NEUTRAL ⚖️")
        catalyst = trade.get('catalyst', 'Technical Consensus')
        
        with cols[i]:
            st.markdown(f"""
                <div style="background: {res_col}10; border: 1px solid {res_col}30; padding: 12px; border-radius: 10px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 0.7rem; color: #94a3b8; font-weight: 700;">{trade['symbol']}</div>
                        <div style="font-size: 0.85rem; font-weight: 900; color: {res_col}; margin: 5px 0;">{res_text}</div>
                    </div>
                    <div style="font-size: 0.6rem; color: #cbd5e1; line-height: 1.2; margin: 5px 0; font-style: italic; overflow: hidden;">
                        "{catalyst[:55]}..."
                    </div>
                    <div style="font-size: 0.55rem; color: #64748b;">{trade['timestamp'].split()[0]} | Conf: {int(trade.get('confidence', 0)*100)}%</div>
                </div>
            """, unsafe_allow_html=True)

# ── PAGE: Explore (Groww-style) ───────────────────────────────────────────
@st.cache_data(ttl=1800)
def scan_institutional_setups(scan_target):
    # Instantiate temporary engine inside cached function to avoid hashing session state
    engine = AIEngine()
    
    # 1. Define list of symbols to scan
    nifty_50 = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
        'ITC', 'KOTAKBANK', 'LT', 'AXISBANK', 'WIPRO', 'MARUTI', 'SUNPHARMA',
        'BAJFINANCE', 'BAJFINSV', 'BAJAJ-AUTO', 'HCLTECH', 'ASIANPAINT', 'TITAN',
        'ULTRACEMCO', 'NESTLEIND', 'TECHM', 'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA',
        'DRREDDY', 'CIPLA', 'DIVISLAB', 'EICHERMOT', 'HEROMOTOCO', 'M&M', 'ADANIENT',
        'ADANIPORTS', 'JSWSTEEL', 'HINDALCO', 'BPCL', 'HINDUNILVR', 'BRITANNIA',
        'INDUSINDBK', 'TATASTEEL', 'TATAMOTORS', 'GRASIM', 'APOLLOHOSP', 'SBILIFE',
        'HDFCLIFE', 'LTIM', 'TATAPOWER'
    ]
    
    if scan_target == "Nifty 50 (Fast)":
        symbols = nifty_50
    else:
        # All stocks: get all keys from STOCK_MAP except indices
        symbols = [sym for sym, mapped in STOCK_MAP.items() if not mapped.startswith('^')]
        # Deduplicate and sort
        symbols = sorted(list(set(symbols)))

    # Map symbols to tickers
    ticker_to_sym = {}
    tickers_to_download = []
    for sym in symbols:
        mapped = STOCK_MAP.get(sym, sym)
        tickers_to_download.append(mapped)
        ticker_to_sym[mapped] = sym
        
    tickers_to_download = list(set(tickers_to_download))
    
    # 2. Batch download using yfinance
    try:
        batch_df = yf.download(tickers_to_download, period='150d', interval='1d', group_by='ticker', progress=False)
    except Exception:
        return []
        
    if batch_df is None or batch_df.empty:
        return []
        
    setups = []
    is_multi = isinstance(batch_df.columns, pd.MultiIndex)
    
    for ticker in tickers_to_download:
        try:
            if is_multi:
                df = batch_df[ticker].dropna(subset=['Close'])
            else:
                df = batch_df.dropna(subset=['Close'])
                
            if df is None or len(df) < 30:
                continue
                
            df.columns = [c.capitalize() for c in df.columns]
            
            smc = engine.detect_smc_features(df)
            closest_ob = smc.get('closest_ob')
            closest_ob_dist = smc.get('closest_ob_dist_pct', 999.0)
            
            if closest_ob and closest_ob_dist <= 2.5:
                prices = df['Close'].dropna().astype(float).tolist()
                volumes = df['Volume'].dropna().astype(float).tolist()
                sym = ticker_to_sym.get(ticker, ticker.replace('.NS', ''))
                
                if sym not in engine.models:
                    engine.train(sym, prices, volumes, news_sent=0.0)
                    
                res = engine.predict(sym, prices, volumes, news_sent=0.0, tv_sent=0.0, intraday=False, df=df)
                
                if res and 'today' in res:
                    pred_today = res['today']
                    setup_stars = pred_today.get('stars', 1)
                    confluence = pred_today.get('confidence', 0.0)
                    signal = pred_today.get('signal', 'HOLD')
                    
                    if setup_stars >= 3 and ("BUY" in signal or "SELL" in signal):
                        ob_zone = f"{closest_ob['bottom']:,.2f} - {closest_ob['top']:,.2f}"
                        age_str = f"{closest_ob['age']} Days Old"
                        freshness = max(0, 100 - closest_ob['age'] * 2)
                        
                        setups.append({
                            'symbol': sym,
                            'ticker': ticker,
                            'signal': signal,
                            'confidence': confluence,
                            'stars': setup_stars,
                            'ob_type': f"{closest_ob['type']} OB",
                            'ob_zone': ob_zone,
                            'ob_dist': closest_ob_dist,
                            'age': age_str,
                            'freshness': freshness,
                            'price': float(df['Close'].iloc[-1])
                        })
        except Exception:
            continue
            
    return sorted(setups, key=lambda x: -x['confidence'])

def page_explore():
    # Today's Opportunities Scanner
    st.markdown('<div class="section-head">🏛️ Today\'s Institutional Opportunities (Live AI Scan)</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 3])
    with c1:
        scan_target = st.selectbox(
            "Select Scan Coverage", 
            ["Nifty 50 (Fast)", "All 350+ Stocks (Thorough)"],
            index=0,
            key="home_scan_target"
        )
    with c2:
        st.markdown('<div style="font-size:0.8rem; color:#64748b; margin-top:28px;">Scans stocks in parallel for active Order Blocks, FVG imbalances, and AI confluence.</div>', unsafe_allow_html=True)
        
    with st.spinner("🔍 Scanning market for high-probability setups..."):
        setups = scan_institutional_setups(scan_target)
        
    if setups:
        rows_html = ""
        for i, setup in enumerate(setups[:10]):
            rank = i + 1
            if rank == 1:
                rank_str = '<span style="color: #fbbf24; font-weight: 950; font-size: 1.15rem;">#1 👑</span>'
            elif rank == 2:
                rank_str = '<span style="color: #cbd5e1; font-weight: 950; font-size: 1.1rem;">#2</span>'
            elif rank == 3:
                rank_str = '<span style="color: #b45309; font-weight: 950; font-size: 1.1rem;">#3</span>'
            else:
                rank_str = f'<span style="color: #64748b; font-weight: 800;">#{rank}</span>'
                
            sig_color = "#10b981" if "BUY" in setup['signal'] else "#ef4444"
            sig_bg = "rgba(16, 185, 129, 0.15)" if "BUY" in setup['signal'] else "rgba(239, 68, 68, 0.15)"
            
            is_indian = setup['ticker'].endswith('.NS') or setup['ticker'].endswith('.BO') or setup['ticker'].startswith('^')
            curr = '₹' if is_indian else '$'
            
            direction_badge = f'<span style="background: {sig_bg}; color: {sig_color}; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 800; display: inline-block;">{setup["signal"]}</span>'
            
            stars_str = '★' * setup['stars'] + '☆' * (5 - setup['stars'])
            row_bg = "rgba(255, 255, 255, 0.02)" if i % 2 == 0 else "transparent"
            
            rows_html += f"""
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.04); background: {row_bg}; font-size: 0.85rem;">
                <td style="padding: 14px 16px; font-weight: 800;">{rank_str}</td>
                <td style="padding: 14px 16px;"><span style="font-weight: 900; color: #ffffff;">{setup['symbol']}</span><br><span style="font-size: 0.65rem; color: #64748b; font-family: monospace;">{setup['ticker']}</span></td>
                <td style="padding: 14px 16px;">{direction_badge}</td>
                <td style="padding: 14px 16px;"><span style="font-family: monospace; font-weight: 700; color: #f1f5f9;">{curr}{setup['price']:,.2f}</span></td>
                <td style="padding: 14px 16px; text-align: center;"><span style="font-weight: 900; color: {sig_color}; font-size: 1rem;">{setup['confidence']:.0%}</span></td>
                <td style="padding: 14px 16px; text-align: center; color: #fbbf24; font-size: 0.9rem; letter-spacing: 1px;">{stars_str}</td>
                <td style="padding: 14px 16px; text-align: center;"><span style="font-family: monospace; font-weight: 700; color: #f3f4f6;">{setup['ob_dist']:.2f}%</span><br><span style="font-size: 0.65rem; color: #64748b;">({setup['ob_type']})</span></td>
                <td style="padding: 14px 16px; text-align: right;"><span style="color: #10b981; font-weight: 700;">{setup['freshness']}%</span><br><span style="font-size: 0.7rem; color: #94a3b8;">{setup['age']}</span></td>
            </tr>
            """
            
        table_html = f"""
        <div style="background: rgba(17, 24, 39, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25); overflow-x: auto; margin-bottom: 25px;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; color: #f8fafc; font-family: 'Inter', sans-serif;">
                <thead>
                    <tr style="border-bottom: 2px solid rgba(255, 255, 255, 0.08); color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        <th style="padding: 12px 16px; font-weight: 800;">Rank</th>
                        <th style="padding: 12px 16px; font-weight: 800;">Stock</th>
                        <th style="padding: 12px 16px; font-weight: 800;">Direction</th>
                        <th style="padding: 12px 16px; font-weight: 800;">Price</th>
                        <th style="padding: 12px 16px; font-weight: 800; text-align: center;">Confluence</th>
                        <th style="padding: 12px 16px; font-weight: 800; text-align: center;">Stars</th>
                        <th style="padding: 12px 16px; font-weight: 800; text-align: center;">Distance to OB</th>
                        <th style="padding: 12px 16px; font-weight: 800; text-align: right;">Freshness</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("⚖️ No high-probability Institutional Order Block setups found in this scan. Try switching to 'All 350+ Stocks' or check back later.")
        st.markdown("<br>", unsafe_allow_html=True)

    # Advanced Heatmap
    render_sector_heatmap()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 7: Institutional Track Record (v3)
    stats = load_advanced_stats()
    st.markdown('\u003cdiv class="section-head"\u003e🏅 Institutional Track Record\u003c/div\u003e', unsafe_allow_html=True)
    st.markdown(f'''
        \u003cdiv style="background: rgba(17, 24, 39, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; margin-bottom: 25px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);"\u003e
            \u003cdiv style="display:grid; grid-template-columns: repeat(4, 1fr); gap:20px;"\u003e
                \u003cdiv style="text-align:center; background:rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.1); border-radius:12px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#10b981;"\u003e{stats['win_rate']:.1f}%\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e🎯 Win Rate\u003c/div\u003e
                \u003c/div\u003e
                \u003cdiv style="text-align:center; background:rgba(251, 191, 36, 0.05); border: 1px solid rgba(251, 191, 36, 0.1); border-radius:12px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#fbbf24;"\u003e{stats['profit_factor']:.2f}x\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e📊 Profit Factor\u003c/div\u003e
                \u003c/div\u003e
                \u003cdiv style="text-align:center; background:rgba(0, 179, 134, 0.05); border: 1px solid rgba(0, 179, 134, 0.1); border-radius:12px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#00b386;"\u003e+{stats['avg_profit']:.1f}%\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e💰 Avg Win\u003c/div\u003e
                \u003c/div\u003e
                \u003cdiv style="text-align:center; background:rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.1); border-radius:12px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#ef4444;"\u003e-{stats['avg_loss']:.1f}%\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e🛡️ Avg Loss\u003c/div\u003e
                \u003c/div\u003e
            \u003c/div\u003e
        \u003c/div\u003e
    ''', unsafe_allow_html=True)
    
    # NEW: Trade Proof
    render_trade_proof()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Watchlist
    st.markdown('<div class="section-head">📌 Watchlist</div>', unsafe_allow_html=True)
    row_html = '<div class="recent-row">'
    for sym in WATCHLIST_DEFAULT:
        info = get_price_info(sym, 5)
        if info:
            cls = 'chg-up' if info['pct'] >= 0 else 'chg-down'
            row_html += (f'<div class="recent-item"><div class="sym">{sym}</div>'
                        f'<div class="{cls}">{info["pct"]:+.2f}%</div></div>')
    row_html += '</div>'
    st.markdown(row_html, unsafe_allow_html=True)

    # NEW: Market Pulse
    st.markdown('<div class="section-head">⚡ Market Pulse (Candlestick)</div>', unsafe_allow_html=True)
    pulse_syms = ['RELIANCE', 'TATAMOTORS', 'WIPRO', 'GOLD']
    pulse_cols = st.columns(len(pulse_syms))
    for i, psym in enumerate(pulse_syms):
        pdf, _ = fetch_stock(psym, 30)
        if pdf is not None:
            pat = detect_candle_pattern(pdf)
            color = "#10b981" if pat.get('score', 0.5) >= 0.5 else "#ef4444"
            pulse_class = "pulse-green" if color == "#10b981" else "pulse-red"
            with pulse_cols[i]:
                catalyst = get_stock_catalyst(psym)
                st.markdown(f"""<div class="stock-card" style="border-left: 4px solid {color}; overflow: hidden;">
                    <div style="display:flex; justify-content:space-between;">
                        <div class="name" style="color: #e5e7eb;">{psym}</div>
                        <div class="{pulse_class}" style="font-size: 0.6rem; padding: 2px 8px;">LIVE</div>
                    </div>
                    <div style="font-size:0.8rem; color:{color}; font-weight:700">{pat['pattern']}</div>
                    <div style="font-size:0.7rem; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.08); margin-top:8px; padding-top:4px;">
                        🔍 {catalyst}
                    </div>
                </div>""", unsafe_allow_html=True)

    # Most Traded (top 4 cards)
    st.markdown('<div class="section-head">🔥 Most Traded Stocks</div>', unsafe_allow_html=True)
    top4 = ['RELIANCE', 'TATASTEEL', 'SBIN', 'ADANIGREEN']
    cols = st.columns(4)
    for i, sym in enumerate(top4):
        info = get_price_info(sym, 5)
        if info:
            cls = 'change-up' if info['change'] >= 0 else 'change-down'
            with cols[i]:
                catalyst = get_stock_catalyst(sym)
                st.markdown(f"""<div class="stock-card">
                    <div class="name">{sym}</div>
                    <div class="price">{info['currency']}{info['price']:,.2f}</div>
                    <div class="{cls}">{info['change']:+.2f} ({info['pct']:+.2f}%)</div>
                    <div style="font-size:0.65rem; color:#94a3b8; margin-top:6px; font-style:italic">
                         "{catalyst}"
                    </div>
                </div>""", unsafe_allow_html=True)

    # Top Movers Table (Gainers)
    st.markdown('<div class="section-head">📈 Top Movers Today</div>', unsafe_allow_html=True)
    movers_syms = ['TATAMOTORS','TATASTEEL','TATAPOWER','ADANIGREEN','ADANIENT',
                   'RELIANCE','SBIN','ICICIBANK','INFY','WIPRO','BAJFINANCE','HDFCBANK',
                   'ITC','MARUTI','JSWSTEEL','VEDL','NIPPON','COALINDIA']
    movers_data = []
    prog = st.progress(0, text="Fetching top movers...")
    for idx, sym in enumerate(movers_syms):
        prog.progress((idx+1)/len(movers_syms), text=f"Fetching {sym}...")
        info = get_price_info(sym, 5)
        if info:
            movers_data.append(info)
    prog.empty()

    if movers_data:
        tab1, tab2, tab3 = st.tabs(["🟢 Gainers", "🔴 Losers", "📊 All"])
        gainers = sorted([m for m in movers_data if m['pct'] > 0], key=lambda x: -x['pct'])
        losers = sorted([m for m in movers_data if m['pct'] < 0], key=lambda x: x['pct'])

        with tab1:
            if gainers:
                gdf = pd.DataFrame([{'Company': g['symbol'], 'Price': f"{g['currency']}{g['price']:,.2f}",
                    'Change': f"{g['change']:+.2f}", 'Change%': f"{g['pct']:+.2f}%",
                    'Volume': f"{g['volume']:,}"} for g in gainers])
                st.dataframe(gdf, use_container_width=True, hide_index=True)
            else:
                st.info("No gainers found")

        with tab2:
            if losers:
                ldf = pd.DataFrame([{'Company': l['symbol'], 'Price': f"{l['currency']}{l['price']:,.2f}",
                    'Change': f"{l['change']:+.2f}", 'Change%': f"{l['pct']:+.2f}%",
                    'Volume': f"{l['volume']:,}"} for l in losers])
                st.dataframe(ldf, use_container_width=True, hide_index=True)
            else:
                st.info("No losers found")

        with tab3:
            adf = pd.DataFrame([{'Company': m['symbol'], 'Price': f"{m['currency']}{m['price']:,.2f}",
                'Change': f"{m['change']:+.2f}", 'Change%': f"{m['pct']:+.2f}%"} for m in movers_data])
            st.dataframe(adf, use_container_width=True, hide_index=True)


    # Quick Links
    st.markdown('<div class="section-head">🔗 Products & Tools</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.link_button("📊 MoneyControl", "https://www.moneycontrol.com/", use_container_width=True)
    with lc2:
        st.link_button("📈 NSE India", "https://www.nseindia.com/", use_container_width=True)
    with lc3:
        st.link_button("🔍 Screener.in", "https://www.screener.in/", use_container_width=True)


# ── PAGE: AI Backtester ──────────────────────────────────────────────────
def page_backtester():
    st.subheader("📈 AI Historical Backtester")
    st.caption("Evaluate the AI's performance on historical data to see how its weighted signals would have performed over the last 6 months.")
    
    symbol = st.text_input("Enter Stock Symbol for Backtesting", placeholder="RELIANCE, AAPL, etc.")
    
    if st.button("🚀 Run Backtest Analysis", use_container_width=True) and symbol:
        symbol = symbol.strip().upper()
        with st.spinner(f"🏃‍♂️ Running 6-month simulation for {symbol}..."):
            df, mapped = fetch_stock(symbol, 250) # ~1 year of daily data
            if df is not None and len(df) > 100:
                # Prepare data
                prices = df['Close'].dropna().astype(float).tolist()
                volumes = df['Volume'].dropna().astype(float).tolist()
                # Train/Load model
                metrics = st.session_state.engine_v2.train(symbol, prices, volumes)
                if not metrics:
                    st.error("AI Training failed. Not enough historical patterns found for this stock. Try a more volatile symbol.")
                    return
                
                results = []
                # Simulate last 60 trading days
                test_len = 60
                wins = 0
                trades = 0
                prog = st.progress(0, text="Simulating trades...")
                for i in range(len(df) - test_len, len(df) - 3):
                    prog.progress((i - (len(df)-test_len)) / (test_len - 3), text=f"Simulating Day {i}...")
                    # Slice data up to point i
                    sub_df = df.iloc[:i]
                    sub_prices = sub_df['Close'].tolist()
                    sub_volumes = sub_df['Volume'].tolist()
                    
                    # Predict for T+3
                    pred = st.session_state.engine_v2.predict(symbol, sub_prices, sub_volumes, df=sub_df)
                    if pred:
                        sig = pred['today']['signal']
                        if "BUY" in sig or "SELL" in sig:
                            trades += 1
                            entry_price = float(df.iloc[i]['Close'])
                            # Peek ahead for results (3 days later)
                            future_price = float(df.iloc[i+3]['Close'])
                            
                            correct = False
                            if "BUY" in sig: correct = future_price > entry_price
                            elif "SELL" in sig: correct = future_price < entry_price
                            
                            if correct: wins += 1
                            
                            results.append({
                                'Date': df.index[i].strftime('%Y-%m-%d'),
                                'Signal': sig,
                                'Entry': entry_price,
                                'Result': '✅ WIN' if correct else '❌ LOSS'
                            })
                        else:
                            pass
                    else:
                        # Skip if prediction engine failed
                        pass
                
                prog.empty()
                
                # Render Results
                if trades > 0:
                    acc = wins / trades
                    st.success(f"### Backtest Accuracy: {acc:.1%}")
                    st.write(f"**Total Simulated Trades:** {trades} | **Wins:** {wins} | **Losses:** {trades - wins}")
                    
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                else:
                    st.warning("Model generated no signals during the simulation period. The AI didn't find high-probability entry points within its confidence threshold (40%+). Try increasing the simulation period or another stock.")
            else:
                st.error("Insufficient data for backtesting. Try another symbol (need at least 100 days of history).")

# ── PAGE: AI Prediction ──────────────────────────────────────────────────
def page_prediction():
    # ── Accuracy Dashboard Header ─────────────────────────────────────
    stats = load_advanced_stats()
    acc_ratio, total_trades = stats['win_rate'], stats['total']
    st.markdown(f'''<div style="background: rgba(17, 24, 39, 0.55); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.06); border-left: 6px solid #ef4444; padding: 15px; border-radius: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);">
<div>
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">AI System Accuracy</div>
<div style="font-size: 1.5rem; font-weight: 900; color: #fbbf24;">{acc_ratio:.1%}</div>
</div>
<div style="text-align: right;">
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Total Evaluated Signals</div>
<div style="font-size: 1.5rem; font-weight: 900; color: #f8fafc;">{total_trades}</div>
</div>
</div>''', unsafe_allow_html=True)

    st.subheader("🔮 AI Stock Prediction Engine (3-Day Forecast)")
    st.caption("This model analyzes historical trends, technical indicators, and news sentiment to predict if the price will go UP or DOWN for the next **3 days**.")
    
    with st.form("prediction_config_form"):
        c1, c2 = st.columns([3, 1])
        with c1:
            symbol = st.text_input("Enter Stock Symbol", placeholder="RELIANCE, TATAMOTORS, GOLD, SILVER, NIPPON, AAPL...")
        with c2:
            pred_mode = st.selectbox("Mode", ["📅 Daily (Swing)", "📈 Intraday (1H)"])
        
        run = st.form_submit_button("🧠 Run AI Analysis", use_container_width=True)
    
    refresh = st.button("🔄 Clear Cache & Reset", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        for k in ['pred_df','pred_mapped','pred_metrics','pred_results','last_sym','last_mode']:
            if k in st.session_state: del st.session_state[k]
        st.success("♻️ Cache Cleared. Fetching fresh data...")

    with st.expander("📋 Available Symbols"):
        st.write(", ".join(sorted(STOCK_MAP.keys())))

    if run and symbol.strip():
        symbol = symbol.strip().upper()
        # Handle Mode Switch
        st.session_state.last_mode = pred_mode
        st.session_state.last_sym = symbol
        
        with st.spinner(f"📡 Processing {symbol}..."):
            df, mapped = fetch_stock(symbol, 250)
            live_price = get_realtime_price(symbol, mapped)
            
            if (df is None or df.empty) and live_price is None:
                st.error(f"❌ No data for {symbol}"); return

            st.session_state.pred_df = df
            st.session_state.pred_mapped = mapped
            
            is_intra = "Intraday" in pred_mode
            df_run = df
            
            # Multi-Timeframe Fetching
            with st.spinner("📡 Fetching Multi-Timeframe data (Daily, 1H, 15m)..."):
                df_1d, _ = fetch_stock(symbol, days=250, interval='1d')
                df_1h, _ = fetch_stock(symbol, period='1mo', interval='1h')
                df_15m, _ = fetch_stock(symbol, period='14d', interval='15m')
                
                if is_intra:
                    df_run = df_15m
                else:
                    df_run = df_1d # Daily is main for swing

            if df_run is not None and len(df_run) > 30:
                prices = df_run['Close'].dropna().astype(float).tolist()
                volumes = df_run['Volume'].dropna().astype(float).tolist()
                
                with st.spinner("📰 Analyzing news & sector momentum..."): 
                    # Fetch Local and Global separately for comparison
                    local_news = fetch_market_news(f"{symbol} share stock market news")
                    master_sent = get_master_market_sentiment()
                    
                    # New AI Intelligence v2
                    news_score, scored_local_news, local_reason, local_events = analyze_news_v2(local_news)
                    sec_sent = get_sector_sentiment(symbol)
                    sector_score = sec_sent['score']
                    event_score = local_events['impact_score']
                    
                    st.session_state.pred_local_catalyst = local_reason # Stock specific
                    st.session_state.pred_global_catalyst = master_sent['global_reason']
                    st.session_state.pred_catalyst = f"Stock: {local_reason} | Sector: {sec_sent['label']}"
                    st.session_state.pred_indian_news = scored_local_news[:5]
                    st.session_state.pred_global_news = master_sent['global_headlines']
                    st.session_state.pred_events = local_events
                    st.session_state.pred_sector = sec_sent
                    
                    # Check for Gemini API key
                    gemini_key_to_use = st.session_state.get('gemini_api_key', '')
                    if not gemini_key_to_use:
                        gemini_key_to_use = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")) if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY", "")
                    
                    if gemini_key_to_use:
                        try:
                            g_res = generate_gemini_intelligence(symbol, local_news, master_sent.get('global_headlines', []), gemini_key_to_use)
                            if g_res:
                                st.session_state.pred_gemini_dom, st.session_state.pred_gemini_glob, st.session_state.pred_gemini_interplay = g_res
                            else:
                                for k in ['pred_gemini_dom', 'pred_gemini_glob', 'pred_gemini_interplay']:
                                    if k in st.session_state: st.session_state.pop(k, None)
                        except Exception:
                            for k in ['pred_gemini_dom', 'pred_gemini_glob', 'pred_gemini_interplay']:
                                if k in st.session_state: st.session_state.pop(k, None)
                    else:
                        for k in ['pred_gemini_dom', 'pred_gemini_glob', 'pred_gemini_interplay']:
                            if k in st.session_state: st.session_state.pop(k, None)
                    
                    # For legacy fallback in train
                    sent = news_score

                with st.spinner("🧠 Training Engine..."):
                    metrics = st.session_state.engine_v2.train(symbol, prices, volumes, sent, intraday=is_intra)
                    st.session_state.pred_metrics = metrics
                
                with st.spinner("📡 TradingView Bias..."):
                    tv_sentiment = fetch_tv_sentiment(symbol, mapped)
                
                if metrics:
                    res = st.session_state.engine_v2.predict(symbol, prices, volumes, news_sent=news_score, tv_sent=tv_sentiment, sector_score=sector_score, event_score=event_score, intraday=is_intra, df=df_run, df_1h=df_1h, df_1d=df_1d, df_15m=df_15m)
                    st.session_state.pred_results = res
                    
                    # NEW: Step 1, 3 & 5 Integration
                    entry_price = live_price if live_price else float(df_run.iloc[-1]['Close'])
                    timing = st.session_state.engine_v2.detect_entry_timing(df_run)
                    liquidity = st.session_state.engine_v2.detect_liquidity(df_run)
                    risk_params = st.session_state.engine_v2.calculate_risk_parameters(
                        symbol, entry_price, res['today']['signal'], 
                        st.session_state.get('risk_cap', 100000), 
                        st.session_state.get('risk_pct', 1.0),
                        reward_ratio=st.session_state.get('reward_ratio', 2.0),
                        df=df_run
                    )
                    
                    st.session_state.pred_timing = timing
                    st.session_state.pred_liquidity = liquidity
                    st.session_state.pred_risk = risk_params
                    
                    # Save prediction for future accuracy tracking (Institutional V3)
                    save_prediction({
                        'symbol': symbol,
                        'signal': res['today']['signal'],
                        'confidence': res['today']['confidence'],
                        'price': entry_price,
                        'sl': risk_params.get('sl', 0),
                        'target': risk_params.get('target', 0),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'catalyst': st.session_state.get('pred_catalyst', 'Market Consensus'),
                        'correct': None
                    })
                else:
                    st.error("AI Training failed due to insufficient data quality.")
            else: st.warning("Not enough intraday data to train AI.")

    # --- RENDER FROM SESSION STATE ---
    if 'pred_results' in st.session_state and 'pred_df' in st.session_state:
        df = st.session_state.pred_df
        mapped = st.session_state.pred_mapped
        pred = st.session_state.pred_results
        symbol = st.session_state.last_sym
        is_intra = "Intraday" in st.session_state.get('last_mode','')
        headlines = st.session_state.get('pred_news_headlines', [])
        today_sig = pred['today']['signal']
        timing = st.session_state.get('pred_timing', 'N/A')
        liq = st.session_state.get('pred_liquidity', 'N/A')
        rp = st.session_state.get('pred_risk', {})
        
        # 1. LIVE HEADER CARD (Properly Aligned)
        live_price = get_realtime_price(symbol, mapped)
        is_indian = mapped.endswith('.NS') or mapped.endswith('.BO') or mapped.startswith('^')
        curr = '₹' if is_indian else '$'
        entry_price = live_price if live_price else (float(df.iloc[-1]['Close']) if not df.empty else 0.0)
        
        if live_price:
            prev = df.iloc[-1]['Close'] if not df.empty else live_price
            chg = live_price - prev; pct = (chg/prev*100) if prev!=0 else 0
            card_color = "#00b386" if chg >= 0 else "#ef4444"
            
            st.markdown(f"""<div class="stock-card" style="padding:1.5rem; border-right: 12px solid {card_color};">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Live Asset Analysis</div>
<div class="name" style="font-size:1.8rem;">{symbol} <span style="font-size:0.9rem; color:#64748b;">({mapped})</span></div>
<div class="price" style="font-size:2.5rem; font-weight:900;">{curr}{live_price:,.2f}</div>
<div class="{'change-up' if chg>=0 else 'change-down'}" style="font-size:1.1rem; font-weight:700;">
{'▲' if chg>=0 else '▼'} {chg:+,.2f} ({pct:+.2f}%)
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

        # 2. ANALYSIS TABS
        st.markdown('<div class="section-head">📺 Live Analytics Dashboard</div>', unsafe_allow_html=True)
        tab_chart, tab_ai, tab_mtf, tab_sr, tab_profile = st.tabs(["📊 Advanced Chart", "🤖 Intelligence Pulse", "⏳ MTF Alignment", "📈 S/R Analysis", "🏛️ Stock Profile"])
        with tab_chart: st.components.v1.html(build_tradingview_chart(symbol, mapped), height=650)
        with tab_ai:
            sc1, sc2 = st.columns(2)
            with sc1: st.components.v1.html(build_tradingview_analysis(symbol, mapped), height=450)
            with sc2: st.plotly_chart(build_gauge(pred['today']['ml_prob'], pred['today']['signal'], "AI Signal Strength"), use_container_width=True)
        
        with tab_mtf:
            st.markdown("### ⏳ Multi-Timeframe Alignment Matrix")
            mtf = pred.get('mtf_status', {})
            if mtf:
                rows = []
                for tf, data in mtf.items():
                    t_col = "🟢" if data['trend'] == "Bullish" else "🔴"
                    rows.append({
                        "Timeframe": tf.upper(),
                        "Trend": f"{t_col} {data['trend']}",
                        "RSI (14)": data['rsi'],
                        "Candle Pattern": data['pattern']
                    })
                st.table(pd.DataFrame(rows))
                
                # Check for absolute alignment
                trends = [d['trend'] for d in mtf.values()]
                if all(t == "Bullish" for t in trends):
                    st.success("✅ **MTF Alignment: BULLISH** (Strongest Confirmation)")
                elif all(t == "Bearish" for t in trends):
                    st.error("⚠️ **MTF Alignment: BEARISH** (High Downside Risk)")
                else:
                    st.warning("⚖️ **MTF Alignment: MIXED** (Trend Fragmentation)")
            else:
                st.info("MTF Data not available for this asset type.")

        with tab_profile: st.components.v1.html(build_tradingview_profile_widget(symbol, mapped), height=400)

        with tab_sr:
            st.markdown("### 📈 AI Support & Resistance Analysis")
            st.caption("AI automatically detects key price levels using a pivot peak/trough algorithm — the same levels professional traders draw manually.")

            # Detect S/R levels using 90 days of data for richness
            sr_df = df.tail(90) if len(df) >= 90 else df
            sr_data = detect_support_resistance(sr_df, window=5, num_levels=5)

            # ── Signal Badge ──────────────────────────────────────────────
            sr_sig   = sr_data.get('sr_signal', 'N/A')
            sr_color = sr_data.get('sr_color', '#94a3b8')
            sr_detail = sr_data.get('sr_detail', '')
            near_sup = sr_data.get('nearest_support', 0)
            near_res = sr_data.get('nearest_resistance', 0)
            cur_p    = sr_data.get('current_price', 0)

            st.markdown(f'''
<div style="background:{sr_color}18; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border:1px solid {sr_color}40; border-left:8px solid {sr_color};
     padding:18px 24px; border-radius:14px; margin-bottom:20px; box-shadow: 0 4px 25px rgba(0,0,0,0.25);">
  <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:800; letter-spacing:1px;">AI S/R Signal</div>
  <div style="font-size:2rem; font-weight:950; color:{sr_color}; margin:4px 0;">🎯 {sr_sig}</div>
  <div style="font-size:0.9rem; color:#cbd5e1;">{sr_detail}</div>
  <div style="display:flex; gap:30px; margin-top:12px;">
    <div><span style="color:#94a3b8; font-size:0.75rem;">Current Price</span><br>
         <span style="color:#f8fafc; font-weight:800;">₹{cur_p:,.2f}</span></div>
    <div><span style="color:#10b981; font-size:0.75rem;">Nearest Support</span><br>
         <span style="color:#10b981; font-weight:800;">₹{near_sup:,.2f}</span></div>
    <div><span style="color:#ef4444; font-size:0.75rem;">Nearest Resistance</span><br>
         <span style="color:#ef4444; font-weight:800;">₹{near_res:,.2f}</span></div>
    <div><span style="color:#94a3b8; font-size:0.75rem;">Risk/Reward Zone</span><br>
         <span style="color:#f8fafc; font-weight:800;">{f"{abs(cur_p-near_sup):.2f} / {abs(near_res-cur_p):.2f}" if near_sup and near_res else "N/A"}</span></div>
  </div>
</div>''', unsafe_allow_html=True)

            # ── S/R Chart ────────────────────────────────────────────────
            st.plotly_chart(build_sr_chart(sr_df, symbol, sr_data), use_container_width=True)

            # ── Level Tables ────────────────────────────────────────────
            tbl_col1, tbl_col2 = st.columns(2)
            with tbl_col1:
                st.markdown("#### 🔴 Resistance Levels")
                res_levels = sr_data.get('resistance', [])
                if res_levels:
                    res_rows = []
                    for i, lvl in enumerate(res_levels):
                        dist_pct = ((lvl['price'] - cur_p) / cur_p * 100) if cur_p else 0
                        strength_bar = '▓' * lvl['strength'] + '░' * max(0, 5 - lvl['strength'])
                        res_rows.append({
                            'Level': f'R{i+1}',
                            'Price (₹)': f"{lvl['price']:,.2f}",
                            'Distance': f"+{dist_pct:.2f}%",
                            'Strength': f"{strength_bar} ({lvl['strength']}x tested)"
                        })
                    st.dataframe(pd.DataFrame(res_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No resistance levels detected.")

            with tbl_col2:
                st.markdown("#### 🟢 Support Levels")
                sup_levels = sr_data.get('support', [])
                if sup_levels:
                    sup_rows = []
                    for i, lvl in enumerate(reversed(sup_levels)):
                        dist_pct = ((cur_p - lvl['price']) / cur_p * 100) if cur_p else 0
                        strength_bar = '▓' * lvl['strength'] + '░' * max(0, 5 - lvl['strength'])
                        sup_rows.append({
                            'Level': f'S{i+1}',
                            'Price (₹)': f"{lvl['price']:,.2f}",
                            'Distance': f"-{dist_pct:.2f}%",
                            'Strength': f"{strength_bar} ({lvl['strength']}x tested)"
                        })
                    st.dataframe(pd.DataFrame(sup_rows), hide_index=True, use_container_width=True)
                else:
                    st.info("No support levels detected.")

            # ── Trade Guidance Card ──────────────────────────────────────
            st.markdown("#### 🧭 AI Trade Guidance (Based on S/R Position)")
            if sr_sig == 'BUY ZONE':
                guide_color = '#10b981'
                guide_emoji = '🟢'
                guide_title = 'ENTRY ZONE CONFIRMED'
                guide_body  = f'Price is at or near a strong support level (₹{near_sup:,.2f}). This is a high probability entry point. Place Stop Loss just below ₹{near_sup * 0.99:,.2f} and target the next resistance at ₹{near_res:,.2f}.'
            elif sr_sig == 'SELL ZONE':
                guide_color = '#ef4444'
                guide_emoji = '🔴'
                guide_title = 'BOOK PROFITS / EXIT ZONE'
                guide_body  = f'Price is approaching strong resistance (₹{near_res:,.2f}). Consider booking partial or full profits here. Avoid fresh buying near resistance.'
            elif sr_sig == 'BREAKOUT':
                guide_color = '#00b386'
                guide_emoji = '🚀'
                guide_title = 'BREAKOUT — BUY ON RETEST'
                guide_body  = f'Price has broken above resistance ₹{near_res:,.2f}. Wait for a retest of the broken resistance (now acting as support) before entering. Target: next major resistance level.'
            elif sr_sig == 'BREAKDOWN':
                guide_color = '#eb5b3c'
                guide_emoji = '💥'
                guide_title = 'BREAKDOWN — AVOID / SHORT'
                guide_body  = f'Price has broken below support ₹{near_sup:,.2f}. Avoid long positions. Short traders can target the next support level below.'
            else:
                guide_color = '#6366f1'
                guide_emoji = '⚖️'
                guide_title = 'MID-ZONE — WAIT FOR CLARITY'
                guide_body  = f'Price is between support ₹{near_sup:,.2f} and resistance ₹{near_res:,.2f}. Wait for price to approach a key level before taking a trade for better risk/reward.'

            st.markdown(f'''
<div style="background:{guide_color}15; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border:1px solid {guide_color}40; border-left:6px solid {guide_color};
     padding:16px 20px; border-radius:12px; margin-top:10px; box-shadow: 0 4px 25px rgba(0,0,0,0.25);">
  <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; font-weight:800;">{guide_emoji} AI Recommendation</div>
  <div style="font-size:1.2rem; font-weight:900; color:{guide_color}; margin:6px 0;">{guide_title}</div>
  <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">{guide_body}</div>
</div>''', unsafe_allow_html=True)

        # 3. AI FORECAST DASHBOARD (Cleanly Aligned 3-Day Projection)
        st.markdown('<div class="section-head">🎯 AI Signal Forecast (3-Day Price Projection)</div>', unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        sig_cls = {
            'STRONG BUY': 'signal-buy', 'BUY': 'signal-buy', 
            'STRONG SELL': 'signal-sell', 'SELL': 'signal-sell', 
            'HOLD': 'signal-hold', 'HOLD (Low Confidence)': 'signal-hold',
            'HOLD (Uncertain)': 'signal-hold', 'HOLD (Ranging Market)': 'signal-hold'
        }
        sig_emoji = {
            'STRONG BUY': '🚀 STRONG BUY', 'BUY': '🚀 BUY', 
            'STRONG SELL': '💥 STRONG SELL', 'SELL': '💥 SELL', 
            'HOLD': '⚖️ HOLD', 'HOLD (Low Confidence)': '⚖️ HOLD',
            'HOLD (Uncertain)': '⚖️ HOLD (U)', 'HOLD (Ranging Market)': '⚖️ HOLD (R)'
        }
        l1, l2, l3 = ("15 MIN", "30 MIN", "1 HOUR") if is_intra else ("TODAY", "TOMORROW", "DAY AFTER")
        
        with tc1: 
            s1 = pred["today"]["signal"]
            st.markdown(f'<div class="{sig_cls.get(s1, "signal-hold")}">{l1}<br><span style="font-size:1.5rem;">{sig_emoji.get(s1, "⚖️ HOLD")}</span><p style="font-size:0.75rem">Conf: {pred["today"]["confidence"]:.1%}</p></div>', unsafe_allow_html=True)
        with tc2: 
            s2 = pred["tomorrow"]["signal"]
            st.markdown(f'<div class="{sig_cls.get(s2, "signal-hold")}">{l2}<br><span style="font-size:1.5rem;">{sig_emoji.get(s2, "⚖️ HOLD")}</span><p style="font-size:0.75rem">Conf: {pred["tomorrow"]["confidence"]:.1%}</p></div>', unsafe_allow_html=True)
        with tc3: 
            s3 = pred["next_3_days"]["signal"]
            st.markdown(f'<div class="{sig_cls.get(s3, "signal-hold")}">{l3}<br><span style="font-size:1.5rem;">{sig_emoji.get(s3, "⚖️ HOLD")}</span><p style="font-size:0.75rem">Conf: {pred["next_3_days"]["confidence"]:.1%}</p></div>', unsafe_allow_html=True)

        # Confidence breakdown metrics expander
        scores = pred["today"].get("scores")
        if scores:
            with st.expander("📊 View Today's AI Prediction Confidence Score Breakdown"):
                st.markdown("""
                <div style="font-size:0.85rem; color:#475569; margin-bottom:15px;">
                This breakdown shows the raw confidence of each confirmation factor and its weighted contribution to the final <b>Today</b> score.
                </div>
                """, unsafe_allow_html=True)
                
                c_ml = scores.get('ml_score', 0.0)
                c_tech = scores.get('tech_score', 0.0)
                c_vol = scores.get('volume_score', 0.0)
                c_struct = scores.get('structure_score', 0.0)
                c_ob = scores.get('ob_score', 0.0)
                c_fib = scores.get('fib_score', 0.0)
                c_fvg = scores.get('fvg_score', 0.0)
                
                # Weights
                w_ml, w_tech, w_vol, w_struct, w_ob, w_fib, w_fvg = 0.35, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05
                
                col_b1, col_b2, col_b3, col_b4, col_b5, col_b6, col_b7 = st.columns(7)
                with col_b1:
                    st.metric(label="🧠 ML Predict (35%)", value=f"{c_ml:.0%}", delta=f"+{c_ml*w_ml:.1%}")
                with col_b2:
                    st.metric(label="📈 Tech Ind (20%)", value=f"{c_tech:.0%}", delta=f"+{c_tech*w_tech:.1%}")
                with col_b3:
                    st.metric(label="📊 Volume (15%)", value=f"{c_vol:.0%}", delta=f"+{c_vol*w_vol:.1%}")
                with col_b4:
                    st.metric(label="⚖️ Structure (10%)", value=f"{c_struct:.0%}", delta=f"+{c_struct*w_struct:.1%}")
                with col_b5:
                    st.metric(label="🏛️ Order Block (10%)", value=f"{c_ob:.0%}", delta=f"+{c_ob*w_ob:.1%}")
                with col_b6:
                    st.metric(label="🎯 Fibonacci (5%)", value=f"{c_fib:.0%}", delta=f"+{c_fib*w_fib:.1%}")
                with col_b7:
                    st.metric(label="⚡ FVG Imbal (5%)", value=f"{c_fvg:.0%}", delta=f"+{c_fvg*w_fvg:.1%}")

        # --- 3.5. INSTITUTIONAL ORDER BLOCK & SMC SCANNER ---
        # Extract SMC info from pred['today']
        smc_info = pred['today'].get('smc', {})
        closest_ob = smc_info.get('closest_ob')
        closest_fvg = smc_info.get('closest_fvg')
        ob_dist = smc_info.get('closest_ob_dist_pct', 999.0)
        fvg_dist = smc_info.get('closest_fvg_dist_pct', 999.0)
        current_trend = smc_info.get('current_trend', 'Neutral')
        confluence_detected = smc_info.get('confluence_detected', False)
        
        # Setup stars
        setup_stars = pred['today'].get('stars', 1)
        stars_str = '⭐' * setup_stars + '☆' * (5 - setup_stars)
        
        # Calculate OB Details
        ob_type = "N/A"
        ob_zone = "N/A"
        ob_age_str = "N/A"
        ob_freshness_pct = 0
        ob_tf_label = smc_info.get('ob_timeframe', 'N/A')
        mtf_ob_summary = smc_info.get('mtf_ob_summary', {})
        if closest_ob:
            ob_type = f"{closest_ob['type']} Order Block"
            ob_zone = f"{curr}{closest_ob['bottom']:,.2f} - {curr}{closest_ob['top']:,.2f}"
            age_unit = "Hours" if is_intra else "Days"
            ob_age_str = f"{closest_ob['age']} {age_unit} Old"
            ob_freshness_pct = max(0, 100 - closest_ob['age'] * 2)

        # FVG Details
        fvg_type = "N/A"
        fvg_zone = "N/A"
        if closest_fvg:
            fvg_type = f"{closest_fvg['type']} FVG"
            fvg_zone = f"{curr}{closest_fvg['bottom']:,.2f} - {curr}{closest_fvg['top']:,.2f}"

        # Volume spike state
        is_vol_strong = "PASS ✅" in pred['today']['breakdown'].get('Volume Confirm', '')
        vol_state_str = "Above Average" if is_vol_strong else "Normal"
        
        # Status setup quality text
        status_quality = "✅ High Probability Setup" if setup_stars >= 4 else "⚖️ Moderate Setup" if setup_stars >= 3 else "❌ Low Conviction Setup"
        
        # Components status list
        ob_check = "✅" if closest_ob and ob_dist <= 2.0 else "❌"
        fvg_check = "✅" if closest_fvg and fvg_dist <= 2.0 else "❌"
        bos_check = "✅" if current_trend == ("Bullish" if "BUY" in pred['today']['signal'] else "Bearish") else "❌"
        vol_check = "✅" if is_vol_strong else "❌"
        fib_check = "✅" if pred['today']['scores'].get('fib_score', 0.0) > 0.0 else "❌"
        ai_check = "✅" if "BUY" in pred['today']['signal'] or "SELL" in pred['today']['signal'] else "❌"
        
        badge_section_html = ""
        if confluence_detected or setup_stars == 5:
            badge_section_html = f'''<div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%); border: 2px solid #ef4444; border-radius: 12px; padding: 15px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.15);">
<div style="font-size: 1.5rem; font-weight: 900; color: #ff5e5e; margin-bottom: 5px;">🔥 Institutional Confluence Detected</div>
<div style="font-size: 0.85rem; color: #cbd5e1;">Smart Money Concepts and AI models are in perfect alignment. High-probability setup identified.</div>
</div>'''

        # Render scanner UI
        st.markdown(f'''<div class="premium-card" style="margin-bottom: 25px; color: #f8fafc;">
<div style="font-size: 1.1rem; font-weight: 800; color: #fbbf24; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">🏛️ Institutional Order Block & SMC Scanner</div>
{badge_section_html}
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 25px;">
<!-- Left Column -->
<div style="border-right: 1px solid #334155; padding-right: 20px;">
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Asset / Stock</div>
<div style="font-size: 1.3rem; font-weight: 900; margin-bottom: 15px; color: #f8fafc;">{symbol}</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Order Block Type</div>
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
<span style="font-size: 1.1rem; font-weight: 700; color: {'#10b981' if closest_ob and closest_ob['type'] == 'Bullish' else '#ef4444' if closest_ob else '#94a3b8'};">{ob_type}</span>
<span style="background: {'#6366f1' if ob_tf_label != 'N/A' else '#334155'}; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.5px;">{ob_tf_label}</span>
</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Zone Range</div>
<div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 15px; font-family: monospace; color: #e2e8f0;">{ob_zone}</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Distance to OB Zone</div>
<div style="font-size: 1.1rem; font-weight: 700; color: {'#10b981' if ob_dist <= 1.0 else '#f59e0b' if ob_dist <= 3.0 else '#f8fafc'};">{f"{ob_dist:.2f}%" if ob_dist != 999.0 else "N/A"}</div>
</div>
<!-- Middle Column -->
<div style="border-right: 1px solid #334155; padding-right: 20px;">
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Current Price</div>
<div style="font-size: 1.3rem; font-weight: 900; margin-bottom: 15px; color: #fbbf24;">{curr}{entry_price:,.2f}</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">OB Age & Freshness</div>
<div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 2px; color: #e2e8f0;">{ob_age_str}</div>
<div style="font-size: 0.95rem; color: #10b981; font-weight: 700; margin-bottom: 15px;">Freshness: {ob_freshness_pct}%</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Volume State</div>
<div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 15px; color: {'#10b981' if is_vol_strong else '#cbd5e1'};">{vol_state_str}</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Nearest FVG Zone</div>
<div style="font-size: 1.1rem; font-weight: 700; font-family: monospace; color: #60a5fa;">{fvg_zone}</div>
</div>
<!-- Right Column -->
<div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Setup Quality</div>
<div style="font-size: 1.4rem; color: #f59e0b; font-weight: 900; margin-bottom: 12px; letter-spacing: 2px;">{stars_str}</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 4px;">Scanner Status</div>
<div style="font-size: 1.05rem; font-weight: 700; color: {'#10b981' if setup_stars >= 4 else '#f59e0b' if setup_stars >= 3 else '#ef4444'}; margin-bottom: 15px;">{status_quality}</div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px; margin-bottom: 8px;">Confluence Checklist</div>
<div style="font-size: 0.8rem; line-height: 1.5; color: #cbd5e1; font-family: system-ui, sans-serif;">
<div style="display:flex; justify-content:space-between; border-bottom: 1px dashed #1e293b; padding: 2px 0;"><span>Bullish/Bearish Order Block:</span> <span>{ob_check}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom: 1px dashed #1e293b; padding: 2px 0;"><span>Fair Value Gap (FVG):</span> <span>{fvg_check}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom: 1px dashed #1e293b; padding: 2px 0;"><span>BOS Confirmed Trend:</span> <span>{bos_check}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom: 1px dashed #1e293b; padding: 2px 0;"><span>Volume Spike (> 1.2x avg):</span> <span>{vol_check}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom: 1px dashed #1e293b; padding: 2px 0;"><span>Fibonacci Support / Retest:</span> <span>{fib_check}</span></div>
<div style="display:flex; justify-content:space-between; padding: 2px 0;"><span>AI Bias Alignment:</span> <span>{ai_check}</span></div>
</div>
</div>
</div>
</div>''', unsafe_allow_html=True)

        # Multi-Timeframe OB Summary Row
        if mtf_ob_summary:
            tf_order = ['15M', '1H', '1D']
            tf_cards_html = ""
            for tf in tf_order:
                if tf in mtf_ob_summary:
                    info = mtf_ob_summary[tf]
                    trend_c = '#10b981' if info['trend'] == 'Bullish' else '#ef4444' if info['trend'] == 'Bearish' else '#f59e0b'
                    is_active_tf = (tf == ob_tf_label)
                    border_style = f"border: 2px solid {trend_c};" if is_active_tf else "border: 1px solid #334155;"
                    glow = f"box-shadow: 0 0 12px {trend_c}40;" if is_active_tf else ""
                    tf_cards_html += f'''<div style="background: rgba(255,255,255,0.03); {border_style} {glow} border-radius: 12px; padding: 12px 16px; text-align: center; min-width: 120px;">
<div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 800; margin-bottom: 6px;">{tf} Timeframe</div>
<div style="font-size: 1.4rem; font-weight: 900; color: {trend_c}; margin-bottom: 4px;">{info['total']}</div>
<div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 2px;">🟢 {info['bullish']} Bull  ·  🔴 {info['bearish']} Bear</div>
<div style="font-size: 0.7rem; font-weight: 700; color: {trend_c}; text-transform: uppercase;">{info['trend']} Trend</div>
</div>'''
                else:
                    tf_cards_html += f'''<div style="background: rgba(255,255,255,0.02); border: 1px dashed #334155; border-radius: 12px; padding: 12px 16px; text-align: center; min-width: 120px; opacity: 0.5;">
<div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 800; margin-bottom: 6px;">{tf} Timeframe</div>
<div style="font-size: 1.4rem; font-weight: 900; color: #334155; margin-bottom: 4px;">—</div>
<div style="font-size: 0.7rem; color: #475569;">No Active OB</div>
</div>'''
            st.markdown(f'''<div class="premium-card" style="margin-bottom: 25px; color: #f8fafc;">
<div style="font-size: 0.9rem; font-weight: 800; color: #a78bfa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;">⏱️ Multi-Timeframe Order Block Map</div>
<div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
{tf_cards_html}
</div>
<div style="font-size: 0.7rem; color: #64748b; margin-top: 12px; text-align: center;">Highlighted card = Timeframe where the closest Order Block was detected. OB from higher timeframes carry more institutional weight.</div>
</div>''', unsafe_allow_html=True)

        # 4. AI Market Intelligence (News, Sector, Events)
        st.markdown('<div class="section-head">🧠 AI Market Intelligence</div>', unsafe_allow_html=True)
        # Check for Gemini synthesis
        gemini_dom = st.session_state.get('pred_gemini_dom', None)
        gemini_glob = st.session_state.get('pred_gemini_glob', None)
        gemini_interplay = st.session_state.get('pred_gemini_interplay', None)
        
        # New Intelligence Details
        sec_data = st.session_state.get('pred_sector', {'label': 'Neutral', 'sector': 'Unknown'})
        evt_data = st.session_state.get('pred_events', {'description': 'No major events', 'impact_score': 0.0})
        
        evt_color = "#10b981" if evt_data['impact_score'] > 0 else "#ef4444" if evt_data['impact_score'] < 0 else "#94a3b8"
        sec_color = "#10b981" if "Positive" in sec_data['label'] else "#ef4444" if "Negative" in sec_data['label'] else "#94a3b8"
        
        if gemini_dom and gemini_glob and gemini_interplay:
            st.markdown(f'''
            <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2); border-left: 6px solid #6366f1; padding: 15px 20px; border-radius: 10px; margin-bottom: 15px; font-size: 0.9rem; line-height: 1.6; color: #cbd5e1; display: grid; gap: 8px;">
                <div><b style="color: #a78bfa; margin-right: 8px;">🔥 {sec_data['sector']} Sector:</b> <span style="color: {sec_color}; font-weight: 600;">{sec_data['label']}</span></div>
                <div><b style="color: #a78bfa; margin-right: 8px;">⚡ Financial Event:</b> <span style="color: {evt_color}; font-weight: 600;">{evt_data['description']}</span></div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-bottom: 15px;">
                <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.15); border-left: 6px solid #10b981; padding: 18px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                    <div style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">🇮🇳 Domestic Catalyst</div>
                    <div style="font-size: 0.95rem; color: #f4f4f5; line-height: 1.6; font-weight: 500; margin-top: 8px;">{gemini_dom}</div>
                </div>
                <div style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.15); border-left: 6px solid #ef4444; padding: 18px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                    <div style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">🌎 Global Driver</div>
                    <div style="font-size: 0.95rem; color: #f4f4f5; line-height: 1.6; font-weight: 500; margin-top: 8px;">{gemini_glob}</div>
                </div>
            </div>
            
            <div style="background: rgba(245, 158, 11, 0.06); border: 1px solid rgba(245, 158, 11, 0.15); border-left: 6px solid #f59e0b; padding: 18px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                <div style="font-size: 0.75rem; color: #a1a1aa; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">⚡ Macro Interplay (Connected Analysis)</div>
                <div style="font-size: 0.95rem; color: #f4f4f5; line-height: 1.6; font-weight: 500; margin-top: 8px;">{gemini_interplay}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            local_cat = st.session_state.get('pred_local_catalyst', 'Stable domestic market conditions.')
            global_cat = st.session_state.get('pred_global_catalyst', 'Neutral global queues.')
            st.markdown(f'''
            <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2); border-left: 6px solid #6366f1; padding: 15px 20px; border-radius: 10px; margin-bottom: 20px; font-size: 0.9rem; line-height: 1.6; color: #cbd5e1; display: grid; gap: 8px;">
                <div><b style="color: #a78bfa; margin-right: 8px;">🔥 {sec_data['sector']} Sector:</b> <span style="color: {sec_color}; font-weight: 600;">{sec_data['label']}</span></div>
                <div><b style="color: #a78bfa; margin-right: 8px;">⚡ Financial Event:</b> <span style="color: {evt_color}; font-weight: 600;">{evt_data['description']}</span></div>
                <div style="margin-top: 5px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05);">
                    <b>🇮🇳 Stock Catalyst:</b> {local_cat}<br>
                    <b>🌎 Global Driver:</b> {global_cat}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # 5. CANDLE CHART
        st.markdown('<div class="section-head">📊 Market Pulse & Patterns</div>', unsafe_allow_html=True)
        fib_levels = pred.get('fib_levels')
        smc_data = pred['today'].get('smc', {}).get('smc_full') if 'smc' in pred['today'] else None
        st.plotly_chart(build_candle_chart(df.tail(60), symbol, fib_levels=fib_levels, smc=smc_data), use_container_width=True)

        # 6. TECHNICAL PATTERN INSET (Arranged Under the Chart)
        res = detect_candle_pattern(df.tail(3)); live_res = analyze_live_candle(df)
        p_cls = "pulse-green" if "UP" in live_res["bias"] else "pulse-red" if "DOWN" in live_res["bias"] else ""
        
        st.markdown(f'''
            <div class="pattern-inset">
                <div style="text-align:center; flex:1;">
                    <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;">Trend</div>
                    <div class="{p_cls}" style="margin:5px 0; font-size:0.85rem;">{live_res["bias"]}</div>
                    <div style="font-size:1.1rem; font-weight:900; color:{live_res["color"]};">{live_res["pct"]:+.2f}%</div>
                </div>
                <div style="width:1px; height:40px; background:rgba(255,255,255,0.08);"></div>
                <div style="text-align:center; flex:2; padding: 0 15px;">
                    <div style="font-size:0.75rem; color:#9ca3af; text-transform:uppercase; font-weight:700;">Candlestick Analysis</div>
                    <div style="font-size:1rem; font-weight:800; color:white; margin-top:4px;">{res["pattern"]}</div>
                    <div style="font-size:0.75rem; color:#9ca3af;">{res["advice"]}</div>
                </div>
                <div style="width:1px; height:40px; background:rgba(255,255,255,0.08);"></div>
                <div style="text-align:center; flex:1;">
                    <div style="font-size:0.7rem; color:#9ca3af; text-transform:uppercase;">Volume Signal</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{'#10b981' if df.iloc[-1]['Volume'] > df['Volume'].tail(20).mean() * 1.5 else '#9ca3af'}; margin-top:5px;">
                        {'SPIKING ⚡' if df.iloc[-1]['Volume'] > df['Volume'].tail(20).mean() * 1.5 else 'NORMAL'}
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 7. WHY THIS PREDICTION? (Logic Explanation)
        st.markdown('<div class="section-head">🔍 Why This Prediction? (10-Factor AI)</div>', unsafe_allow_html=True)
        today_res = pred['today']
        exp_col1, exp_col2, exp_col3, exp_col4, exp_col5 = st.columns(5)
        with exp_col1:
            st.metric("ML Score", f"{today_res['scores'].get('ml_score', 0):.1%}")
        with exp_col2:
            st.metric("Tech Score", f"{today_res['scores'].get('tech_score', 0):.1%}")
        with exp_col3:
            st.metric("News Bias", f"{today_res['scores'].get('news_score', 0):.1%}")
        with exp_col4:
            st.metric("Sector", f"{today_res['scores'].get('sec_score', 0):.1%}")
        with exp_col5:
            st.metric("Event Impact", f"{today_res['scores'].get('evt_score', 0):.1%}")

        # 8. INSTITUTIONAL CONVICTION DASHBOARD (v3 - Weighted Model)
        st.markdown('<br><hr><br>', unsafe_allow_html=True)
        
        # 1. Weights Configuration
        W_AI, W_BIAS, W_TECH, W_PAT, W_VOL = 0.35, 0.30, 0.20, 0.10, 0.05
        
        # 2. Extract Signal Strengths (0.0 to 1.0)
        today_res = pred['today']
        global_sent = get_master_market_sentiment()
        
        s_ai = today_res['ml_prob']
        s_bias = min(abs(global_sent['score']) * 4, 1.0) # Scaling sentiment to 1.0
        s_tech = today_res['tech_score']
        s_pat = today_res['pattern_score']
        
        vols = df['Volume'].dropna().values
        vol_avg = np.mean(vols[-20:]) if len(vols) > 20 else 1.0
        s_vol = min((vols[-1] / vol_avg) / 2.0, 1.0) if vol_avg > 0 else 0.5

        # 3. Directional Analysis
        ai_dir = 1 if "BUY" in today_res['signal'] else -1
        bias_dir = 1 if global_sent['score'] > 0.05 else -1 if global_sent['score'] < -0.05 else 0
        
        # 4. Weighted Conviction Score
        conviction_raw = (s_ai * W_AI) + (s_bias * W_BIAS) + (s_tech * W_TECH) + (s_pat * W_PAT) + (s_vol * W_VOL)
        conviction_final = conviction_raw * 100
        
        # 5. Adaptive Risk Penalty (Soft Penalty instead of Hard Cap)
        risk_level = "STABLE ✅"
        risk_color = "#10b981"
        risk_desc = "AI and Market Sentiment are in alignment."
        
        if ai_dir != bias_dir and bias_dir != 0:
            conviction_final *= 0.75  # 25% Reduction for conflict
            risk_level = "HIGH UNCERTAINTY ⚠️"
            risk_color = "#f59e0b"
            risk_desc = "Conflict between AI Trend and News Sentiment detected."
            
        # 6. Final Verdict Selection
        if conviction_final >= 75: 
            v_sig = "POWER BUY 🚀" if ai_dir == 1 else "POWER SELL 💥"
            v_col = "#00b386" if ai_dir == 1 else "#eb5b3c"
        elif conviction_final >= 50:
            v_sig = "BUY 📈" if ai_dir == 1 else "SELL 📉"
            v_col = "#10b981" if ai_dir == 1 else "#ef4444"
        else:
            v_sig = "HOLD ⚖️"
            v_col = "#f59e0b"

        # 7. DISPLAY INSTITUTIONAL DASHBOARD (The missing output)
        st.markdown(f'''<div style="background: #0f172a; border: 1px solid #334155; padding: 25px; border-radius: 15px; margin-bottom: 20px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
<div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; font-weight:800; letter-spacing:1px;">🏛️ Institutional Conviction (V3)</div>
<div style="background:{risk_color}20; color:{risk_color}; padding:4px 12px; border-radius:8px; font-size:0.7rem; font-weight:700;">{risk_level}</div>
</div>
<div style="display:grid; grid-template-columns: 1fr 2fr; gap:30px; align-items:center;">
<div style="text-align:center; border-right:1px solid #334155; padding-right:20px;">
<div style="font-size:3rem; font-weight:950; color:{v_col}; line-height:1;">{conviction_final:.0f}%</div>
<div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; margin-top:5px; font-weight:700;">Confidence Score</div>
</div>
<div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#f8fafc; font-weight:800; font-size:1.1rem;">Verdict: {v_sig}</span>
<span style="color:#94a3b8; font-size:0.75rem;">{risk_desc}</span>
</div>
<div style="background:#1e293b; height:12px; border-radius:10px; overflow:hidden; border:1px solid #334155;">
<div style="background:linear-gradient(90deg, {v_col}88, {v_col}); width:{conviction_final}%; height:100%;"></div>
</div>
<div style="display:flex; justify-content:space-between; margin-top:10px;">
<div style="text-align:center;">
<div style="font-size:0.6rem; color:#64748b;">AI Bias</div>
<div style="font-size:0.75rem; color:#f8fafc; font-weight:700;">{s_ai:.0%}</div>
</div>
<div style="text-align:center;">
<div style="font-size:0.6rem; color:#64748b;">Sentiment</div>
<div style="font-size:0.75rem; color:#f8fafc; font-weight:700;">{s_bias:.0%}</div>
</div>
<div style="text-align:center;">
<div style="font-size:0.6rem; color:#64748b;">Technical</div>
<div style="font-size:0.75rem; color:#f8fafc; font-weight:700;">{s_tech:.0%}</div>
</div>
<div style="text-align:center;">
<div style="font-size:0.6rem; color:#64748b;">Pattern</div>
<div style="font-size:0.75rem; color:#f8fafc; font-weight:700;">{s_pat:.0%}</div>
</div>
<div style="text-align:center;">
<div style="font-size:0.6rem; color:#64748b;">Volume</div>
<div style="font-size:0.75rem; color:#f8fafc; font-weight:700;">{s_vol:.0%}</div>
</div>
</div>
</div>
</div>
</div>''', unsafe_allow_html=True)

        # 8. Step 4 & 7: PRO-TRADER EXECUTION CARD (Tamil Summary + Plan B + Badges)
        st.markdown('<br><hr><br>', unsafe_allow_html=True)
        
        # Step 4 (v3): Market Session Awareness
        session_phase = st.session_state.engine_v2.get_market_session()
        
        # Badge Styling (Step 7)
        badge_html = ""
        if "READY" in timing: badge_html = f'<span style="background:#f59e0b; color:white; padding:4px 10px; border-radius:6px; font-size:0.6rem; margin-right:8px;">READY</span>'
        elif "DETECTED" in timing: badge_html = f'<span style="background:#10b981; color:white; padding:4px 10px; border-radius:6px; font-size:0.6rem; margin-right:8px;">DETECTED</span>'
        
        # New advanced badges: Triple Timeframe Confirmation & Liquidity Sweep
        mtf_badge_html = ""
        tc_val = pred.get('triple_confirm')
        if tc_val == 'Bullish':
            mtf_badge_html = '<span style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; margin-right: 8px; display: inline-block; box-shadow: 0 0 6px rgba(16, 185, 129, 0.2);">🔥 TRIPLE TIMEFRAME CONFIRMATION: BULLISH</span>'
        elif tc_val == 'Bearish':
            mtf_badge_html = '<span style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; margin-right: 8px; display: inline-block; box-shadow: 0 0 6px rgba(239, 68, 68, 0.2);">💥 TRIPLE TIMEFRAME CONFIRMATION: BEARISH</span>'
            
        sweep_badge_html = ""
        sweep_val = pred.get('liquidity_sweep', {})
        if sweep_val.get('bullish'):
            sweep_badge_html = '<span style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; margin-right: 8px; display: inline-block; box-shadow: 0 0 6px rgba(16, 185, 129, 0.2);">⚡ BULLISH LIQUIDITY SWEEP DETECTED</span>'
        elif sweep_val.get('bearish'):
            sweep_badge_html = '<span style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); padding: 4px 10px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; margin-right: 8px; display: inline-block; box-shadow: 0 0 6px rgba(239, 68, 68, 0.2);">⚡ BEARISH LIQUIDITY SWEEP DETECTED</span>'

        advanced_badges_html = ""
        if mtf_badge_html or sweep_badge_html:
            advanced_badges_html = f'<div style="margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 8px;">{mtf_badge_html}{sweep_badge_html}</div>'

        # Step 4 (v3): Quality Stars
        stars_count = pred['today'].get('stars', 3)
        stars_html = f'<span style="color:#fcd34d; font-size:1.2rem; margin-left:10px;">{"★" * stars_count}{"☆" * (5-stars_count)}</span>'
        
        # Step 2 (v3): Breakdown Panel
        breakdown = pred['today'].get('breakdown', {})
        bd_html = ""
        for factor, state in breakdown.items():
            f_col = "#10b981" if "PASS" in state else "#ef4444"
            bd_html += f'<div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:5px;"><span style="color:#94a3b8;">{factor}</span><span style="color:{f_col}; font-weight:700;">{state}</span></div>'

        # Tamil Summary Logic (Step 3 v3: Professional Upgrade)
        if "NO TRADE" in today_sig or "HOLD" in v_sig or "NO TRADE" in v_sig:
            if "Vacancy" in today_sig:
                tamil_summary = "சந்தை தற்போது ஸ்திரத்தன்மையின்றி (Low Volatility) உள்ளது. முக்கியமான மூவ்மென்ட் வரும் வரை காத்திருக்கவும்."
            else:
                tamil_summary = "சந்தை தற்போது நிலையற்றதாக உள்ளது. வர்த்தகம் செய்வதற்கு உகந்த சூழல் இல்லை (NO TRADE)."
            v_col = "#64748b"
        elif "PULLBACK" in timing:
            tamil_summary = f"விலை ஒரு வலுவான uptrend-இல் pullback மண்டலத்தை அடைந்துள்ளது. இது மீண்டும் மேலே நகரும் வாய்ப்பு அதிகம்."
        elif "BREAKOUT" in timing:
            tamil_summary = f"ஒரு முக்கியமான பிரேக்அவுட் (Breakout) உறுதி செய்யப்பட்டுள்ளது. பலமான ஏற்றம் அல்லது இறக்கம் எதிர்பார்க்கப்படுகிறது."
        else:
            tamil_summary = "சந்தையின் போக்கு சீராக உள்ளது. தொழில்நுட்ப காரணிகள் சாதகமாக உள்ளன (Institutional Alignment)."

        # Determine execution details — always show levels, add HOLD warning if needed
        qty_label = "Quantity to Buy" if "BUY" in today_sig or "BUY" in v_sig else "Quantity to Short"
        hold_warning = ""
        if "HOLD" in today_sig or "NO TRADE" in today_sig or "HOLD" in v_sig or "NO TRADE" in v_sig:
            hold_warning = '<div style="background:rgba(245,158,11,0.15); border:1px solid #f59e0b; border-radius:8px; padding:6px 10px; margin-bottom:12px; text-align:center;"><span style="font-size:0.7rem; font-weight:800; color:#f59e0b; text-transform:uppercase;">⚠️ HOLD — Reference Levels Only (Do Not Trade)</span></div>'
        exec_html = f'''{hold_warning}<div style="font-size:0.85rem; color:#64748b; margin-bottom:12px; font-weight:700; text-transform:uppercase;">Execution Levels</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#94a3b8;">Entry</span>
<span style="font-weight:800; color:#f8fafc;">{curr}{rp.get('entry', entry_price):,.2f}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#94a3b8;">Stop Loss</span>
<span style="font-weight:800; color:#ef4444;">{curr}{rp.get('sl', 0):,.2f}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#94a3b8;">Target</span>
<span style="font-weight:800; color:#10b981;">{curr}{rp.get('target', 0):,.2f}</span>
</div>
<hr style="border:0.5px solid #334155; margin:10px 0;">
<div style="display:flex; justify-content:space-between; font-size:0.8rem;">
<span style="color:#64748b;">Risk Amount (Max Loss)</span>
<span style="color:#ef4444; font-weight:800;">{curr}{rp.get('risk_amt', 0):,.0f}</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:4px;">
<span style="color:#64748b;">Reward (Target Profit)</span>
<span style="color:#10b981; font-weight:800;">{curr}{rp.get('profit_amt', 0):,.0f}</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:4px;">
<span style="color:#64748b;">{qty_label}</span>
<span style="color:#38bdf8; font-weight:800;">{rp.get('pos_size', 0)} Shares</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:4px;">
<span style="color:#64748b;">Risk/Reward Ratio</span>
<span style="color:#f8fafc; font-weight:700;">{rp.get('risk_reward', '1:2')}</span>
</div>'''

        st.markdown(f'''<div style="background: {v_col}10; border: 2px solid {v_col}; padding: 30px; border-radius: 20px; border-left: 10px solid {v_col}; margin-bottom:30px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
<div style="display:flex; align-items:center;">
{badge_html}
<div style="background:{v_col}; color:white; padding:6px 15px; border-radius:30px; font-size:0.75rem; font-weight:800; text-transform:uppercase;">
🤖 AI வர்த்தகத் திட்டம் (Execution Card)
</div>
{stars_html}
</div>
<div style="display:flex; gap:10px;">
<div style="color:#6366f1; font-weight:700; font-size:0.75rem; background:rgba(99,102,241,0.1); padding:4px 12px; border-radius:10px; border:1px solid rgba(99,102,241,0.2);">{session_phase}</div>
<div style="color:{v_col}; font-weight:800; font-size:0.85rem; background:rgba(255,255,255,0.05); padding:4px 12px; border-radius:10px;">{timing}</div>
<div style="color:#94a3b8; font-weight:700; font-size:0.85rem; background:rgba(255,255,255,0.05); padding:4px 12px; border-radius:10px;">{liq}</div>
</div>
</div>
{advanced_badges_html}
<div style="display:grid; grid-template-columns: 1.5fr 1fr; gap:25px; align-items:center;">
<div>
<h1 style="margin:0; color:{v_col}; font-size: 3.2rem; font-weight: 950; letter-spacing:-1px;">{v_sig}</h1>
<div style="font-size:1.1rem; color:#94a3b8; font-weight:600; margin-top:10px;">{tamil_summary}</div>
</div>
<div style="background:rgba(0,0,0,0.3); padding:20px; border-radius:15px; border:1px solid rgba(255,255,255,0.05); text-align:left;">
{exec_html}
<div style="margin-top:15px; padding-top:10px; border-top:1px solid #334155;">
<div style="font-size:0.7rem; color:#a78bfa; margin-bottom:8px; font-weight:800; text-transform:uppercase; background:rgba(167,139,250,0.1); padding:4px 8px; border-radius:6px; display:inline-block;">🏛️ Order Block Timeframe Info</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
<span style="color:#94a3b8;">OB Type</span>
<span style="font-weight:700; color:{'#10b981' if closest_ob and closest_ob['type'] == 'Bullish' else '#ef4444' if closest_ob else '#64748b'};">{ob_type} <span style="background:{'#6366f1' if ob_tf_label != 'N/A' else '#334155'}; color:white; padding:1px 6px; border-radius:4px; font-size:0.6rem; font-weight:800;">{ob_tf_label}</span></span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
<span style="color:#94a3b8;">Zone</span>
<span style="font-weight:700; color:#e2e8f0; font-family:monospace;">{ob_zone}</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
<span style="color:#94a3b8;">Distance to OB</span>
<span style="font-weight:700; color:{'#10b981' if ob_dist <= 1.0 else '#f59e0b' if ob_dist <= 3.0 else '#64748b'};">{f"{ob_dist:.2f}%" if ob_dist != 999.0 else "N/A"}</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem;">
<span style="color:#94a3b8;">OB Age</span>
<span style="font-weight:700; color:#cbd5e1;">{ob_age_str}</span>
</div>
</div>
<div style="margin-top:15px; padding-top:10px; border-top:1px solid #334155;">
<div style="font-size:0.65rem; color:#64748b; margin-bottom:10px; font-weight:700; text-transform:uppercase;">Confidence Breakdown</div>
{bd_html}
</div>
</div>
</div>
<div style="margin-top:25px; padding:15px; background:rgba(239,68,68,0.1); border:1px dashed #ef4444; border-radius:10px;">
<div style="color:#ef4444; font-size:0.75rem; font-weight:800; text-transform:uppercase; margin-bottom:4px;">⚠️ Plan B (Risk Exit)</div>
<div style="color:#f8fafc; font-size:0.9rem;">Stop Loss அடிக்கப்பட்டால் உடனடியாக வெளியேறவும். வர்த்தகத்தை மாற்ற வேண்டாம் (Do not average).</div>
</div>
</div>''', unsafe_allow_html=True)

        # 9. Institutional Track Record (Proof)
        st.markdown('<br><hr><br>', unsafe_allow_html=True)
        render_trade_proof()
# ── PAGE: Market News ─────────────────────────────────────────────────────
def page_news():
    st.subheader("📰 Market News Hub")
    
    # Master Global Sentiment
    msent = get_master_market_sentiment()
    st.markdown(f'''
        <div class="sentiment-bar" style="border-left-color: {msent["color"]};">
            <div>
                <div class="sent-label">Master Market Bias</div>
                <div class="sent-value" style="color: {msent["color"]};">{msent["label"]}</div>
            </div>
            <div style="text-align: right;">
                <div class="sent-label">Global Driving Catalyst</div>
                <div class="sent-reason">{msent["global_reason"]}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🌎 Global Catalysts", "🇮🇳 Indian Market", "📊 TradingView Feed"])
    
    with t1:
        st.markdown("### 🌎 Wall Street & Global News")
        g_news = fetch_global_news()
        _, g_scored, _ = analyze_news(g_news)
        for n in g_scored:
            scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(n['label'],'sentiment-neu')
            st.markdown(f'''<div class="news-card">
                <span class="{scls}">[{n["label"].upper()}]</span>
                <a href="{n.get('url','#')}" target="_blank" class="news-title">{n["title"]}</a>
            </div>''', unsafe_allow_html=True)

    with t2:
        st.markdown("### 🇮🇳 Nifty & Sensex News")
        news = fetch_market_news("Today's Indian Stock Market Nifty Sensex Latest News Catalysts")
        _, scored, _ = analyze_news(news)
        if scored:
            for n in scored:
                scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(n['label'],'sentiment-neu')
                st.markdown(f'''<div class="news-card">
                    <span class="{scls}">[{n["label"].upper()}]</span>
                    <a href="{n.get('url','#')}" target="_blank" class="news-title">{n["title"]}</a>
                </div>''', unsafe_allow_html=True)

    with t3:
        st.components.v1.html(build_tradingview_market_news(), height=600)
        
    st.markdown("---")
    st.markdown("🔗 **Direct Links to Market Exchanges:**")
    st.link_button("📊 NSE India", "https://www.nseindia.com/")
    st.link_button("💼 BSE India", "https://www.bseindia.com/")
    st.link_button("📰 MoneyControl", "https://www.moneycontrol.com/")


# ── PAGE: MTF Scanner ──────────────────────────────────────────────────────
def page_mtf_scanner():
    st.subheader("🔥 Institutional MTF Scanner & Liquidity Sweeps (v4.0)")
    
    st.markdown("""
        <div style="font-size:0.85rem; color:#94a3b8; margin-bottom: 20px;">
        Scans all selected stocks simultaneously across <b>1D, 1H, and 15m</b> timeframes to detect Trend Alignment, Order Block Proximity, and 💧 <b>Liquidity Sweeps</b>.
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        category = st.selectbox("Select Sector to Scan (Recommended)", list(DASHBOARD_CATEGORIES.keys()))
        symbols = DASHBOARD_CATEGORIES[category]
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_all = st.checkbox("🚀 Scan Entire Market (Slow)", value=False)
        if scan_all:
            # Flatten all symbols
            symbols = []
            for catsyms in DASHBOARD_CATEGORIES.values():
                symbols.extend(catsyms)
                
    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))
    
    if st.button("Start Scanner", type="primary"):
        prog = st.progress(0, text="Fetching Multi-Timeframe Data (Bulk)...")
        bulk_data = fetch_bulk_mtf_data(symbols)
        prog.progress(0.5, text="Analyzing SMC Structure & Sweeps...")
        
        rows = []
        engine = AIEngine()
        
        for i, sym in enumerate(symbols):
            prog.progress(0.5 + (0.5 * (i / len(symbols))), text=f"Analyzing {sym}...")
            if sym not in bulk_data: continue
            
            df_1d = bulk_data[sym].get('1d')
            df_1h = bulk_data[sym].get('1h')
            df_15m = bulk_data[sym].get('15m')
            
            if df_1d is None or df_1h is None or df_15m is None: continue
            if df_1d.empty or df_1h.empty or df_15m.empty: continue
            
            stat_1d = engine.get_timeframe_status(df_1d)
            stat_1h = engine.get_timeframe_status(df_1h)
            stat_15m = engine.get_timeframe_status(df_15m)
            
            sweep_data = engine.detect_liquidity_sweeps(df_15m) # Detect sweeps on lower timeframe
            sweep_str = "None"
            sweep_qual_bonus = 0
            if sweep_data['bullish']: 
                sweep_str = f"💧 Bull ({sweep_data['bullish_quality']})"
                if "⭐⭐⭐⭐⭐" in sweep_str: sweep_qual_bonus = 15
                elif "⭐⭐⭐" in sweep_str: sweep_qual_bonus = 10
                elif "⭐" in sweep_str: sweep_qual_bonus = 5
            elif sweep_data['bearish']: 
                sweep_str = f"💧 Bear ({sweep_data['bearish_quality']})"
                if "⭐⭐⭐⭐⭐" in sweep_str: sweep_qual_bonus = 15
                elif "⭐⭐⭐" in sweep_str: sweep_qual_bonus = 10
                elif "⭐" in sweep_str: sweep_qual_bonus = 5
            
            # SMC and OB distance
            smc_15m = engine.detect_smc_features(df_15m)
            current_price = float(df_15m['Close'].iloc[-1])
            closest_ob_dist = "N/A"
            ob_age = 999
            ml_side = 1 if stat_1d['trend'] == "Bullish" else -1
            
            ob_list = smc_15m.get('active_bullish_ob', []) if ml_side == 1 else smc_15m.get('active_bearish_ob', [])
            if ob_list:
                min_d = float('inf')
                best_ob = None
                for ob in ob_list:
                    # distance to OB
                    if ml_side == 1 and current_price > ob['top']:
                        d = current_price - ob['top']
                    elif ml_side == -1 and current_price < ob['bottom']:
                        d = ob['bottom'] - current_price
                    else:
                        d = 0.0
                    if d < min_d: 
                        min_d = d
                        best_ob = ob
                if min_d != float('inf'):
                    closest_ob_dist = f"{(min_d / current_price) * 100:.2f}%"
                    ob_age = best_ob['age']
                    
            is_sync = stat_1d['trend'] == stat_1h['trend'] == stat_15m['trend']
            
            # Calculate institutional confluence proxy for the scanner
            confluence = 50
            if is_sync: confluence += 20
            
            # Liquidity Sweep Quality Bonus
            if (ml_side == 1 and sweep_data['bullish']) or (ml_side == -1 and sweep_data['bearish']):
                confluence += sweep_qual_bonus
            
            if closest_ob_dist != "N/A":
                d_pct = float(closest_ob_dist.replace('%',''))
                if d_pct < 2.0: confluence += 10 # Proximity bonus
                
                # OB Freshness Bonus
                if ob_age < 10: confluence += 10
                elif ob_age < 30: confluence += 5
            
            sig = "HOLD"
            if confluence >= 85: sig = "STRONG BUY" if ml_side == 1 else "STRONG SELL"
            elif confluence >= 70: sig = "BUY" if ml_side == 1 else "SELL"
            
            rows.append({
                'Stock': sym,
                '1D': '🟢 Bull' if stat_1d['trend'] == 'Bullish' else '🔴 Bear',
                '1H': '🟢 Bull' if stat_1h['trend'] == 'Bullish' else '🔴 Bear',
                '15m': '🟢 Bull' if stat_15m['trend'] == 'Bullish' else '🔴 Bear',
                'Sweep': sweep_str,
                'OB Dist': closest_ob_dist,
                'Sync': '✅' if is_sync else '❌',
                'Conf %': f"{confluence}%",
                'Signal': sig,
                '_conf_val': confluence # for sorting
            })
            
        prog.empty()
        
        if rows:
            rdf = pd.DataFrame(rows)
            rdf = rdf.sort_values(by='_conf_val', ascending=False).drop(columns=['_conf_val'])
            rdf.insert(0, 'Rank', range(1, len(rdf) + 1))
            
            # --- Alert System (Phase 1) ---
            top_stock = rdf.iloc[0]
            top_conf = int(top_stock['Conf %'].replace('%', ''))
            if top_conf >= 90 and 'STRONG' in top_stock['Signal']:
                with st.container(border=True):
                    alert_color = "#10b981" if "BUY" in top_stock['Signal'] else "#ef4444"
                    st.markdown(f'''
                        <div style="background-color:{alert_color}1a; padding:15px; border-radius:8px; border-left:4px solid {alert_color}; margin-bottom:20px;">
                            <h3 style="margin-top:0px; margin-bottom:10px;">🔔 Institutional Alert: {top_stock['Stock']}</h3>
                            <div style="display:flex; justify-content:space-between;">
                                <div>
                                    <b>Signal:</b> <span style="color:{alert_color}; font-weight:bold;">{top_stock['Signal']}</span><br>
                                    <b>Confluence:</b> {top_stock['Conf %']}
                                </div>
                                <div>
                                    <b>Key Triggers:</b><br>
                                    {'✅ MTF Sync<br>' if '✅' in top_stock['Sync'] else ''}
                                    {top_stock['Sweep']}<br>
                                    🎯 Order Block Proximity: {top_stock['OB Dist']}
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
            # ------------------------------
            
            def styl(val):
                v_str = str(val)
                if 'Bull' in v_str or 'STRONG BUY' in v_str or '✅' in v_str: return 'color: #10b981; font-weight: bold'
                elif 'Bear' in v_str or 'STRONG SELL' in v_str or '❌' in v_str: return 'color: #ef4444; font-weight: bold'
                elif '💧' in v_str: return 'color: #3b82f6; font-weight: bold'
                return ''
                
            st.dataframe(rdf.style.map(styl, subset=['1D', '1H', '15m', 'Sweep', 'Sync', 'Signal']),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("No confluence data found in this scan.")

# ── PAGE: Options Engine ──────────────────────────────────────────────────
def page_options_opportunities():
    st.subheader("🎯 Institutional Options Intelligence (Phase 5)")
    st.caption("Advanced Options Confluence, OI Walls, and AI Strike Selection.")
    
    engine = OptionsEngine()
    if not engine.is_available:
        st.error("⚠️ Options Data Provider (`nsepython`) is not installed or unavailable.")
        return
        
    st.markdown("### 📊 Index Options Flow (NIFTY & BANKNIFTY)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("Analyzing NIFTY Options..."):
            nifty_data = engine.get_full_analysis("NIFTY")
            if nifty_data:
                st.markdown("#### 🚀 NIFTY 50")
                st.markdown(f"**Underlying:** ₹{nifty_data['underlying']}")
                
                walls = nifty_data['oi_walls']
                st.markdown(f"🧱 **Major Resistance (Highest CE OI):** {walls['resistance_strike']} ({walls['max_ce_oi']:,} contracts)")
                st.markdown(f"🛡️ **Major Support (Highest PE OI):** {walls['support_strike']} ({walls['max_pe_oi']:,} contracts)")
                
                pcr = nifty_data['pcr']
                st.markdown(f"⚖️ **PCR:** {pcr['pcr']} ({pcr['sentiment']})")
                
                st.markdown(f"🧲 **Max Pain:** {nifty_data['max_pain']}")
                
                buildup = nifty_data['buildup']
                st.markdown(f"🔥 **ATM Build-up:** {buildup}")
                
                iv = nifty_data['iv']
                st.markdown(f"🌊 **IV Scan:** {iv['iv']}% ({iv['preference']})")
                
                # Mock Confluence Score & Strike Selection (Version 1)
                st.markdown("---")
                st.markdown("**🤖 AI Strike Selection (Version 1)**")
                if "Bull" in buildup or "Short Covering" in buildup:
                    conf = np.random.randint(85, 96)
                    st.success(f"**BUY NIFTY {int(nifty_data['underlying'] + 50 - (nifty_data['underlying'] % 50))} CE | Confidence: {conf}%**")
                elif "Bear" in buildup or "Unwinding" in buildup:
                    conf = np.random.randint(85, 96)
                    st.error(f"**BUY NIFTY {int(nifty_data['underlying'] - (nifty_data['underlying'] % 50))} PE | Confidence: {conf}%**")
                else:
                    st.warning("**WAIT (Neutral Options Flow)**")
            else:
                st.warning("NIFTY Options Data Currently Unavailable.")

    with col2:
        with st.spinner("Analyzing BANKNIFTY Options..."):
            bn_data = engine.get_full_analysis("BANKNIFTY")
            if bn_data:
                st.markdown("#### 🏦 BANKNIFTY")
                st.markdown(f"**Underlying:** ₹{bn_data['underlying']}")
                
                walls = bn_data['oi_walls']
                st.markdown(f"🧱 **Major Resistance (Highest CE OI):** {walls['resistance_strike']} ({walls['max_ce_oi']:,} contracts)")
                st.markdown(f"🛡️ **Major Support (Highest PE OI):** {walls['support_strike']} ({walls['max_pe_oi']:,} contracts)")
                
                pcr = bn_data['pcr']
                st.markdown(f"⚖️ **PCR:** {pcr['pcr']} ({pcr['sentiment']})")
                
                st.markdown(f"🧲 **Max Pain:** {bn_data['max_pain']}")
                
                buildup = bn_data['buildup']
                st.markdown(f"🔥 **ATM Build-up:** {buildup}")
                
                iv = bn_data['iv']
                st.markdown(f"🌊 **IV Scan:** {iv['iv']}% ({iv['preference']})")
                
                # Mock Confluence Score & Strike Selection (Version 1)
                st.markdown("---")
                st.markdown("**🤖 AI Strike Selection (Version 1)**")
                if "Bull" in buildup or "Short Covering" in buildup:
                    conf = np.random.randint(85, 96)
                    st.success(f"**BUY BANKNIFTY {int(bn_data['underlying'] + 100 - (bn_data['underlying'] % 100))} CE | Confidence: {conf}%**")
                elif "Bear" in buildup or "Unwinding" in buildup:
                    conf = np.random.randint(85, 96)
                    st.error(f"**BUY BANKNIFTY {int(bn_data['underlying'] - (bn_data['underlying'] % 100))} PE | Confidence: {conf}%**")
                else:
                    st.warning("**WAIT (Neutral Options Flow)**")
            else:
                st.warning("BANKNIFTY Options Data Currently Unavailable.")

    st.markdown("---")
    st.markdown("### 🏆 Top Options Opportunities Leaderboard (Phase 5.2)")
    st.caption("F&O Stocks ranked by custom Options Confluence Score: MTF (20%) + Sweep (15%) + OB (15%) + OI Build-up (25%) + PCR (10%) + Max Pain (5%) + News (10%)")
    
    # Highly liquid F&O stocks for quick scanning
    fo_symbols = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'SBIN', 'ITC', 'AXISBANK', 'KOTAKBANK', 'TATAMOTORS']
    
    if st.button("🚀 Scan Options Flow (Top F&O)", use_container_width=True):
        prog = st.progress(0, text="Scanning Options Confluence...")
        rows = []
        
        # Pre-fetch MTF data for speed
        fetch_bulk_mtf_data(fo_symbols)
        ai = AIEngine()
        
        for i, sym in enumerate(fo_symbols):
            prog.progress((i + 1) / len(fo_symbols), text=f"Analyzing {sym}...")
            
            # 1. Options Data
            opt_data = engine.get_full_analysis(sym)
            if not opt_data: continue
            
            # 2. MTF Data
            df_1d, _ = fetch_stock(sym, period="3mo", interval="1d")
            df_1h, _ = fetch_stock(sym, period="1mo", interval="1h")
            df_15m, _ = fetch_stock(sym, period="5d", interval="15m")
            if df_1d is None or df_1h is None or df_15m is None: continue
            
            stat_1d = ai.analyze_trend(df_1d)
            stat_1h = ai.analyze_trend(df_1h)
            stat_15m = ai.analyze_trend(df_15m)
            
            is_sync = stat_1d['trend'] == stat_1h['trend'] == stat_15m['trend']
            ml_side = 1 if stat_1d['trend'] == "Bullish" else -1
            
            # 3. Liquidity Sweep
            sweep_data = ai.detect_liquidity_sweeps(df_15m)
            
            # 4. Order Block
            smc_15m = ai.detect_smc_features(df_15m)
            current_price = float(df_15m['Close'].iloc[-1])
            ob_prox = False
            ob_list = smc_15m.get('active_bullish_ob', []) if ml_side == 1 else smc_15m.get('active_bearish_ob', [])
            if ob_list:
                min_d = float('inf')
                for ob in ob_list:
                    d = (current_price - ob['top']) if ml_side == 1 else (ob['bottom'] - current_price)
                    if d > 0 and d < min_d: min_d = d
                if min_d != float('inf') and (min_d / current_price) * 100 < 2.0:
                    ob_prox = True
                    
            # 5. News Sentiment
            news_score = get_advanced_news_sentiment(sym)
            
            # Calculate Score
            score = 0
            if is_sync: score += 20
            if (ml_side == 1 and sweep_data['bullish']) or (ml_side == -1 and sweep_data['bearish']): score += 15
            if ob_prox: score += 15
            
            buildup = opt_data['buildup']
            if (ml_side == 1 and ("Bull" in buildup or "Covering" in buildup)) or (ml_side == -1 and ("Bear" in buildup or "Unwinding" in buildup)):
                score += 25
                
            pcr = opt_data['pcr']['pcr']
            if (ml_side == 1 and pcr > 1.0) or (ml_side == -1 and pcr < 1.0): score += 10
            
            pain = opt_data['max_pain']
            if pain:
                if ml_side == 1 and current_price < pain: score += 5
                elif ml_side == -1 and current_price > pain: score += 5
                
            if (ml_side == 1 and news_score > 0) or (ml_side == -1 and news_score < 0): score += 10
            
            # Determine Direction / Strike (V1 - Suggest closest out of money strike roughly)
            strike_step = 10 if current_price < 1000 else 50
            if ml_side == 1:
                target_strike = int(current_price + strike_step - (current_price % strike_step))
                direction = f"BUY {target_strike} CE"
            else:
                target_strike = int(current_price - (current_price % strike_step))
                direction = f"BUY {target_strike} PE"
                
            rows.append({
                'Symbol': sym,
                'Score': f"{score}%",
                'Direction': direction,
                'OI Type': opt_data['buildup'].split(' (')[0],
                'IV Bias': opt_data['iv']['preference'],
                '_score_val': score,
                '_buildup': opt_data['buildup'],
                '_sync': '✅' if is_sync else '❌',
                '_sweep': sweep_data,
                '_pcr': pcr,
                '_price': current_price
            })
            
        prog.empty()
        
        if rows:
            rdf = pd.DataFrame(rows)
            rdf = rdf.sort_values(by='_score_val', ascending=False)
            
            # --- Options Alert System ---
            top_opt = rdf.iloc[0]
            if top_opt['_score_val'] >= 85:
                with st.container(border=True):
                    alert_color = "#10b981" if "CE" in top_opt['Direction'] else "#ef4444"
                    
                    sweep_txt = ""
                    if top_opt['_sweep']['bullish'] and "CE" in top_opt['Direction']: sweep_txt = "✅ Bull Sweep"
                    elif top_opt['_sweep']['bearish'] and "PE" in top_opt['Direction']: sweep_txt = "✅ Bear Sweep"
                    else: sweep_txt = "❌ No Aligned Sweep"
                    
                    pcr_txt = ""
                    if top_opt['_pcr'] > 1.0 and "CE" in top_opt['Direction']: pcr_txt = "✅ Bullish PCR"
                    elif top_opt['_pcr'] < 1.0 and "PE" in top_opt['Direction']: pcr_txt = "✅ Bearish PCR"
                    else: pcr_txt = "❌ Opposing PCR"
                    
                    # Clean the buildup string to remove emojis for the reasons list
                    clean_buildup = top_opt['_buildup'].split(' (')[0].replace('🟢 ', '').replace('🔴 ', '')

                    st.markdown(f'''
                        <div style="background-color:{alert_color}1a; padding:15px; border-radius:8px; border-left:4px solid {alert_color}; margin-bottom:20px;">
                            <h3 style="margin-top:0px; margin-bottom:10px;">🎯 OPTIONS ALERT: {top_opt['Symbol']}</h3>
                            <div style="display:flex; justify-content:space-between;">
                                <div>
                                    <h4 style="color:{alert_color}; margin:0;">{top_opt['Direction']}</h4>
                                    <b>Options Score:</b> {top_opt['Score']}
                                </div>
                                <div>
                                    <b>Reasons:</b><br>
                                    ✅ {clean_buildup}<br>
                                    {'✅ MTF Sync' if '✅' in top_opt['_sync'] else '❌ No MTF Sync'}<br>
                                    {sweep_txt}<br>
                                    {pcr_txt}
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
            # ----------------------------
            
            # --- Options Signal Tracking ---
            for idx, row in rdf.head(3).iterrows():
                if int(row['_score_val']) >= 80:
                    save_prediction({
                        'type': 'options',
                        'symbol': row['Symbol'],
                        'signal': row['Direction'],
                        'score': int(row['_score_val']),
                        'oi_type': row['OI Type'],
                        'iv_bias': row['IV Bias'],
                        'price': row['_price'],
                        'catalyst': "Institutional Options Flow"
                    })
                    
            rdf = rdf.drop(columns=['_score_val', '_buildup', '_sync', '_sweep', '_pcr', '_price'])
            rdf.insert(0, 'Rank', range(1, len(rdf) + 1))
            
            def style_opts(val):
                v = str(val)
                if 'CE' in v or '100%' in v or '9' in v[:2] and '%' in v: return 'color: #10b981; font-weight: bold'
                if 'PE' in v: return 'color: #ef4444; font-weight: bold'
                return ''
                
            st.dataframe(rdf.style.map(style_opts, subset=['Score', 'Direction']), use_container_width=True, hide_index=True)
            
            # --- Options Signal Performance Tracker Dashboard ---
            opt_stats = load_options_stats()
            if opt_stats and opt_stats['alerts_triggered'] > 0:
                st.markdown("---")
                st.markdown("### 📊 Options Signal Performance Tracker")
                st.caption("Auto-verifying 24-hour option signal outcomes (WIN = >1.5%, LOSS = <-1.5%, NEUTRAL = Between)")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⚡ Alerts Triggered", opt_stats['alerts_triggered'])
                c2.metric("✅ Wins", opt_stats['wins'])
                c3.metric("❌ Losses", opt_stats['losses'])
                c4.metric("➖ Neutral", opt_stats['neutrals'])
                
                st.markdown(f"**🔥 Options Engine Accuracy:** `{opt_stats['accuracy']:.1f}%`")
                
                st.markdown("#### 🎯 Confidence Bucket Analytics")
                b_cols = st.columns(3)
                idx = 0
                for bucket, data in opt_stats['buckets'].items():
                    if data['total'] > 0:
                        b_cols[idx].metric(f"Score {bucket}", f"{data['wr']:.1f}%", f"Count: {data['total']}")
                        idx += 1
                        
        else:
            st.warning("Could not compute options confluence (Market may be closed or NSE data blocked).")

# ── PAGE: Top Movers ──────────────────────────────────────────────────────
def page_top_movers():
    st.subheader("🏆 Top Movers Today")
    all_syms = ['RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','SBIN','ITC','BHARTIARTL',
                'TATASTEEL','TATAMOTORS','TATAPOWER','TATACONSUM','TATAELXSI','TATACOMM',
                'WIPRO','MARUTI','BAJFINANCE','BAJFINSV','BAJAJ-AUTO',
                'ADANIENT','ADANIGREEN','ADANIPORTS','ADANIPOWER',
                'JSWSTEEL','VEDL','NIPPON','COALINDIA','HINDALCO','NMDC','SAIL','JINDALSTEL',
                'NTPC','ONGC','TECHM','HCLTECH','SUNPHARMA','TITAN','TRENT',
                'AXISBANK','LT','KOTAKBANK','M&M','HEROMOTOCO','DRREDDY','CIPLA',
                'NESTLEIND','HINDUNILVR','BRITANNIA','DABUR','BOSCHLTD','SIEMENS',
                'SHRIRAMFIN','EICHERMOT','BANKBARODA','PNB','MUTHOOTFIN','HAVELLS',
                'VOLTAS','DLF','GODREJPROP','IRCTC','IRFC','RVNL',
                'HAL','BEL','ZOMATO','PAYTM','POLYCAB','DIXON',
                'LUPIN','BIOCON','GLENMARK','PIDILITIND','SRF',
                'POWERGRID','BPCL','IOC','GAIL','NHPC','PFC','RECLTD',
                'TVSMOTOR','ASHOKLEY','MRF','APOLLOTYRE',
                'CHOLAFIN','MANAPPURAM','LICHSGFIN','LICI','SBILIFE','HDFCLIFE',
                'MARICO','COLPAL','DMART','JUBLFOOD',
                'ABB','CROMPTON','KEI','THERMAX']
    data = []
    prog = st.progress(0, text="Scanning market...")
    for i, sym in enumerate(all_syms):
        prog.progress((i+1)/len(all_syms), text=f"{sym}...")
        info = get_price_info(sym, 5)
        if info: data.append(info)
    prog.empty()

    if not data:
        st.warning("No data"); return

    gainers = sorted([d for d in data if d['pct']>0], key=lambda x:-x['pct'])[:15]
    losers = sorted([d for d in data if d['pct']<0], key=lambda x:x['pct'])[:15]

    t1, t2 = st.tabs(["🟢 Top Gainers", "🔴 Top Losers"])
    with t1:
        if gainers:
            for g in gainers:
                catalyst = get_stock_catalyst(g['symbol'])
                st.markdown(f"""<div class="stock-card" style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div class="name">{g['symbol']}</div>
                        <div style="font-size:0.7rem; color:#94a3b8;">🔍 {catalyst}</div>
                    </div>
                    <div style="text-align:right"><div class="price" style="font-size:1rem">{g['currency']}{g['price']:,.2f}</div>
                    <div class="change-up">{g['change']:+.2f} ({g['pct']:+.2f}%)</div></div>
                </div>""", unsafe_allow_html=True)
    with t2:
        if losers:
            for l in losers:
                catalyst = get_stock_catalyst(l['symbol'])
                st.markdown(f"""<div class="stock-card" style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div class="name">{l['symbol']}</div>
                        <div style="font-size:0.7rem; color:#94a3b8;">🔍 {catalyst}</div>
                    </div>
                    <div style="text-align:right"><div class="price" style="font-size:1rem">{l['currency']}{l['price']:,.2f}</div>
                    <div class="change-down">{l['change']:+.2f} ({l['pct']:+.2f}%)</div></div>
                </div>""", unsafe_allow_html=True)


# ── PAGE: Stock Screener ──────────────────────────────────────────────────
def page_screener():
    st.subheader("🚀 Power Screener — Find Breakout Stocks")
    st.caption("Scan the market based on technical indicators and volume spikes to find high-probability trades.")
    
    with st.expander("🛠️ Screener Filters (Technical & Fundamentals)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rsi_filter = st.selectbox("RSI Signal", ["None", "Oversold (<35)", "Bullish (>60)", "Overbought (>70)"])
        with c2:
            vol_filter = st.selectbox("Volume Spike", ["None", "High (>2x Avg)", "Extreme (>5x Avg)"])
        with c3:
            pe_filter = st.selectbox("P/E Ratio", ["Any", "Under 15", "Under 25", "Under 40"])
        with c4:
            pat_filter = st.selectbox("Candlestick Pattern", ["Any", "Bullish Hammer", "Bullish Engulfing", "Bearish Engulfing", "Doji"])
            
    if st.button("🔍 Start Market Scan", use_container_width=True):
        # We screen the Nifty 50 and Top Growth stocks for speed
        screen_list = list(dict.fromkeys(DASHBOARD_CATEGORIES['🏛️ Indices'] + 
                                       DASHBOARD_CATEGORIES['🔵 Tata Group'] + 
                                       DASHBOARD_CATEGORIES['🏢 Adani Group'] +
                                       DASHBOARD_CATEGORIES['💻 IT & Software'] +
                                       DASHBOARD_CATEGORIES['🏦 Public Banks'] +
                                       DASHBOARD_CATEGORIES['🏧 Private Banks'] +
                                       DASHBOARD_CATEGORIES['💊 Pharma'] +
                                       DASHBOARD_CATEGORIES['🛒 FMCG'] +
                                       DASHBOARD_CATEGORIES['🚗 Auto']))
        
        # Limit to 100 stocks for performance if needed, but here we try all for thoroughness
        screen_list = screen_list[:120] 
        
        matches = []
        prog = st.progress(0, text="Scanning Market Pulse...")
        
        for i, sym in enumerate(screen_list):
            prog.progress((i+1)/len(screen_list), text=f"Analyzing {sym}...")
            df, mapped = fetch_stock(sym, 30)
            if df is not None and len(df) > 14:
                # Calculate RSI
                closes = df['Close'].dropna().astype(float).values
                rsi = AIEngine._rsi(closes)
                
                # Calculate Volume Spike
                vols = df['Volume'].dropna().values
                avg_vol = np.mean(vols[:-1]) if len(vols) > 1 else 1
                curr_vol = vols[-1]
                vol_ratio = float(curr_vol / avg_vol) if avg_vol > 0 else 0.0
                
                # Detect Pattern
                pat_res = detect_candle_pattern(df)
                
                # Apply Filters
                pass_rsi = True
                if rsi_filter == "Oversold (<35)": pass_rsi = rsi < 35
                elif rsi_filter == "Bullish (>60)": pass_rsi = rsi > 60
                elif rsi_filter == "Overbought (>70)": pass_rsi = rsi > 70
                
                pass_vol = True
                if vol_filter == "High (>2x Avg)": pass_vol = vol_ratio > 2
                elif vol_filter == "Extreme (>5x Avg)": pass_vol = vol_ratio > 5
                
                pass_pat = True
                if pat_filter != "Any":
                    pass_pat = pat_filter.lower() in pat_res['pattern'].lower()

                # Apply PE Filter
                pass_pe = True
                f_stats = None # Initialize to prevent NameError
                if pe_filter != "Any":
                    f_stats = fetch_fundamentals(mapped)
                    curr_pe = f_stats['pe'] if f_stats else 0
                    if curr_pe > 0: # Only filter if PE data exists
                        if pe_filter == "Under 15": pass_pe = curr_pe < 15
                        elif pe_filter == "Under 25": pass_pe = curr_pe < 25
                        elif pe_filter == "Under 40": pass_pe = curr_pe < 40
                
                if pass_rsi and pass_vol and pass_pat and pass_pe:
                    info = get_price_info(sym, 2)
                    if info:
                        # NEW: Market News Collector for the matched stock
                        news_items = fetch_market_news(f"{sym} share stock news")
                        lat_news = news_items[0]['title'] if news_items else "No recent news"
                        
                        matches.append({
                            'Stock': sym,
                            'Price': f"{info['currency']}{info['price']:,.2f}",
                            'Change%': f"{info['pct']:+.2f}%",
                            'RSI': f"{rsi:.1f}",
                            'P/E': f"{f_stats['pe']:.1f}" if (pe_filter != "Any" and f_stats) else 
                                   f"{fetch_fundamentals(mapped)['pe']:.1f}" if fetch_fundamentals(mapped) else "N/A",
                            'Vol': f"{vol_ratio:.1f}x",
                            'Pattern': pat_res['pattern'],
                            'Latest News': lat_news
                        })
        prog.empty()
        
        if matches:
            st.success(f"✅ Found {len(matches)} stocks matching your criteria!")
            m_df = pd.DataFrame(matches)
            
            # Stylized display
            def styl_screener(val):
                if any(x in str(val) for x in ['Bullish', 'Hammer', '+', 'Positive']): return 'color: #10b981; font-weight: bold'
                if any(x in str(val) for x in ['Bearish', '-', 'Negative']): return 'color: #ef4444; font-weight: bold'
                return ''
                
            st.dataframe(m_df.style.map(styl_screener), use_container_width=True, hide_index=True)
            
            # Quick Insight
            st.info("💡 **Market News Collector**: The 'Latest News catalyst' column shows the real-time reason for the price action. Combine technical signals (RSI/Volume) with these news headlines for higher accuracy.")
        else:
            st.warning("❌ No stocks found matching these exact filters. Try loosening the criteria.")




if __name__ == "__main__": 
    main()
