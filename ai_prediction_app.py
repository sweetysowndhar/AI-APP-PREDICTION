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
from prediction_tracker import save_prediction, load_history, load_advanced_stats, auto_verify_signals

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Market Predictor Pro", page_icon="🧠", layout="wide",
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

st.markdown("""
<style>
/* ── MUST BE FIRST: Google Fonts import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Mobile-Specific Overrides ── */
@media (max-width: 768px) {
    /* Prevent iOS auto-zoom on inputs */
    input, select, textarea { font-size: 16px !important; }

    .main-title { font-size: 1.4rem !important; padding: 0 8px; }
    .sub-title { font-size: 0.8rem !important; }
    .ticker-bar { gap: 12px; font-size: 0.75rem; padding: 6px 10px; }
    .stock-card { padding: 0.75rem; margin: 0.3rem 0; }
    .stock-card .price { font-size: 1rem; }
    .signal-buy, .signal-sell, .signal-hold { padding: 0.9rem; font-size: 1rem; }

    /* Verdict Cards */
    .verdict-h1 { font-size: 2rem !important; margin: 8px 0 !important; }
    .verdict-desc { font-size: 0.9rem !important; }
    .verdict-container { padding: 16px !important; border-radius: 14px !important; }

    /* Forecast boxes stack vertically on mobile */
    .forecast-container > div { display: block !important; width: 100% !important; margin-bottom: 10px; }

    /* Pattern inset stacks on mobile */
    .pattern-inset { flex-direction: column; gap: 10px; }

    /* News card wraps properly */
    .news-card { flex-wrap: wrap; gap: 6px; padding: 0.6rem 0.9rem; }
    .news-title { font-size: 0.88rem; }

    /* Section head smaller */
    .section-head { font-size: 0.95rem; margin: 1.2rem 0 0.7rem 0; }

    /* Recent row scrollable */
    .recent-row { gap: 10px; }
    .recent-item { min-width: 70px; padding: 6px 10px; }
}

/* Ticker Bar */
.ticker-bar {
    background: #0f172a; padding: 8px 16px; border-radius: 8px; margin-bottom: 1rem;
    display: flex; gap: 24px; overflow-x: auto; white-space: nowrap; font-size: 0.85rem;
    border: 1px solid #1e293b;
}
.ticker-item { display: inline-block; }
.ticker-name { color: #94a3b8; font-weight: 600; }
.ticker-price { color: #e2e8f0; font-weight: 700; margin-left: 6px; }
.ticker-up { color: #10b981; font-weight: 600; margin-left: 4px; }
.ticker-down { color: #ef4444; font-weight: 600; margin-left: 4px; }

/* Main Title */
.main-title {
    font-size: 2rem; font-weight: 800; text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.3rem;
}
.sub-title { text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }

/* Stock Cards (Groww style) */
.stock-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 1rem; margin: 0.4rem 0; transition: all 0.2s;
    cursor: pointer;
}
.stock-card:hover { border-color: #667eea; transform: translateY(-2px); box-shadow: 0 4px 20px rgba(102,126,234,0.15); }
.stock-card .name { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.stock-card .price { font-size: 1.2rem; font-weight: 700; color: white; }
.stock-card .change-up { color: #10b981; font-size: 0.85rem; font-weight: 600; }
.stock-card .change-down { color: #ef4444; font-size: 0.85rem; font-weight: 600; }

/* Recently Viewed Row */
.recent-row { display: flex; gap: 16px; overflow-x: auto; padding: 8px 0; }
.recent-item {
    text-align: center; min-width: 80px; padding: 8px 12px;
    background: #1e293b; border-radius: 10px; border: 1px solid #334155;
}
.recent-item .sym { font-size: 0.8rem; font-weight: 700; color: #e2e8f0; }
.recent-item .chg-up { font-size: 0.75rem; color: #10b981; font-weight: 600; }
.recent-item .chg-down { font-size: 0.75rem; color: #ef4444; font-weight: 600; }

/* Section Headers */
.section-head {
    font-size: 1.1rem; font-weight: 800; color: #f8fafc;
    margin: 2rem 0 1rem 0; border-left: 5px solid #667eea;
    padding-left: 12px; letter-spacing: -0.02em;
}

/* Signal Cards */
.signal-buy { background: linear-gradient(135deg, #00b386, #10b981); color: white; padding: 1.2rem; border-radius: 14px; text-align: center; font-size: 1.2rem; font-weight: 700; box-shadow: 0 6px 24px rgba(0,179,134,0.3); }
.signal-sell { background: linear-gradient(135deg, #eb5b3c, #ef4444); color: white; padding: 1.2rem; border-radius: 14px; text-align: center; font-size: 1.2rem; font-weight: 700; box-shadow: 0 6px 24px rgba(235,91,60,0.3); }
.signal-hold { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; padding: 1.2rem; border-radius: 14px; text-align: center; font-size: 1.2rem; font-weight: 700; box-shadow: 0 6px 24px rgba(245,158,11,0.3); }

/* News Card (MoneyControl Style) */
.news-card {
    background: #0f172a; border: 1px solid #1e293b;
    padding: 0.75rem 1.25rem; border-radius: 8px; margin-bottom: 0.5rem;
    transition: all 0.2s; cursor: pointer;
    display: flex; align-items: center; gap: 10px;
}
.news-card:hover { background: #1e293b; border-color: #334155; }
.sentiment-pos { color: #2ecc71; font-weight: 800; font-family: 'monospace'; }
.sentiment-neg { color: #e74c3c; font-weight: 800; font-family: 'monospace'; }
.sentiment-neu { color: #94a3b8; font-weight: 800; font-family: 'monospace'; }
.news-title { color: #f8fafc; font-size: 0.95rem; font-weight: 500; text-decoration: none; }
.news-title:hover { color: #38bdf8; }

/* Pattern Inset Layout */
.pattern-inset {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-top: -10px;
    margin-bottom: 25px;
    display: flex;
    justify-content: space-around;
    align-items: center;
}

/* Index mini card */
.idx-card { background: #0f172a; border: 1px solid #1e293b; padding: 0.8rem; border-radius: 10px; margin: 0.3rem 0; }

/* Live Pulse Animations */
.pulse-green {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white; padding: 6px 18px; border-radius: 20px;
    font-weight: 800; font-size: 1rem; display: inline-block;
    animation: live-pulse-green 1.8s infinite;
    box-shadow: 0 0 20px rgba(16,185,129,0.4);
    border: 1px solid rgba(255,255,255,0.2);
}
.pulse-red {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white; padding: 6px 18px; border-radius: 20px;
    font-weight: 800; font-size: 1rem; display: inline-block;
    animation: live-pulse-red 1.8s infinite;
    box-shadow: 0 0 20px rgba(239,68,68,0.4);
    border: 1px solid rgba(255,255,255,0.2);
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
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 12px rgba(16,185,129, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129, 0); }
}
@keyframes live-pulse-red {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239,68,68, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 12px rgba(239,68,68, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239,68,68, 0); }
}
/* Market Sentiment Bar */
.sentiment-bar {
    background: #1e293b; border: 1px solid #334155; padding: 12px 20px;
    border-radius: 12px; margin-bottom: 1.5rem; display: flex;
    justify-content: space-between; align-items: center;
    border-left: 6px solid #667eea;
}
.sent-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; }
.sent-value { font-size: 1.1rem; font-weight: 800; margin-top: 2px; }
.sent-reason { font-size: 0.9rem; color: #e2e8f0; font-weight: 500; }
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

# ── Sentiment ─────────────────────────────────────────────────────────────
POS_WORDS = ['rally','gain','surge','bullish','record','high','jump','soar','beat',
             'outperform','buy','upgrade','profit','growth','boom','recover','strong','rise','up']
NEG_WORDS = ['fall','drop','crash','bearish','low','plunge','miss','sell','downgrade',
             'loss','decline','weak','cut','fear','risk','slump','down','tank','tumble']

# Catalyst Categories (Expanded for better 'Reasoning')
CATALYSTS = {
    'Earnings & Growth': ['dividend', 'earnings', 'profit', 'revenue', 'income', 'growth', 'beat', 'margin', 'sales', 'eps', 'guidance', 'upside'],
    'Deals & Orders': ['merger', 'acquisition', 'deal', 'partnership', 'contract', 'order', 'agreement', 'tender', 'jv', 'outperform', 'collaboration'],
    'Policy & Govt': ['policy', 'regulation', 'tax', 'hike', 'cut', 'budget', 'rbi', 'sebi', 'government', 'fed', 'interest', 'reserve', 'subsidy'],
    'Market Risk': ['loss', 'debt', 'default', 'scam', 'fraud', 'investigation', 'penalty', 'crisis', 'downgrade', 'fii-selling', 'inflation', 'war', 'geopolitical'],
    'Technical Breakout': ['breakout', 'support', 'resistance', 'moving-average', 'ema', 'rsi', 'oversold', 'overbought', 'crossover', 'momentum', 'volume-spike', 'bullish-pattern'],
    'Corporate Action': ['bonus', 'split', 'buyback', 'rights-issue', 'listing', 'ipo', 'management-change', 'ceo', 'board-approval']
}

def score_headline(text):
    t = text.lower()
    return sum(1 for w in POS_WORDS if w in t) - sum(1 for w in NEG_WORDS if w in t)

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

def analyze_news(headlines):
    if not headlines: return 0.0, [], "Technical Momentum (No News Data)"
    scored = []
    cat_counts = {k: 0 for k in CATALYSTS.keys()}
    best_h = None
    max_cat_match = -1
    
    # Filter headlines: Prioritize factual statements over speculative questions
    for h in headlines:
        title = h.get('title', '') if isinstance(h, dict) else str(h)
        # Skip purely speculative headlines if they are too short or just a single question
        if title.endswith('?') and len(title.split()) < 5: continue
        
        sc = score_headline(title)
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
        
        label = 'positive' if sc > 0 else 'negative' if sc < 0 else 'neutral'
        entry = {**(h if isinstance(h, dict) else {'title': title}), 'score': sc, 'label': label, 'catalysts': found_cats}
        scored.append(entry)
    
    if not scored: return 0.0, [], "Technical Indicator dominance"
    
    avg = sum(s['score'] for s in scored) / len(scored)
    
    # Identify Primary Catalyst & Build Descriptive Reason (Reason News)
    top_cat = max(cat_counts, key=cat_counts.get) if any(cat_counts.values()) else "Broad Market Trends"
    
    prefix = ""
    if top_cat == "Earnings & Growth": prefix = "Robust Corporate Earnings" if avg > 0 else "Weak Earnings Performance"
    elif top_cat == "Deals & Orders": prefix = "Strategic New Contracts or Acquisitions" if avg > 0 else "Cancelled Deals or Orders"
    elif top_cat == "Policy & Govt": prefix = "Positive Regulatory Support" if avg > 0 else "Regulatory Headwinds"
    elif top_cat == "Market Risk": prefix = "Macro Stability" if avg > 0 else "Heightened Market Risk"
    elif top_cat == "Technical Breakout": prefix = "Technical Breakout" if avg > 0 else "Technical Breakdown"
    elif top_cat == "Corporate Action": prefix = "Positive Corporate Action" if avg > 0 else "Internal Governance Scrutiny"
    else: prefix = "Bullish Sentiment" if avg > 0 else "Bearish Sentiment" if avg < 0 else "Market Dynamics"

    # Append direct headline reason if available
    h_snippet = f': "{best_h[:75]}..."' if best_h else ""
    primary = f"{prefix}{h_snippet}"
    
    return round(avg, 3), scored, primary

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


# ── AI Engine ─────────────────────────────────────────────────────────────
class AIEngine:
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

    def train(self, symbol, prices, volumes, news_sent=0.0):
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
        # Min 0.8% target, Max 4% target
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

    def predict(self, symbol, prices, volumes, news_sent=0.0, tv_sent=0.0, intraday=False, df=None, df_1h=None, df_1d=None):
        if symbol not in self.models:
            if not self.load_model() or symbol not in self.models: return None
        
        sc = self.scalers[symbol]
        
        # 1. MTF Status Analysis
        main_status = self.get_timeframe_status(df)
        mtf_data = {}
        if df_1h is not None: mtf_data["1h"] = self.get_timeframe_status(df_1h)
        if df_1d is not None: mtf_data["1d"] = self.get_timeframe_status(df_1d)
        
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
        
        # 4. Feature Extraction for current point
        prices_arr = np.array(prices, dtype=float)
        volumes_arr = np.array(volumes, dtype=float)
        
        # Global moms array for _features
        g_moms = [g_mean_mom] * len(prices_arr)
        f_latest = self._features(prices_arr, volumes_arr, len(prices_arr), global_moms=g_moms)
        if f_latest is None: return None
        
        feat = sc.transform([f_latest])
        
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
            is_mtf_sync = (mtf_data["1d"]["trend"] == mtf_data["1h"]["trend"] == main_status["trend"])
            if is_mtf_sync: mtf_alignment *= 1.2 # Bonus for sync

        is_trending = f_latest[-1] > 0.5 
        
        for label, step_key in zip(labels, steps):
            m_set = self.models[symbol][step_key]
            
            probs_rf = m_set['rf'].predict_proba(feat)[0]
            probs_gb = m_set['gb'].predict_proba(feat)[0]
            
            classes = list(m_set['rf'].classes_)
            p_buy = (probs_rf[classes.index(1)] + probs_gb[classes.index(1)]) / 2 if 1 in classes else 0
            p_sell = (probs_rf[classes.index(-1)] + probs_gb[classes.index(-1)]) / 2 if -1 in classes else 0
            
            raw_prob = p_buy if p_buy > p_sell else p_sell
            ml_side = 1 if p_buy > p_sell else -1
            
            # Base final score (Dynamic Weights for Backtesting vs Live)
            w_ml, w_tech, w_news, w_tv = 0.35, 0.35, 0.15, 0.15
            score_sum = (w_ml * raw_prob) + (w_tech * main_status['score'])
            denominator = w_ml + w_tech
            
            # Only add sentiment weights if data is actually present (above neutral threshold)
            if abs(news_sent) > 0.01:
                score_sum += w_news * abs(news_sent)
                denominator += w_news
            
            if abs(tv_sent) > 0.01:
                score_sum += w_tv * abs(tv_sent)
                denominator += w_tv
                
            final_score = score_sum / denominator if denominator > 0 else 0
            
            # Applying Multipliers
            final_score *= vol_mult
            if not is_trending: final_score *= 0.95 # Minimal penalty for non-trending if other scores are high
            final_score *= mtf_alignment
            final_score = min(max(final_score, 0), 1)
            
            # Signal Logic
            if vol_mult <= 0.1: sig = "NO TRADE (Low Volatility)"
            elif is_conflict: sig = "NO TRADE (Trend Conflict)"
            elif final_score >= 0.65: sig = "STRONG BUY" if ml_side == 1 else "STRONG SELL"
            elif final_score >= 0.40: sig = "BUY" if ml_side == 1 else "SELL"
            else: sig = "NO TRADE (Low Confidence)"
            
            stars = 5 if final_score >= 0.85 else 4 if final_score >= 0.75 else 3 if final_score >= 0.65 else 2 if final_score >= 0.50 else 1
            
            results[label] = {
                'signal': sig, 'confidence': round(final_score, 4), 'stars': stars,
                'ml_prob': round(raw_prob, 4), 'tech_score': main_status['score'],
                'pattern_score': 0.5,
                'is_trending': is_trending,
                'breakdown': {
                    'Trend Alignment': "PASS ✅" if not is_conflict else "FAIL ❌",
                    'Volume Confirm': "PASS ✅" if is_vol_strong else "FAIL ❌",
                    'MTF Sync': "PASS ✅" if is_mtf_sync else "FAIL ❌",
                    'Volatility OK': "PASS ✅" if vol_mult > 0 else "FAIL ❌"
                }
            }
        results['mtf_status'] = mtf_data
        results['volatility'] = vol_label
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


def build_candle_chart(df, symbol):
    """
    Build simple candlestick chart
    """
    UP_COLOR = '#00b386'
    DOWN_COLOR = '#eb5b3c'
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        name='Price'
    ), row=1, col=1)
    
    colors = [UP_COLOR if c >= o else DOWN_COLOR for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Vol', opacity=0.3), row=2, col=1)
    
    fig.update_layout(template='plotly_dark', height=450, showlegend=False, hovermode='x unified')
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
    status_text, status_color = get_market_status()
    st.markdown(f'<div class="main-title">🧠 AI Market Predictor Pro</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Real-time BSE • NSE • MoneyControl — AI-Powered Predictions &nbsp;'
                f'<span style="background:{status_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{status_text}</span></div>', unsafe_allow_html=True)

    # Self-healing engine initialization: Re-instantiate if stale or missing methods
    if 'engine' not in st.session_state or not hasattr(st.session_state.engine, 'predict'):
        st.session_state.engine = AIEngine()

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
        st.markdown("### 🧠 AI Predictor Pro")
        page = st.radio("Navigate", [
            "🏠 Explore", "🔮 AI Prediction", "📈 AI Backtester", "🔍 Stock Screener", "📰 Market News",
            "📊 All Stocks", "🏆 Top Movers"
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
            <div style="background: #1e3a8a; border: 1px solid #3b82f6; padding: 18px; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 1.2rem; margin-right: 12px;">📉</span>
                    <span style="color: #60a5fa; font-size: 1.1rem; font-weight: 700;">Max Loss: ₹{max_loss:,.0f}</span>
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 12px;">💰</span>
                    <span style="color: #38bdf8; font-size: 1.1rem; font-weight: 700;">Target Profit: ₹{target_profit:,.0f}</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("**Risk Management Engine**")
        st.caption(f"📍 Position sizing is based on ₹{max_loss:,.0f} risk.")
        st.caption(f"📍 Targets are set at {reward_ratio}x your risk.")
        st.caption("🤖 **Auto-Verification Active**: Tracking SL/Target hits.")
        
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
    elif page == "📊 All Stocks":
        page_all_stocks()
    elif page == "🏆 Top Movers":
        page_top_movers()

    st.markdown("---")
    st.caption("🧠 AI Market Predictor Pro • BSE • NSE • MoneyControl")

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
    """Step 6: Sector Money Flow Engine"""
    st.markdown('<div class="section-head">🔥 Sector Money Flow (Live Leaders)</div>', unsafe_allow_html=True)
    
    sectors = {
        "Banking": {"emoji": "🏦", "stocks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]},
        "IT": {"emoji": "💻", "stocks": ["TCS.NS", "INFY.NS", "HCLTECH.NS"]},
        "Auto": {"emoji": "🚗", "stocks": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS"]},
        "Energy": {"emoji": "⚡", "stocks": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS"]},
        "Pharma": {"emoji": "💊", "stocks": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"]}
    }
    
    cols = st.columns(len(sectors))
    
    for i, (name, sdata) in enumerate(sectors.items()):
        total_chg = 0
        leaders = []
        
        for stock in sdata["stocks"]:
            info = get_price_info(stock, 1)
            if info:
                total_chg += info['pct']
                if info['pct'] > 0:
                    leaders.append(f"{stock.split('.')[0]} 🔥")
                else:
                    leaders.append(f"{stock.split('.')[0]} ❄️")
        
        avg_chg = total_chg / len(sdata["stocks"]) if sdata["stocks"] else 0
        status = "STRONG 🔥" if avg_chg > 0.5 else "WEAK ❌" if avg_chg < -0.5 else "NEUTRAL ⚖️"
        color = "#00b386" if avg_chg > 0 else "#eb5b3c"
        
        with cols[i]:
            st.markdown(f"""
                <div style="background: {color}10; border: 1px solid {color}30; padding: 15px; border-radius: 12px; height: 160px;">
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">{sdata['emoji']} {name}</div>
                    <div style="font-size: 1.2rem; font-weight: 900; color: {color}; margin: 5px 0;">{status}</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {color};">{avg_chg:+.2f}%</div>
                    <div style="font-size: 0.65rem; color: #64748b; margin-top: 10px; border-top: 1px solid #334155; padding-top: 5px;">
                        {' | '.join(leaders[:2])}
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
def page_explore():
    # Advanced Heatmap
    render_sector_heatmap()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 7: Institutional Track Record (v3)
    stats = load_advanced_stats()
    st.markdown('\u003cdiv class="section-head"\u003e🏅 Institutional Track Record\u003c/div\u003e', unsafe_allow_html=True)
    st.markdown(f'''
        \u003cdiv style="background:#0f172a; border-radius:15px; border:1px solid #334155; padding:20px; margin-bottom:25px;"\u003e
            \u003cdiv style="display:grid; grid-template-columns: repeat(4, 1fr); gap:20px;"\u003e
                \u003cdiv style="text-align:center; background:#10b98110; border-radius:10px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#10b981;"\u003e{stats['win_rate']:.1f}%\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e🎯 Win Rate\u003c/div\u003e
                \u003c/div\u003e
                \u003cdiv style="text-align:center; background:#38bdf810; border-radius:10px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#38bdf8;"\u003e{stats['profit_factor']:.2f}x\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e📊 Profit Factor\u003c/div\u003e
                \u003c/div\u003e
                \u003cdiv style="text-align:center; background:#00b38610; border-radius:10px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#00b386;"\u003e+{stats['avg_profit']:.1f}%\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e💰 Avg Win\u003c/div\u003e
                \u003c/div\u003e
                \u003cdiv style="text-align:center; background:#ef444410; border-radius:10px; padding:15px;"\u003e
                    \u003cdiv style="font-size:2rem; font-weight:950; color:#ef4444;"\u003e-{stats['avg_loss']:.1f}%\u003c/div\u003e
                    \u003cdiv style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:5px;"\u003e🛡️ Avg Loss\u003c/div\u003e
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
                        <div class="name">{psym}</div>
                        <div class="{pulse_class}" style="font-size: 0.6rem; padding: 2px 8px;">LIVE</div>
                    </div>
                    <div style="font-size:0.8rem; color:{color}; font-weight:700">{pat['pattern']}</div>
                    <div style="font-size:0.7rem; color:#94a3b8; border-top:1px solid #334155; margin-top:8px; padding-top:4px;">
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
                metrics = st.session_state.engine.train(symbol, prices, volumes)
                if not metrics:
                    st.error("AI Training failed. Not enough historical patterns found for this stock. Try a more volatile symbol.")
                    return
                
                results = []
                # Simulate last 60 trading days
                test_len = 60
                wins = 0
                trades = 0
                prog = st.progress(0, text="Simulating trades...")
                for i in range(len(df) - test_len, len(df) - 5):
                    prog.progress((i - (len(df)-test_len)) / (test_len - 5), text=f"Simulating Day {i}...")
                    # Slice data up to point i
                    sub_df = df.iloc[:i]
                    sub_prices = sub_df['Close'].tolist()
                    sub_volumes = sub_df['Volume'].tolist()
                    
                    # Predict for T+3
                    pred = st.session_state.engine.predict(symbol, sub_prices, sub_volumes, df=sub_df)
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
    st.markdown(f'''<div style="background: #1e293b; border: 1px solid #334155; padding: 15px; border-radius: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #667eea;">
<div>
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">AI System Accuracy</div>
<div style="font-size: 1.5rem; font-weight: 900; color: #f8fafc;">{acc_ratio:.1%}</div>
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
                
                if is_intra:
                    df_run, _ = fetch_stock(symbol, interval='15m', period='14d')
                else:
                    df_run = df_1d # Daily is main for swing

            if df_run is not None and len(df_run) > 30:
                prices = df_run['Close'].dropna().astype(float).tolist()
                volumes = df_run['Volume'].dropna().astype(float).tolist()
                
                with st.spinner("📰 Analyzing news..."): 
                    # Fetch Local and Global separately for comparison
                    local_news = fetch_market_news(f"{symbol} share stock market news")
                    master_sent = get_master_market_sentiment()
                    
                    st.session_state.pred_catalyst = master_sent['global_reason']
                    st.session_state.pred_indian_news = analyze_news(local_news)[1][:5]
                    st.session_state.pred_global_news = master_sent['global_headlines']
                    sent = master_sent['score']

                with st.spinner("🧠 Training Engine..."):
                    metrics = st.session_state.engine.train(symbol, prices, volumes, sent)
                    st.session_state.pred_metrics = metrics
                
                with st.spinner("📡 TradingView Bias..."):
                    tv_sentiment = fetch_tv_sentiment(symbol, mapped)
                
                if metrics:
                    res = st.session_state.engine.predict(symbol, prices, volumes, sent, tv_sent=tv_sentiment, intraday=is_intra, df=df_run, df_1h=df_1h, df_1d=df_1d)
                    st.session_state.pred_results = res
                    
                    # NEW: Step 1, 3 & 5 Integration
                    entry_price = live_price if live_price else float(df_run.iloc[-1]['Close'])
                    timing = st.session_state.engine.detect_entry_timing(df_run)
                    liquidity = st.session_state.engine.detect_liquidity(df_run)
                    risk_params = st.session_state.engine.calculate_risk_parameters(
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
<div style="background:{sr_color}18; border:2px solid {sr_color}; border-left:8px solid {sr_color};
     padding:18px 24px; border-radius:14px; margin-bottom:20px;">
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
<div style="background:{guide_color}15; border:1px solid {guide_color}40; border-left:6px solid {guide_color};
     padding:16px 20px; border-radius:12px; margin-top:10px;">
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

        # 4. EXECUTIVE REASONING SECTION (Separated Indian & Global News)
        st.markdown('<div class="section-head">🧠 AI Sentiment Pulse (Local vs Global)</div>', unsafe_allow_html=True)
        
        catalyst = st.session_state.get('pred_catalyst', 'No major catalyst detected.')
        st.markdown(f'''<div style="background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3); border-left: 6px solid #6366f1; padding: 15px 20px; border-radius: 10px; margin-bottom: 20px;">
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 800; margin-bottom: 5px;">Unified Market Driver (Primary Reason)</div>
<div style="font-size: 1.1rem; color: #f8fafc; font-weight: 600;">{catalyst}</div>
</div>''', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("### 🇮🇳 Indian Catalysts")
            i_news = st.session_state.get('pred_indian_news', [])
            for h in i_news:
                scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(h['label'],'sentiment-neu')
                st.markdown(f'''<div class="news-card"><span class="{scls}">[{h["label"].upper()}]</span> {h["title"]}</div>''', unsafe_allow_html=True)
        with sc2:
            st.markdown("### 🌎 Global Market Drivers")
            g_news = st.session_state.get('pred_global_news', [])
            for h in g_news:
                scls = {'positive':'sentiment-pos','negative':'sentiment-neg'}.get(h['label'],'sentiment-neu')
                st.markdown(f'''<div class="news-card"><span class="{scls}">[{h["label"].upper()}]</span> {h["title"]}</div>''', unsafe_allow_html=True)

        # 5. CANDLE CHART
        st.markdown('<div class="section-head">📊 Market Pulse & Patterns</div>', unsafe_allow_html=True)
        st.plotly_chart(build_candle_chart(df.tail(60), symbol), use_container_width=True)

        # 6. TECHNICAL PATTERN INSET (Arranged Under the Chart)
        res = detect_candle_pattern(df.tail(3)); live_res = analyze_live_candle(df)
        p_cls = "pulse-green" if "UP" in live_res["bias"] else "pulse-red" if "DOWN" in live_res["bias"] else ""
        
        st.markdown(f'''
            <div class="pattern-inset">
                <div style="text-align:center; flex:1;">
                    <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Trend</div>
                    <div class="{p_cls}" style="margin:5px 0; font-size:0.85rem;">{live_res["bias"]}</div>
                    <div style="font-size:1.1rem; font-weight:900; color:{live_res["color"]};">{live_res["pct"]:+.2f}%</div>
                </div>
                <div style="width:1px; height:40px; background:#334155;"></div>
                <div style="text-align:center; flex:2; padding: 0 15px;">
                    <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Candlestick Analysis</div>
                    <div style="font-size:1rem; font-weight:800; color:#f8fafc; margin-top:4px;">{res["pattern"]}</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">{res["advice"]}</div>
                </div>
                <div style="width:1px; height:40px; background:#334155;"></div>
                <div style="text-align:center; flex:1;">
                    <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Volume Signal</div>
                    <div style="font-size:0.95rem; font-weight:700; color:{'#10b981' if df.iloc[-1]['Volume'] > df['Volume'].tail(20).mean() * 1.5 else '#cbd5e1'}; margin-top:5px;">
                        {'SPIKING ⚡' if df.iloc[-1]['Volume'] > df['Volume'].tail(20).mean() * 1.5 else 'NORMAL'}
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 7. WHY THIS PREDICTION? (Logic Explanation)
        st.markdown('<div class="section-head">🔍 Why This Prediction?</div>', unsafe_allow_html=True)
        today_res = pred['today']
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.metric("ML Probability", f"{today_res['ml_prob']:.1%}", help="Probability from Random Forest & Gradient Boosting ensembles.")
        with exp_col2:
            st.metric("Technical Score", f"{today_res['tech_score']:.1%}", help="Score based on RSI and EMA crossovers.")
        with exp_col3:
            st.metric("Pattern Score", f"{today_res['pattern_score']:.1%}", help="Score from candlestick pattern recognition.")

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
        session_phase = st.session_state.engine.get_market_session()
        
        # Badge Styling (Step 7)
        badge_html = ""
        if "READY" in timing: badge_html = f'<span style="background:#f59e0b; color:white; padding:4px 10px; border-radius:6px; font-size:0.6rem; margin-right:8px;">READY</span>'
        elif "DETECTED" in timing: badge_html = f'<span style="background:#10b981; color:white; padding:4px 10px; border-radius:6px; font-size:0.6rem; margin-right:8px;">DETECTED</span>'
        
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
        if "NO TRADE" in today_sig:
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
<div style="display:grid; grid-template-columns: 1.5fr 1fr; gap:25px; align-items:center;">
<div>
<h1 style="margin:0; color:{v_col}; font-size: 3.2rem; font-weight: 950; letter-spacing:-1px;">{v_sig}</h1>
<div style="font-size:1.1rem; color:#94a3b8; font-weight:600; margin-top:10px;">{tamil_summary}</div>
</div>
<div style="background:rgba(0,0,0,0.3); padding:20px; border-radius:15px; border:1px solid rgba(255,255,255,0.05); text-align:left;">
<div style="font-size:0.85rem; color:#64748b; margin-bottom:12px; font-weight:700; text-transform:uppercase;">Execution Levels</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#94a3b8;">Entry</span>
<span style="font-weight:800; color:#f8fafc;">₹{rp.get('entry', 0):,.2f}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#94a3b8;">Stop Loss</span>
<span style="font-weight:800; color:#ef4444;">₹{rp.get('sl', 0):,.2f}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="color:#94a3b8;">Target</span>
<span style="font-weight:800; color:#10b981;">₹{rp.get('target', 0):,.2f}</span>
</div>
<hr style="border:0.5px solid #334155; margin:10px 0;">
<div style="display:flex; justify-content:space-between; font-size:0.8rem;">
<span style="color:#64748b;">Risk Amount (Max Loss)</span>
<span style="color:#ef4444; font-weight:800;">₹{rp.get('risk_amt', 0):,.0f}</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:4px;">
<span style="color:#64748b;">Reward (Target Profit)</span>
<span style="color:#10b981; font-weight:800;">₹{rp.get('profit_amt', 0):,.0f}</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:4px;">
<span style="color:#64748b;">Quantity to Buy</span>
<span style="color:#38bdf8; font-weight:800;">{rp.get('pos_size', 0)} Shares</span>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:4px;">
<span style="color:#64748b;">Risk/Reward Ratio</span>
<span style="color:#f8fafc; font-weight:700;">{rp.get('risk_reward', '1:2')}</span>
</div>
<div style="margin-top:20px; padding-top:10px; border-top:1px solid #334155;">
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


# ── PAGE: All Stocks ──────────────────────────────────────────────────────
def page_all_stocks():
    st.subheader("📊 All Stocks — Live Prices")
    category = st.selectbox("Select Category", list(DASHBOARD_CATEGORIES.keys()))
    symbols = DASHBOARD_CATEGORIES[category]
    # Remove duplicates
    symbols = list(dict.fromkeys(symbols))
    
    rows = []
    prog = st.progress(0, text="Loading...")
    for i, sym in enumerate(symbols):
        prog.progress((i+1)/len(symbols), text=f"Fetching {sym}...")
        info = get_price_info(sym, 5)
        if info:
            rows.append({'Stock': sym, 'Price': f"{info['currency']}{info['price']:,.2f}",
                'Change': f"{info['change']:+.2f}", 'Change%': f"{info['pct']:+.2f}%",
                'Volume': f"{info['volume']:,}"})
    prog.empty()

    if rows:
        rdf = pd.DataFrame(rows)
        def styl(val):
            if '+' in str(val): return 'color: #10b981; font-weight: bold'
            elif '-' in str(val): return 'color: #ef4444; font-weight: bold'
            return ''
        st.dataframe(rdf.style.map(styl, subset=['Change','Change%']),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("No data available")


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
