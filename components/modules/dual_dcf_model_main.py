#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Stock Valuation Program 
### Goal: create python algorithm to query financials and historical data to formulate multiple valuation models


# In[2]:


import pandas as pd
import requests
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
import time
import random


# In[3]:


from sec_api import XbrlApi
import yfinance as yf


# In[4]:


headers = {'User-Agent': "ericbstratford@gmail.com"}


tickers_cik = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
tickers_cik = pd.json_normalize(pd.json_normalize(tickers_cik.json(), max_level=0).values[0])
tickers_cik["cik_str"] = tickers_cik["cik_str"].astype(str).str.zfill(10)
tickers_cik.set_index("ticker",inplace=True)

# response = requests.get("https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Assets.json", headers=headers)
# assets_timeserie = pd.json_normalize(response.json()["units"]["USD"])
# assets_timeserie["filed"] = pd.to_datetime(assets_timeserie["filed"])
# assets_timeserie = assets_timeserie.sort_values("end")


# In[5]:


def get_tags(ticker):
    cik_str = "CIK"+str(tickers_cik.cik_str.loc[ticker])
    title = str(tickers_cik.title.loc[ticker])
    request = "https://data.sec.gov/api/xbrl/companyfacts/"+cik_str+".json"
    res = requests.get(request, headers = headers).json()
    tags = list(res['facts']['us-gaap'].keys())
    return tags
# get_tags("PEP")


# In[6]:


def get_fact(ticker, tag):
    request = "https://data.sec.gov/api/xbrl/companyconcept/CIK"+str(tickers_cik.cik_str.loc[ticker])+"/us-gaap/"+tag+".json"
    response = requests.get(request, headers = headers)
    df = pd.json_normalize(response.json()["units"]['USD'])
    df["filed"] = pd.to_datetime(df["filed"])
    df["end"] = pd.to_datetime(df["end"])
    df.sort_values("end", inplace=True)
    df = df.groupby("end")[['val']].mean().rename(columns={"val":tag})
    return df

def get_other(ticker, tag, unit):
    request = "https://data.sec.gov/api/xbrl/companyconcept/CIK"+str(tickers_cik.cik_str.loc[ticker])+"/us-gaap/"+tag+".json"
    response = requests.get(request, headers = headers)
    df = pd.json_normalize(response.json()["units"][unit])
    df["filed"] = pd.to_datetime(df["filed"])
    df["end"] = pd.to_datetime(df["end"])
    df.sort_values("end", inplace=True)
    df = df.groupby("end")[['val']].mean().rename(columns={"val":tag})
    return df

def get_market_val(ticker, tag):
    market_data = yf.Ticker(ticker)
    out = market_data.info[tag]
    return out


# In[7]:


# yf.Ticker("AAPL").info.keys()
# yf.Ticker("SPY").history(period='5y')
# yf.Ticker("PEP").get_cash_flow()


# In[8]:


# get_market_val("QQQ", "fiveYearAverageReturn")
# get_market_val("SPY", "fiveYearAverageReturn")


# In[9]:


def discounted_cash_flow(ticker):
    facts = ["OperatingIncomeLoss", "AccountsPayableCurrent", "AccountsReceivableNetCurrent", 
             ("EffectiveIncomeTaxRateContinuingOperations", "pure"), 
            ]
    da1 = ["DepreciationAndAmortization"]
    da2 = ["Depreciation", "AmortizationOfIntangibleAssets"]
    ce1 = ["PaymentsToAcquirePropertyPlantAndEquipment"]
    ce2 = ["PaymentsToAcquireProductiveAssets"]
    
    tags = get_tags(ticker)
    
#     return yf_dcf(ticker)
    
    master_df = pd.DataFrame()
    for fact in facts:
        try:
            if type(fact)==tuple:
                fact_df = get_other(ticker, fact[0], fact[1])
            else:
                fact_df = get_fact(ticker, fact)
        except:
            print(ticker, "Unavailable Cash Flow:", fact)
            return
        
        if master_df.empty:
            master_df = fact_df
        else:
            master_df = master_df.merge(fact_df, how='left', on='end')
    
    # Depreciation and Amortization
    if da1[0] in tags:
        fact_df = get_fact(ticker, da1[0])
        master_df = master_df.merge(fact_df, how='left', on='end')
    elif da2[0] in tags and da2[1] in tags:
        fact_df = get_fact(ticker, da2[0])
        master_df = master_df.merge(fact_df, how='left', on='end')
        fact_df = get_fact(ticker, da2[1])
        master_df = master_df.merge(fact_df, how='left', on='end')
    else:
        raise ValueError("No Valid Depr&Amor")
    
    if ce1[0] in tags:
        fact_df = get_fact(ticker, ce1[0])
        master_df = master_df.merge(fact_df, how='left', on='end')
    elif ce2[0] in tags:
        fact_df = get_fact(ticker, ce2[0])
        master_df = master_df.merge(fact_df, how='left', on='end')
    else:
        raise ValueError("No Valid CapEx")
        
    
        
    

    # Calculate Operating Cash Flow
    if "DepreciationAndAmortization" not in master_df.columns:
#         master_df['Depreciation'] = master_df['Depreciation'].ffill()/4
        master_df['NonCashExpenses'] = master_df['Depreciation'] + master_df['AmortizationOfIntangibleAssets']
    else:
        master_df['NonCashExpenses'] = master_df['DepreciationAndAmortization']
        
    master_df = master_df.fillna(0)
    
    master_df['AccRecNet'] = master_df['AccountsReceivableNetCurrent']
    master_df['AccPayChg'] = master_df['AccountsPayableCurrent'].diff()
    master_df['AccRecChg'] = master_df['AccRecNet'].diff()
    master_df['NetWorkCapChg'] = master_df['AccPayChg'] + master_df['AccRecChg']
    master_df['Taxes'] = master_df['OperatingIncomeLoss'] * master_df['EffectiveIncomeTaxRateContinuingOperations']
    master_df['OCF'] = master_df['OperatingIncomeLoss'] + master_df['NonCashExpenses'] - master_df['NetWorkCapChg'] - master_df['Taxes']

    # Calculate Free Cash Flow
    if "PaymentsToAcquirePropertyPlantAndEquipment" in master_df.columns:
        master_df['CapEx'] = master_df['PaymentsToAcquirePropertyPlantAndEquipment']
    else:
        master_df['CapEx'] = master_df['PaymentsToAcquireProductiveAssets']
    master_df['FCF'] = master_df['OCF'] - master_df['CapEx']
    
    master_df.dropna(how='any', inplace=True)
    master_df = master_df[master_df.index >= pd.to_datetime("2014-01-01")]
    
    
    # Project Future Free Cash Flows
    df = master_df[['FCF']]
    df.reset_index(inplace=True)
    def calculate_cagr(first, last, periods):
        return (last / first) ** (1 / periods) - 1
    first_fcf = df['FCF'].iloc[0]
    last_fcf = df['FCF'].iloc[-1]
    n_periods = len(df) - 1
    cagr = calculate_cagr(first_fcf, last_fcf, n_periods)
    forecast_period = 10 * 4  # 10 years in quarters
    future_dates = pd.date_range(start=df['end'].iloc[-1], periods=forecast_period + 1, freq='QE')
    future_fcfs = [df['FCF'].iloc[-1] * (1 + cagr) ** i for i in range(1, forecast_period + 1)]
    future_df = pd.DataFrame({'end': future_dates[1:], 'FCF': future_fcfs})
    df_projected = future_df
    
    
    # Query Market Values
    market_cap = get_market_val(ticker, "marketCap")
    total_debt = get_market_val(ticker, "totalDebt")
    shares_outstanding = get_market_val(ticker, "sharesOutstanding")
    current_price = get_market_val(ticker, "currentPrice")
    market_enterprise_value = get_market_val(ticker, "enterpriseValue")
    forward_pe = get_market_val(ticker, "forwardPE")
    ebitda = get_market_val(ticker, "ebitda")
    total_cash = get_market_val(ticker, "totalCash")
    interest_expense = get_fact(ticker, "InterestExpense").iloc[-1,0]
    
    
    # WACC - Weighted Average Cost of Capital
    risk_free_rate = 0.03  # General Risk-free rate
    market_risk_premium = 0.06  # General Market risk premium
    tax_rate = 0.21  # Corporate tax rate

    beta = get_market_val(ticker,"beta")

    cost_of_equity = risk_free_rate + beta * market_risk_premium
    cost_of_debt = (interest_expense / total_debt) * (1 - tax_rate)

    market_value_equity = market_cap
    market_value_debt = total_debt

    equity_weight = market_value_equity / (market_value_equity + market_value_debt)
    debt_weight = market_value_debt / (market_value_equity + market_value_debt)

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt
    q_wacc = (1 + wacc) ** (1/4) - 1 # Transform WACC to Quarterly 

    
    discount_rate = q_wacc
#     print("wacc:", wacc)
    

    g = cagr
#     print("g:", g)
    FCF_terminal_qtr = df_projected['FCF'].iloc[-1]
    FCF_terminal_yr = FCF_terminal_qtr*4
    terminal_value = FCF_terminal_yr * (1 + g) / (wacc - g)
    
    # Present-Value
    # Create an array to store the discounted cash flows
    discounted_cash_flows = []
    # Discount each projected cash flow to its present value
    for i, cash_flow in enumerate(df_projected['FCF']):
        discounted_cash_flow = cash_flow / (1 + discount_rate) ** (i + 1)  # Discounted to present value
        discounted_cash_flows.append(discounted_cash_flow)
    # Discount the terminal value to its present value
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** len(df_projected))
    pv_fcf = np.sum(discounted_cash_flows)
    pv_tv = discounted_terminal_value
    enterprise_value = pv_fcf + pv_tv
    
    # Intrinsic Value
    equity_value = enterprise_value - total_debt
    intrinsic_value_per_share = equity_value / shares_outstanding
#     print("Current Price:", current_price)
#     print("DCF Intrinsic Value:", intrinsic_value_per_share)

    return intrinsic_value_per_share, current_price

# discounted_cash_flow("AAPL")


# In[10]:


# ticker = "GE"
# print(get_market_val(ticker, "shortName"))
# print(get_market_val(ticker, "sectorKey"))
# print(get_market_val(ticker, "industryKey"))


# In[11]:


# Industry List. Manually generated to avoid repetetive webscraping calls
sector_industry_tickers = {
    'technology': {
        'software-infrastructure': ['MSFT', 'ORCL', 'ADBE', 'PANW', 'SNPS', 'CRWD', 'PLTR', 'FTNT', 'SQ', 'MDB'],
        'software-application': ['CRM', 'SAP', 'INTU', 'NOW', 'UBER', 'CDNS', 'SHOP', 'WDAY', 'ROP', 'SNOW'],
        'computer-hardware' : ['DELL', 'ANET', 'SMCI', 'HPQ', 'WDC', 'NTAP', 'STX', 'PSTG', 'LOGI', 'IONQ'],
        'electronic-components': ['APH', 'TEL', 'GLW', 'JBL', 'FLEX'],
        'scientific-technical-instruments': ['GRMN', 'KEYS', 'FTV', 'TDY', 'TRMB', 'MKSI', 'COHR', 'CGNX', 'ST'],
        'semiconductors': ['NVDA', 'TSM', 'AVGO', 'AMD', 'QCOM', 'TXN', 'MU', 'INTC', 'ARM', 'ADI'],
        'information-technology-services': ['ACN', 'IBM', 'FI', 'INFY', 'FIS', 'IT', 'CTSH', 'CDW', 'WIT', 'GIB'],
        'communication-equipment': ['CSCO', 'MSI', 'HPE', 'NOK', 'ERIC', 'ZBRA', 'JNPR', 'UI', 'CIEN', 'SATS'],
        'consumer-electronics': ['AAPL', 'SONY', 'LPL', 'VZIO', 'SONO', 'GPRO', 'VUZI', 'KOSS'],
        'electronics-computer-distribution': ['SNX', 'ARW', 'NSIT', 'AVT', 'SCSC', 'SNPO'],
        'semiconductor-equipment-materials': ['ASML', 'AMAT', 'LRCX', 'KLAX', 'TER', 'AMKR', 'ENTG'],
        'solar': ['FSLR', 'ENPH', 'NXT', 'SEDG', 'RUN', 'ARRY', 'JKS', 'SHLS', 'NOVA', 'SPWR'],
    },
    'communication-services': {
        'telecom-services': ['TMUS', 'VZ', 'CMCSA', 'T', 'AMX', 'CHTR', 'BCE', 'ORAN'],
        'publishing': ['PSO', 'NYT', 'WLY', 'SCHL', 'GCI'],
        'entertainment': ['NFLX', 'DIS', 'LYV', 'WBD', 'FWONK', 'WMG', 'NWSA', 'FOXA', 'SIRI', 'PARA'],
        'electronic-gaming-multimedia': ['NTES', 'EA', 'TTWO', 'RBLX', 'BILI', 'PLTK'],
        'advertising-agencies' : ['OMC', 'IPG', 'ZD', 'CCO', 'MGNI', 'STGW', 'ADV', 'CDLX', 'IAS', 'CRTO'],
        'broadcasting': ['LSXMK', 'TGNA', 'GTN', 'FUBO', 'IHRT', 'MDIA'],
        'internet-content-information': ['GOOG', 'META', 'TCEHY', 'PROSY', 'SPOT', 'DASH', 'BIDU', 'PINS', 'SNAP', 'TME'],
    },
    'healthcare': {
        'drug-manufacturers-general': ['LLY', 'JNJ', 'MRK', 'ABBV', 'AZN', 'PFE', 'NVS', 'AMGN', 'GILD', 'BMY'],
        'biotechnology': ['NVO', 'VRTX', 'REGN', 'MRNA', 'BNTX', 'RPRX', 'BMRN', 'TECH', 'INCY', 'ROIV'],
        'drug-manufacturers-specialty-generic': ['ZTS', 'TAK', 'HLN', 'TEVA', 'VTRS', 'CTLT', 'ELAN', 'ITCI'],
        'pharmaceutical-retailers': ['WBA', 'HITI', 'PETS', 'AEGY', 'ALST', 'MEDS'],
        'medical-devices': ['ABT', 'SYK', 'MDT', 'BSX', 'EW', 'DXCM', 'PHG', 'ZBH', 'SWAV', 'PODD'],
        'diagnostics-research': ['TMO', 'DHR', 'A', 'IDXX', 'IQV', 'ILMN', 'NTRA', 'EXAS', 'CRL', 'QGEN'],
        'healthcare-plans': ['UNH', 'ELV', 'CI', 'CVS', 'HUM', 'CNC', 'OSCR', 'ALHC', 'CLOV', 'MOH'],
        'medical-care-facilities': ['HCA', 'FMS', 'THC', 'DVA', 'UHS', 'EHC', 'ACHC', 'OPCH', 'AGL', 'LFST'],
        'health-information-services': ['GEHC', 'VEEV', 'DOCS', 'RCM', 'TXG', 'GDRX', 'EVH', 'TDOC', 'MDRX', 'SDGR'],
        'medical-instruments-supplies': ['ISRG', 'BDX', 'RMD', 'WST', 'COO', 'BAX', 'HOLX'],
        'medical-distribution': ['MCK', 'COR', 'CAH', 'HSIC', 'PDCO', 'OMI', 'GEGP', 'ITNS', 'KOSK', 'COSM'],
    },
    'financial-services': {
        'asset-management': ['BX', 'BLK', 'KKR', 'BN', 'APO', 'ARES', 'BK', 'AMP', 'OWL', 'TROW'],
        'banks-diversified': ['JPM', 'BAC', 'WFC', 'HSBC', 'C', 'MUFG', 'TD', 'UBS', 'SAN', 'BCS'],
        'banks-regional': ['HDB', 'IBN', 'USB', 'ITUB', 'PNC', 'NU', 'TFC', 'MFG', 'LYG', 'DB'],
        'capital-markets': ['MS', 'GS', 'SCHW', 'IBKR', 'RJF', 'HOOD', 'XP', 'NMR', 'FUTU'],
        'insurance-life': ['MET', 'AFL', 'MFC', 'PRU', 'UNM', 'GL', 'LNC', 'CNO', 'GNW', 'BHF'],
        'insurance-reinsurance': ['EG', 'RGA', 'RNR', 'SPNT', 'HG', 'GLRE', 'MHLD'],
        'insurance-brokers': ['MMC', 'AON', 'AJG', 'WTW', 'BRO', 'BRP', 'SLQT', 'ERIE', 'CRVL', 'RELI'],
        'shell-companies': ['CVII', 'AACT', 'ANSC', 'NETD', 'RRAC', 'IPXX', 'HYAC', 'JWSM', 'TRTL', 'PLAO'],
        'mortgage-finance': ['RKT', 'UWMC', 'FNMA', 'COOP', 'PFSI', 'FMCC'],
        'financial-data-stock-exchanges': ['SPGI', 'ICE', 'CME', 'NDAQ', 'MCO', 'COIN', 'CBOE', 'LNSTY', 'MSCI'],
        'insurance-property-casualty': ['PGR', 'CB', 'TRV', 'ALL', 'MSADY', 'WRB', 'CINF'],
        'insurance-specialty': ['FNF', 'RYAN', 'AIZ', 'AXS', 'FAF', 'ESNT', 'MTG', 'ACT', 'RDN', 'AGO', 'NMIH'],
        'insurance-diversified': ['BRK-A', 'AIG', 'ACGL', 'SLF', 'EQH', 'AEG', 'ORI', 'GSHD', 'FIHL', 'WDH'],
        'financial-conglomerates': ['IX', 'VOYA', 'RILY', 'TREE'],
        'credit-services': ['V', 'MA', 'AXP', 'PYPL', 'COF', 'DFS', 'SYF', 'ALLY', 'SOFI', 'LU']
    },
    'consumer-cyclical': {
        'Retail': ['AMZN', 'BABA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'TGT', 'ROST'],
        'Automobiles': ['TSLA', 'TM', 'GM', 'F', 'HMC', 'NIO', 'RACE', 'LKQ', 'XPEV', 'LI'],
        'Hotels, Restaurants & Leisure': ['BKNG', 'MAR', 'HLT', 'SBUX', 'YUM', 'CMG', 'MGM', 'RCL', 'CCL', 'EXPE']
    },
    'consumer-defensive': {
        'Beverages': ['KO', 'PEP', 'BUD', 'DEO', 'STZ', 'MNST', 'TAP', 'SAM', 'KDP', 'CCE'],
        'Food & Staples Retailing': ['WMT', 'COST', 'TGT', 'KR', 'WBA', 'SYY', 'CVS', 'WMT', 'COST', 'JD'],
        'Household Products': ['PG', 'CL', 'CHD', 'KMB', 'CLX', 'SPB', 'EPC', 'HRL', 'EL', 'NUS']
    },
    'industrials': {
        'Aerospace & Defense': ['BA', 'LMT', 'NOC', 'GD', 'RTN', 'TXT', 'LHX', 'HII', 'TDY', 'HEI'],
        'Machinery': ['CAT', 'DE', 'ITW', 'DOV', 'ROK', 'PH', 'ETN', 'IR', 'SNA', 'FLS'],
        'Transportation': ['FDX', 'UPS', 'UNP', 'NSC', 'CSX', 'LUV', 'DAL', 'AAL', 'UAL', 'JBHT']
    },
    'energy': {
        'Oil, Gas & Consumable Fuels': ['XOM', 'CVX', 'COP', 'OXY', 'PSX', 'EOG', 'MPC', 'VLO', 'HAL', 'SLB'],
        'Energy Equipment & Services': ['SLB', 'HAL', 'BKR', 'FTI', 'NOV', 'DRQ', 'FET', 'XPRO', 'TDW', 'NBR']
    },
    'utilities': {
        'Electric Utilities': ['NEE', 'DUK', 'SO', 'EXC', 'AEP', 'D', 'SRE', 'XEL', 'PPL', 'PEG'],
        'Multi-Utilities': ['DUK', 'SO', 'EXC', 'PEG', 'WEC', 'ED', 'DTE', 'AEE', 'PNW', 'ES']
    },
    'real-estate': {
        'Equity REITs': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'EQR', 'AVB', 'ESS', 'MAA'],
        'Real Estate Management & Development': ['CBRE', 'CWK', 'JLL', 'CIGI', 'HFF', 'RLGY', 'HOUS', 'SVN', 'NMRK', 'MC']
    },
    'basic-materials': {
        'Chemicals': ['LIN', 'DD', 'SHW', 'PPG', 'APD', 'EMN', 'ALB', 'LYB', 'CF', 'FMC'],
        'Metals & Mining': ['BHP', 'RIO', 'VALE', 'FCX', 'NEM', 'SCCO', 'STLD', 'ATI', 'CMC', 'WOR']
    }
}


# In[16]:


def portfolio_dist(stocks_list, preference):
    preferences = {'conservative': 1, 'moderate': 2, 'aggressive': 3}
    preference_level = preferences.get(preference.lower(), 2)
    
    stock_recs = {}
    stock_weights = {}
    total_weight = 0
    
    for ticker in stocks_list:
        rec = None
        weight = 0
        
        # ETF handling
        try:
            sector_key = get_market_val(ticker, "sectorKey")
            industry_key = get_market_val(ticker, "industryKey")
            pe = get_market_val(ticker, "forwardPE")
            peg_ratio = get_market_val(ticker, "pegRatio")
        
            # Get list of comparables
            if industry_key in sector_industry_tickers.get(sector_key, {}):
                comparables = sector_industry_tickers[sector_key][industry_key]
            else:
                comparables = random.sample(
                    [tick for industry in sector_industry_tickers.get(sector_key, {}).values() for tick in industry], 
                    10
                )

            # Calculate comparable PE ratio
            comparable_pe_total = 0
            comp_count = 0
            for comp in comparables:
                try:
                    comparable_pe_total += get_market_val(comp, "forwardPE")
                    comp_count += 1
                except:
                    continue
            comparable_pe_avg = comparable_pe_total/comp_count
            pe_comp_ratio = pe/comparable_pe_avg

            try:
                ivps, current_price = discounted_cash_flow(ticker)

                # Use DCF for initial recommendation
                if ivps > current_price * 1.1:
                    weight += 5
                elif ivps > current_price * 1.03:
                    weight += 4
                elif ivps < current_price * 0.9:
                    weight += -5
                elif ivps < current_price * 0.97:
                    weight += -4
                else:
                    rec = "Hold"
                    weight += 1

            except:
                # Comparable PEG ratios in absense of DCF
                peg_ratio = get_market_val(ticker, "pegRatio")
                comparable_peg_total = 0
                comp_count = 0
                for comp in comparables:
                    try:
                        comparable_peg_total += get_market_val(comp, "pegRatio")
                        comp_count += 1
                    except:
                        continue
                comparable_peg_avg = comparable_peg_total/comp_count

                # Use PEG ratio to comparable industry average
                if peg_ratio > comparable_peg_avg * 1.2:
                    weight += -3
                elif peg_ratio > comparable_peg_avg * 1.03:
                    weight += -2
                elif peg_ratio < comparable_peg_avg * 0.8:
                    weight += 3
                elif peg_ratio < comparable_peg_avg * 0.97:
                    weight += 2
                else:
                    weight += 1
                    
        except: # If ETF
            # Use SPY as baseline
            spy_ar = get_market_val("SPY", "fiveYearAverageReturn")
            avg_returns = get_market_val(ticker, "fiveYearAverageReturn")
            
            trailing_pe = get_market_val(ticker, "trailingPE")
            spy_pe = get_market_val(ticker, "trailingPE")
            
            pe_comp_ratio = trailing_pe / spy_pe
            if avg_returns > spy_ar * 1.2:
                weight += 4
            elif avg_returns > spy_ar * 1.05:
                weight += 2
            elif avg_returns < spy_ar * 0.95:
                weight += -2
            elif avg_returns < spy_ar * 0.8:
                weight += -4
            else:
                weight += 1
                
        # Use comparable PE ratios and preferences to adjust weights
        if preference_level == 1:  # Conservative
            if pe_comp_ratio < 0.9:
                weight += 1
            elif pe_comp_ratio > 1.1:
                weight -= 1
        elif preference_level == 3:  # Aggressive
            if pe_comp_ratio > 1.1:
                weight += 1
            elif pe_comp_ratio < 0.9:
                weight -= 1

        weight = max(0, weight)
        if weight > 0:
            rec = "Buy"
        elif weight > 2:
            rec = "Strong Buy"
        elif weight < 0:
            rec = "Sell"
        elif weight < -2:
            rec = "Strong Sell"
        else:
            rec = "Hold"
        stock_recs[ticker] = rec
        stock_weights[ticker] = weight
        total_weight += abs(weight)
        print(ticker, rec, weight)
        
    for ticker in stock_weights:
        stock_weights[ticker] = (stock_weights[ticker] / total_weight)
    
    return stock_recs, stock_weights
            

"""
TESTING VALUATION BELOW
"""
    
# stock_recs, stock_weights = portfolio_dist(['X', 'AAPL', 'META', 'QQQ', 'BA'], 10000, "aggressive")


# In[17]:


# stock_recs
# stock_weights

