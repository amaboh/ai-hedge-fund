import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

def get_market_indices(start_date: str, end_date: str) -> Dict[str, float]:
    """Fetch major market indices performance."""
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ"
    }
    
    performance = {}
    for symbol, name in indices.items():
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                # Proper way to get float values from Series
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                performance[name] = ((end_price - start_price) / start_price) * 100
            else:
                performance[name] = 0.0
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            performance[name] = 0.0
    
    return performance

def get_bond_yields() -> Dict[str, float]:
    """Fetch current bond yields."""
    bonds = {
        "^TNX": "10-Year Treasury",
        "^IRX": "13-Week Treasury",
        "^FVX": "5-Year Treasury"
    }
    
    yields = {}
    for symbol, name in bonds.items():
        try:
            data = yf.Ticker(symbol).history(period="1d")
            if not data.empty:
                yields[name] = data['Close'].iloc[-1]
            else:
                yields[name] = 0.0
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            yields[name] = 0.0
    
    return yields

def get_sector_performance(start_date: str, end_date: str) -> Dict[str, float]:
    """Fetch sector ETF performance."""
    sectors = {
        "XLK": "Technology",
        "XLF": "Financial",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrial",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate"
    }
    
    performance = {}
    for symbol, name in sectors.items():
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                # Proper way to get float values from Series
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                performance[name] = ((end_price - start_price) / start_price) * 100
            else:
                performance[name] = 0.0
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            performance[name] = 0.0
    
    return performance

def get_commodity_prices() -> Dict[str, Dict[str, float]]:
    """Fetch major commodity prices and their recent changes."""
    commodities = {
        "GC=F": "Gold",
        "CL=F": "Crude Oil",
        "SI=F": "Silver",
        "HG=F": "Copper"
    }
    
    prices = {}
    for symbol, name in commodities.items():
        try:
            data = yf.Ticker(symbol).history(period="1mo")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                month_ago_price = data['Close'].iloc[0]
                prices[name] = {
                    "price": float(current_price),
                    "change": float(((current_price - month_ago_price) / month_ago_price) * 100)
                }
            else:
                prices[name] = {"price": 0.0, "change": 0.0}
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            prices[name] = {"price": 0.0, "change": 0.0}
    
    return prices

def get_currency_strength() -> Dict[str, Dict[str, float]]:
    """Fetch major currency pairs performance."""
    pairs = {
        "EURUSD=X": "EUR/USD",
        "JPY=X": "USD/JPY",
        "GBPUSD=X": "GBP/USD",
        "CNY=X": "USD/CNY"
    }
    
    exchange_rates = {}
    for symbol, name in pairs.items():
        try:
            data = yf.Ticker(symbol).history(period="1mo")
            if not data.empty:
                current_rate = data['Close'].iloc[-1]
                month_ago_rate = data['Close'].iloc[0]
                exchange_rates[name] = {
                    "rate": float(current_rate),
                    "change": float(((current_rate - month_ago_rate) / month_ago_rate) * 100)
                }
            else:
                exchange_rates[name] = {"rate": 0.0, "change": 0.0}
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
            exchange_rates[name] = {"rate": 0.0, "change": 0.0}
    
    return exchange_rates

def get_economic_indicators() -> Dict[str, float]:
    """Fetch or simulate various economic indicators."""
    # In a real implementation, this would fetch from an economic data API
    return {
        "gdp_growth": 2.5,
        "inflation_rate": 3.1,
        "unemployment_rate": 3.8,
        "consumer_confidence": 105.7,
        "retail_sales_growth": 1.8,
        "manufacturing_pmi": 52.3
    }