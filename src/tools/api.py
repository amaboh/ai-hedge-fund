import os
from typing import Dict, Any, List, Optional
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

# Financial Datasets API functions
def get_prices_yf(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """Fetch price data from Yahoo Finance."""
    try:
        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        prices = []
        for date, row in ticker_data.iterrows():
            prices.append({
                "time": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"].iloc[0] if isinstance(row["Open"], pd.Series) else row["Open"]),
                "high": float(row["High"].iloc[0] if isinstance(row["High"], pd.Series) else row["High"]),
                "low": float(row["Low"].iloc[0] if isinstance(row["Low"], pd.Series) else row["Low"]),
                "close": float(row["Close"].iloc[0] if isinstance(row["Close"], pd.Series) else row["Close"]),
                "volume": float(row["Volume"].iloc[0] if isinstance(row["Volume"], pd.Series) else row["Volume"])
            })
        return prices
    except Exception as e:
        raise Exception(f"Error fetching price data from Yahoo Finance: {str(e)}")

def get_prices_fd(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """Fetch price data from Financial Datasets API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    data = response.json()
    return data.get("prices", [])

def get_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    data_source: str = "yahoo"
) -> List[Dict[str, Any]]:
    """Unified function to fetch price data from specified source."""
    try:
        if data_source == "yahoo":
            return get_prices_yf(ticker, start_date, end_date)
        return get_prices_fd(ticker, start_date, end_date)
    except Exception as e:
        if data_source == "financialdatasets":
            print(f"Financial Datasets API error: {str(e)}. Falling back to Yahoo Finance...")
            return get_prices_yf(ticker, start_date, end_date)
        raise e

def get_financial_metrics_yf(
    ticker: str,
    report_period: Optional[str] = None,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial metrics from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = [{
            "return_on_equity": info.get("returnOnEquity", 0),
            "net_margin": info.get("profitMargins", 0),
            "operating_margin": info.get("operatingMargins", 0),
            "revenue_growth": info.get("revenueGrowth", 0),
            "earnings_growth": info.get("earningsGrowth", 0),
            "book_value_growth": 0,
            "current_ratio": info.get("currentRatio", 0),
            "debt_to_equity": info.get("debtToEquity", 0) if info.get("debtToEquity") else 0,
            "free_cash_flow_per_share": info.get("freeCashflow", 0) / info.get("sharesOutstanding", 1) if info.get("sharesOutstanding") else 0,
            "earnings_per_share": info.get("trailingEps", 0),
            "price_to_earnings_ratio": info.get("trailingPE", 0),
            "price_to_book_ratio": info.get("priceToBook", 0),
            "price_to_sales_ratio": info.get("priceToSalesTrailing12Months", 0),
        }]
        return metrics
    except Exception as e:
        raise Exception(f"Error fetching financial metrics from Yahoo Finance: {str(e)}")

def get_financial_metrics_fd(
    ticker: str,
    report_period: str,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial metrics from Financial Datasets API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    data = response.json()
    return data.get("financial_metrics", [])

def get_financial_metrics(
    ticker: str,
    report_period: str,
    period: str = 'ttm',
    limit: int = 1,
    data_source: str = "yahoo"
) -> List[Dict[str, Any]]:
    """Unified function to fetch financial metrics."""
    try:
        if data_source == "yahoo":
            return get_financial_metrics_yf(ticker, report_period, period, limit)
        return get_financial_metrics_fd(ticker, report_period, period, limit)
    except Exception as e:
        if data_source == "financialdatasets":
            print(f"Financial Datasets API error: {str(e)}. Falling back to Yahoo Finance...")
            return get_financial_metrics_yf(ticker, report_period, period, limit)
        raise e

def get_insider_trades_yf(
    ticker: str,
    end_date: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Fetch insider trades from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        insider_trades = stock.insider_transactions
        
        if insider_trades is None or insider_trades.empty:
            return []
        
        trades = []
        for _, trade in insider_trades.head(limit).iterrows():
            trades.append({
                "transaction_shares": float(trade.get("Shares", 0)),
                "transaction_date": trade.index.strftime("%Y-%m-%d")[0],
                "insider_name": trade.get("Insider", ""),
                "transaction_type": trade.get("Transaction", "")
            })
        return trades
    except Exception as e:
        print(f"Warning: Error fetching insider trades from Yahoo Finance: {str(e)}")
        return []

def get_insider_trades_fd(
    ticker: str,
    end_date: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Fetch insider trades from Financial Datasets API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_lte={end_date}"
        f"&limit={limit}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    data = response.json()
    return data.get("insider_trades", [])

def get_insider_trades(
    ticker: str,
    end_date: str,
    limit: int = 5,
    data_source: str = "yahoo"
) -> List[Dict[str, Any]]:
    """Unified function to fetch insider trades."""
    try:
        if data_source == "yahoo":
            return get_insider_trades_yf(ticker, end_date, limit)
        return get_insider_trades_fd(ticker, end_date, limit)
    except Exception as e:
        if data_source == "financialdatasets":
            print(f"Financial Datasets API error: {str(e)}. Falling back to Yahoo Finance...")
            return get_insider_trades_yf(ticker, end_date, limit)
        return []

def get_market_cap_yf(ticker: str) -> float:
    """Fetch market cap from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        return float(stock.info.get("marketCap", 0))
    except Exception as e:
        raise Exception(f"Error fetching market cap from Yahoo Finance: {str(e)}")

def get_market_cap_fd(ticker: str) -> float:
    """Fetch market cap from Financial Datasets API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = f'https://api.financialdatasets.ai/company/facts?ticker={ticker}'
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    data = response.json()
    company_facts = data.get('company_facts', {})
    return float(company_facts.get('market_cap', 0))

def get_market_cap(
    ticker: str,
    data_source: str = "yahoo"
) -> float:
    """Unified function to fetch market cap."""
    try:
        if data_source == "yahoo":
            return get_market_cap_yf(ticker)
        return get_market_cap_fd(ticker)
    except Exception as e:
        if data_source == "financialdatasets":
            print(f"Financial Datasets API error: {str(e)}. Falling back to Yahoo Finance...")
            return get_market_cap_yf(ticker)
        raise e

def search_line_items_yf(
    ticker: str,
    line_items: List[str],
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial line items from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        
        if financials is None or financials.empty:
            return [{"free_cash_flow": 0}]
        
        result = {}
        for item in line_items:
            if item == "free_cash_flow":
                try:
                    operating_cash_flow = float(financials.loc["Total Cash From Operating Activities"].iloc[0])
                    capital_expenditure = abs(float(financials.loc["Capital Expenditures"].iloc[0]))
                    fcf = operating_cash_flow - capital_expenditure
                except (KeyError, IndexError):
                    fcf = stock.info.get("freeCashflow", 0)
                result["free_cash_flow"] = fcf
        
        return [result]
    except Exception as e:
        print(f"Warning: Error fetching line items from Yahoo Finance: {str(e)}")
        return [{"free_cash_flow": 0}]

def search_line_items_fd(
    ticker: str,
    line_items: List[str],
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial line items from Financial Datasets API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "period": period,
        "limit": limit
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    data = response.json()
    return data.get("search_results", [])

def search_line_items(
    ticker: str,
    line_items: List[str],
    period: str = 'ttm',
    limit: int = 1,
    data_source: str = "yahoo"
) -> List[Dict[str, Any]]:
    """Unified function to fetch financial line items."""
    try:
        if data_source == "yahoo":
            return search_line_items_yf(ticker, line_items, period, limit)
        return search_line_items_fd(ticker, line_items, period, limit)
    except Exception as e:
        if data_source == "financialdatasets":
            print(f"Financial Datasets API error: {str(e)}. Falling back to Yahoo Finance...")
            return search_line_items_yf(ticker, line_items, period, limit)
        return [{"free_cash_flow": 0}]

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    data_source: str = "yahoo"
) -> pd.DataFrame:
    """Get price data and convert to DataFrame."""
    prices = get_prices(ticker, start_date, end_date, data_source=data_source)
    return prices_to_df(prices)