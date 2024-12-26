from langchain_openai.chat_models import ChatOpenAI

from agents.state import AgentState
from tools.api import (
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    search_line_items,
)

from datetime import datetime

llm = ChatOpenAI(model="gpt-4o")

def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    messages = state["messages"]
    data = state["data"]
    
    # Get data source from metadata or default to "yahoo"
    data_source = data.get("data_source", "yahoo")

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    if not data["start_date"]:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date_obj.replace(month=end_date_obj.month - 3) if end_date_obj.month > 3 else \
            end_date_obj.replace(year=end_date_obj.year - 1, month=end_date_obj.month + 9)
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    try:
        # Get market data from the specified source with fallback to Yahoo Finance
        prices = get_prices(
            ticker=data["ticker"], 
            start_date=start_date, 
            end_date=end_date,
            data_source=data_source
        )

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=data["ticker"], 
            report_period=end_date, 
            period='ttm', 
            limit=1,
            data_source=data_source
        )

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=data["ticker"], 
            end_date=end_date,
            limit=5,
            data_source=data_source
        )

        # Get the market cap
        market_cap = get_market_cap(
            ticker=data["ticker"],
            data_source=data_source
        )

        # Get the line_items
        financial_line_items = search_line_items(
            ticker=data["ticker"], 
            line_items=["free_cash_flow"],
            period='ttm',
            limit=1,
            data_source=data_source
        )

        # Update the state with all the gathered data
        updated_data = {
            **data,
            "prices": prices,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "insider_trades": insider_trades,
            "market_cap": market_cap,
            "financial_line_items": financial_line_items,
        }

        # Log data source being used (helpful for debugging)
        print(f"Market data retrieved successfully from {data_source}")

        return {
            "messages": messages,
            "data": updated_data
        }

    except Exception as e:
        error_msg = f"Error in market_data_agent: {str(e)}"
        print(error_msg)
        
        # If using Financial Datasets API, try falling back to Yahoo Finance
        if data_source == "financialdatasets":
            print("Attempting to fall back to Yahoo Finance...")
            data["data_source"] = "yahoo"
            return market_data_agent(state)
        else:
            # If already using Yahoo Finance or other error, raise the exception
            raise Exception(error_msg)