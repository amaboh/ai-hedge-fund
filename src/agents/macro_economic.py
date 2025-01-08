from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
from tools.macro_tools import (
    get_market_indices,
    get_bond_yields,
    get_sector_performance,
    get_commodity_prices,
    get_currency_strength,
    get_economic_indicators
)
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union

def safe_float(value: Any) -> float:
    """Safely convert any value to float."""
    try:
        if hasattr(value, 'iloc'):  # If it's a pandas Series
            return float(value.iloc[0])
        return float(value)
    except (TypeError, ValueError, IndexError):
        return 0.0

def generate_macro_summary(analysis_data: Dict[str, Any]) -> str:
    """Generate human-readable summary of macroeconomic analysis."""
    summary_parts = []
    
    # Market Environment
    market_text = (
        f"Market Environment:\n"
        f"Overall Market Stance: {analysis_data['signal'].title()} "
        f"(Confidence: {analysis_data['confidence']})\n"
    )
    summary_parts.append(market_text)
    
    # Leading Indicators
    leading_summary = (
        f"Leading Indicators:\n"
        f"- Market Performance: {analysis_data['reasoning']['market_performance']['signal']}\n"
        f"- Yield Curve: {analysis_data['reasoning']['yield_curve']['signal']}\n"
        f"- Sector Rotation: {analysis_data['reasoning']['sector_rotation']['signal']}\n"
        f"- Manufacturing PMI: {analysis_data['reasoning']['leading_economic']['signal']}"
    )
    summary_parts.append(leading_summary)
    
    # Lagging Indicators
    lagging_summary = (
        f"Lagging Indicators:\n"
        f"- GDP Growth: {analysis_data['reasoning']['gdp']['signal']}\n"
        f"- Inflation: {analysis_data['reasoning']['inflation']['signal']}\n"
        f"- Employment: {analysis_data['reasoning']['employment']['signal']}\n"
        f"- Currency Strength: {analysis_data['reasoning']['currency']['signal']}"
    )
    summary_parts.append(lagging_summary)
    
    # Risk Factors
    risk_summary = (
        f"Risk Assessment:\n"
        f"- Economic Risk Level: {analysis_data['risk_level']}\n"
        f"- Key Concerns: {', '.join(analysis_data['key_risks'])}"
    )
    summary_parts.append(risk_summary)
    
    # Investment Implications
    implications = (
        f"Investment Implications:\n"
        f"{analysis_data['investment_implications']}"
    )
    summary_parts.append(implications)
    
    return "\n\n".join(summary_parts)

def generate_investment_implications(
    overall_signal: str,
    risk_level: str,
    risk_factors: List[str],
    sector_data: Dict[str, Union[float, Any]]
) -> str:
    """Generate investment implications based on macro analysis."""
    implications = []
    
    # Market stance implications
    if overall_signal == 'bullish':
        implications.append(
            "Market conditions support risk-taking, but maintain disciplined position sizing."
        )
    elif overall_signal == 'bearish':
        implications.append(
            "Defensive positioning recommended with focus on quality and value."
        )
    else:
        implications.append(
            "Mixed signals suggest balanced positioning with selective opportunities."
        )
    
    # Risk-based recommendations
    if risk_level == "High":
        implications.append(
            "High risk environment warrants reduced position sizes and increased hedging."
        )
    elif risk_level == "Low":
        implications.append(
            "Favorable risk environment supports strategic growth positioning."
        )
    
    # Convert each sector performance to float and sort
    sector_list = [(sector, safe_float(perf)) for sector, perf in sector_data.items()]
    top_sectors = sorted(sector_list, key=lambda x: x[1], reverse=True)[:3]
    
    implications.append(
        f"Strongest sectors: {', '.join(sector[0] for sector in top_sectors)}"
    )
    
    return " ".join(implications)

def macro_economic_agent(state: AgentState):
    """Analyzes macroeconomic conditions and their investment implications."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    
    # Get dates for analysis
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    start_date = data["start_date"] or (
        datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=90)
    ).strftime('%Y-%m-%d')
    
    # Collect macroeconomic data
    market_indices = get_market_indices(start_date, end_date)
    bond_yields = get_bond_yields()
    sector_perf = get_sector_performance(start_date, end_date)
    commodities = get_commodity_prices()
    currencies = get_currency_strength()
    economic = get_economic_indicators()
    
    # Initialize signals and reasoning
    signals = []
    reasoning = {}
    
    # 1. Market Performance Analysis
    market_score = sum(1 for perf in market_indices.values() if safe_float(perf) > 0)
    market_signal = 'bullish' if market_score >= 2 else 'bearish'
    signals.append(market_signal)
    reasoning["market_performance"] = {
        "signal": market_signal,
        "details": {
            "indices": {k: safe_float(v) for k, v in market_indices.items()},
            "score": market_score
        }
    }
    
    # 2. Yield Curve Analysis
    try:
        yield_spread = safe_float(bond_yields["10-Year Treasury"]) - safe_float(bond_yields["13-Week Treasury"])
        yield_signal = 'bearish' if yield_spread < 0 else 'neutral' if yield_spread < 0.5 else 'bullish'
    except (KeyError, TypeError):
        yield_signal = 'neutral'
        yield_spread = 0
    
    signals.append(yield_signal)
    reasoning["yield_curve"] = {
        "signal": yield_signal,
        "details": {
            "yields": {k: safe_float(v) for k, v in bond_yields.items()},
            "spread": yield_spread
        }
    }
    
    # 3. Sector Rotation Analysis
    defensive_sectors = ["Consumer Staples", "Utilities", "Healthcare"]
    cyclical_sectors = ["Technology", "Consumer Discretionary", "Materials"]
    
    defensive_perf = sum(safe_float(sector_perf.get(sector, 0)) 
                        for sector in defensive_sectors) / len(defensive_sectors)
    cyclical_perf = sum(safe_float(sector_perf.get(sector, 0)) 
                       for sector in cyclical_sectors) / len(cyclical_sectors)
    
    sector_signal = 'bullish' if cyclical_perf > defensive_perf else 'bearish'
    signals.append(sector_signal)
    reasoning["sector_rotation"] = {
        "signal": sector_signal,
        "details": {
            "sectors": {k: safe_float(v) for k, v in sector_perf.items()},
            "defensive_avg": defensive_perf,
            "cyclical_avg": cyclical_perf
        }
    }
    
    # 4. Economic Indicators Analysis
    # Leading Indicators
    leading_score = 0
    if safe_float(economic["manufacturing_pmi"]) > 50:
        leading_score += 1
    if safe_float(economic["consumer_confidence"]) > 100:
        leading_score += 1
    if safe_float(economic["retail_sales_growth"]) > 0:
        leading_score += 1
        
    leading_signal = 'bullish' if leading_score >= 2 else 'bearish'
    signals.append(leading_signal)
    reasoning["leading_economic"] = {
        "signal": leading_signal,
        "details": {
            "pmi": safe_float(economic["manufacturing_pmi"]),
            "confidence": safe_float(economic["consumer_confidence"]),
            "retail_sales": safe_float(economic["retail_sales_growth"])
        }
    }
    
    # GDP Analysis
    gdp_signal = 'bullish' if safe_float(economic["gdp_growth"]) > 2.0 else 'bearish'
    signals.append(gdp_signal)
    reasoning["gdp"] = {
        "signal": gdp_signal,
        "details": {"growth": safe_float(economic["gdp_growth"])}
    }
    
    # Inflation Analysis
    inflation_rate = safe_float(economic["inflation_rate"])
    inflation_signal = 'bearish' if inflation_rate > 4.0 else 'neutral' if inflation_rate > 2.0 else 'bullish'
    signals.append(inflation_signal)
    reasoning["inflation"] = {
        "signal": inflation_signal,
        "details": {"rate": inflation_rate}
    }
    
    # Employment Analysis
    unemployment_rate = safe_float(economic["unemployment_rate"])
    employment_signal = 'bullish' if unemployment_rate < 4.0 else 'bearish'
    signals.append(employment_signal)
    reasoning["employment"] = {
        "signal": employment_signal,
        "details": {"rate": unemployment_rate}
    }
    
    # Currency Analysis
    usd_strength = sum(1 for curr in currencies.values() if safe_float(curr["change"]) > 0)
    currency_signal = 'bullish' if usd_strength >= 2 else 'bearish'
    signals.append(currency_signal)
    reasoning["currency"] = {
        "signal": currency_signal,
        "details": currencies
    }
    
    # Determine overall signal and confidence
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    total_signals = len(signals)
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    # Determine risk level and key risks
    risk_factors = []
    if yield_spread < 0:
        risk_factors.append("Inverted Yield Curve")
    if inflation_rate > 4.0:
        risk_factors.append("High Inflation")
    if unemployment_rate > 5.0:
        risk_factors.append("High Unemployment")
    if sum(1 for perf in market_indices.values() if safe_float(perf) < 0) >= len(market_indices):
        risk_factors.append("Broad Market Weakness")
    
    risk_level = "High" if len(risk_factors) >= 2 else "Moderate" if len(risk_factors) == 1 else "Low"
    
    # Generate investment implications
    investment_implications = generate_investment_implications(
        overall_signal,
        risk_level,
        risk_factors,
        sector_perf
    )
    
    # Prepare message content
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning,
        "risk_level": risk_level,
        "key_risks": risk_factors,
        "investment_implications": investment_implications
    }
    
    # Create the message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="macro_economic_agent",
    )
    
    # Show reasoning if requested
    if show_reasoning:
        print("\n==========  Macroeconomic Analysis Summary  ==========")
        text_summary = generate_macro_summary(message_content)
        print(text_summary)
        print("=" * 50)
        
        print("\n==========  Macroeconomic Analysis Details  ==========")
        show_agent_reasoning(message_content, "Macroeconomic Analysis Agent")
    
    return {
        "messages": [message],
        "data": data,
    }