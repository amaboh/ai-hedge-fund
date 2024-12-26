from langchain_core.messages import HumanMessage
from agents.state import AgentState, show_agent_reasoning
import json

def generate_text_summary(metrics, signals, reasoning_data):
    """Generates a natural language summary of the fundamental analysis results.
    
    Args:
        metrics: Dictionary containing financial metrics
        signals: List of signal indicators (bullish/bearish/neutral)
        reasoning_data: Dictionary containing reasoning and signal information
    
    Returns:
        str: A human-readable summary of the fundamental analysis
    """
    summary_parts = []
    
    # Profitability summary explains the company's ability to generate returns
    profitability_text = (
        f"Profitability analysis shows {reasoning_data['reasoning']['Profitability']['signal']} indicators with "
        f"Return on Equity at {metrics['return_on_equity']:.1%}, "
        f"Net Margin at {metrics['net_margin']:.1%}, and "
        f"Operating Margin at {metrics['operating_margin']:.1%}."
    )
    summary_parts.append(profitability_text)
    
    # Growth summary indicates the company's expansion trajectory
    growth_text = (
        f"Growth metrics indicate {reasoning_data['reasoning']['Growth']['signal']} performance with "
        f"Revenue Growth at {metrics['revenue_growth']:.1%} and "
        f"Earnings Growth at {metrics['earnings_growth']:.1%}."
    )
    summary_parts.append(growth_text)
    
    # Financial health summary shows the company's stability and solvency
    health_text = (
        f"Financial health is {reasoning_data['reasoning']['Financial_Health']['signal']} with "
        f"Current Ratio at {metrics['current_ratio']:.2f} and "
        f"Debt-to-Equity at {metrics['debt_to_equity']:.2f}."
    )
    summary_parts.append(health_text)
    
    # Valuation summary indicates if the stock is fairly priced
    valuation_text = (
        f"Valuation metrics are {reasoning_data['reasoning']['Price_Ratios']['signal']} with "
        f"P/E at {metrics['price_to_earnings_ratio']:.2f}, "
        f"P/B at {metrics['price_to_book_ratio']:.2f}, and "
        f"P/S at {metrics['price_to_sales_ratio']:.2f}."
    )
    summary_parts.append(valuation_text)
    
    # Overall summary provides a comprehensive view of all indicators
    bullish_count = signals.count('bullish')
    bearish_count = signals.count('bearish')
    neutral_count = signals.count('neutral')
    
    overall_text = (
        f"\nOverall Analysis: Found {bullish_count} bullish, {bearish_count} bearish, "
        f"and {neutral_count} neutral indicators. "
        f"The analysis suggests a {reasoning_data['signal']} stance with "
        f"{reasoning_data['confidence']} confidence based on the comprehensive evaluation "
        f"of profitability, growth, financial health, and valuation metrics."
    )
    summary_parts.append(overall_text)
    
    return "\n".join(summary_parts)

def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """Calculate intrinsic value using Discounted Cash Flow (DCF) method.
    
    Args:
        free_cash_flow: Current free cash flow
        growth_rate: Expected annual growth rate
        discount_rate: Required rate of return
        terminal_growth_rate: Long-term growth rate
        num_years: Projection period in years
    
    Returns:
        float: Calculated intrinsic value
    """
    if not free_cash_flow:
        return 0
        
    # Project future cash flows using compound growth
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]
    
    # Calculate present value of each projected cash flow
    present_values = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows)]
    
    # Calculate terminal value using perpetuity growth model
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years
    
    # Sum all present values and terminal value
    return sum(present_values) + terminal_present_value

def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals.
    
    This agent evaluates a company's financial health through multiple lenses:
    - Profitability
    - Growth
    - Financial Health
    - Valuation
    - Intrinsic Value
    
    Args:
        state: Agent state containing market data and metadata
    
    Returns:
        dict: Updated state with analysis results
    """
    print(f"State metadata: {state.get('metadata', {})}")
    
    show_reasoning = state["metadata"]["show_reasoning"]
    print(f"Show reasoning flag: {show_reasoning}")
    
    data = state["data"]
    metrics = data["financial_metrics"][0]
    financial_line_item = data["financial_line_items"][0]
    market_cap = data["market_cap"]

    signals = []
    reasoning = {}
    
    # 1. Profitability Analysis
    profitability_score = 0
    if metrics["return_on_equity"] > 0.15:  # Strong ROE above 15%
        profitability_score += 1
    if metrics["net_margin"] > 0.20:  # Healthy profit margins
        profitability_score += 1
    if metrics["operating_margin"] > 0.15:  # Strong operating efficiency
        profitability_score += 1
        
    signals.append('bullish' if profitability_score >= 2 else 'bearish' if profitability_score == 0 else 'neutral')
    reasoning["Profitability"] = {
        "signal": signals[0],
        "details": f"ROE: {metrics['return_on_equity']:.2%}, Net Margin: {metrics['net_margin']:.2%}, Op Margin: {metrics['operating_margin']:.2%}"
    }
    
    # 2. Growth Analysis
    growth_score = 0
    if metrics["revenue_growth"] > 0.10:  # 10% revenue growth
        growth_score += 1
    if metrics["earnings_growth"] > 0.10:  # 10% earnings growth
        growth_score += 1
    if metrics.get("book_value_growth", 0) > 0.10:  # 10% book value growth
        growth_score += 1
        
    signals.append('bullish' if growth_score >= 2 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning["Growth"] = {
        "signal": signals[1],
        "details": f"Revenue Growth: {metrics['revenue_growth']:.2%}, Earnings Growth: {metrics['earnings_growth']:.2%}"
    }
    
    # 3. Financial Health
    health_score = 0
    if metrics["current_ratio"] > 1.5:  # Strong liquidity
        health_score += 1
    if metrics["debt_to_equity"] < 0.5:  # Conservative debt levels
        health_score += 1
    if metrics["free_cash_flow_per_share"] > metrics["earnings_per_share"] * 0.8:  # Strong FCF conversion
        health_score += 1
        
    signals.append('bullish' if health_score >= 2 else 'bearish' if health_score == 0 else 'neutral')
    reasoning["Financial_Health"] = {
        "signal": signals[2],
        "details": f"Current Ratio: {metrics['current_ratio']:.2f}, D/E: {metrics['debt_to_equity']:.2f}"
    }
    
    # 4. Price to X ratios
    pe_ratio = metrics["price_to_earnings_ratio"]
    pb_ratio = metrics["price_to_book_ratio"]
    ps_ratio = metrics["price_to_sales_ratio"]
    
    price_ratio_score = 0
    if pe_ratio < 25:  # Reasonable P/E ratio
        price_ratio_score += 1
    if pb_ratio < 3:  # Reasonable P/B ratio
        price_ratio_score += 1
    if ps_ratio < 5:  # Reasonable P/S ratio
        price_ratio_score += 1
        
    signals.append('bullish' if price_ratio_score >= 2 else 'bearish' if price_ratio_score == 0 else 'neutral')
    reasoning["Price_Ratios"] = {
        "signal": signals[3],
        "details": f"P/E: {pe_ratio:.2f}, P/B: {pb_ratio:.2f}, P/S: {ps_ratio:.2f}"
    }

    # 5. Intrinsic Value
    free_cash_flow = financial_line_item.get('free_cash_flow', 0)
    growth_rate = metrics["earnings_growth"]
    try:
        intrinsic_value = calculate_intrinsic_value(
            free_cash_flow=free_cash_flow,
            growth_rate=growth_rate,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )
    except Exception as e:
        print(f"\nDebug: Error calculating intrinsic value: {str(e)}")
        intrinsic_value = 0
        
    signals.append('bullish' if market_cap < intrinsic_value else 'bearish')
    reasoning["Intrinsic_Value"] = {
        "signal": signals[4],
        "details": f"Intrinsic Value: ${intrinsic_value:,.2f}, Market Cap: ${market_cap:,.2f}"
    }
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }
    
    print(json.dumps(message_content, indent=2))

    # Generate the text summary before creating the message
    text_summary = generate_text_summary(metrics, signals, message_content)
    
    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamentals_agent",
    )
    
    if show_reasoning:
        print("\n==========  Fundamental Analysis Summary  ==========")
        print(text_summary)
        print("=" * 50)
        
        print("\n==========  Fundamental Analysis Details  ==========")
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")
    
    return {
        "messages": [message],
        "data": data,
    }