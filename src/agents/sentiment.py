from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning

import json

##### Sentiment Agent #####
def generate_sentiment_summary(signal_data, insider_trades):
    """Generate human-readable summary of sentiment analysis."""
    summary_parts = []
    
    if insider_trades:
        # Analyze insider trading patterns
        total_trades = len(insider_trades)
        buy_trades = sum(1 for trade in insider_trades if trade.get("transaction_shares", 0) > 0)
        sell_trades = sum(1 for trade in insider_trades if trade.get("transaction_shares", 0) < 0)
        
        insider_summary = (
            f"Insider Trading Analysis:\n"
            f"- Total Trades: {total_trades}\n"
            f"- Buy Trades: {buy_trades}\n"
            f"- Sell Trades: {sell_trades}\n"
            f"- Net Position: {signal_data['signal'].title()}"
        )
        summary_parts.append(insider_summary)
        
        # Add trade details
        trade_details = []
        for trade in insider_trades:
            shares = trade.get("transaction_shares", 0)
            date = trade.get("transaction_date", "N/A")
            insider = trade.get("insider_name", "Unknown Insider")
            trade_details.append(
                f"- {date}: {insider} {'bought' if shares > 0 else 'sold'} "
                f"{abs(shares):,.0f} shares"
            )
        
        if trade_details:
            summary_parts.append("Recent Trades:\n" + "\n".join(trade_details))
    else:
        summary_parts.append(
            "No recent insider trades detected. This could indicate either trading "
            "restrictions or a lack of strong insider sentiment in either direction."
        )
    
    # Overall Analysis
    overall_text = (
        f"\nOverall Sentiment: {signal_data['signal'].title()}\n"
        f"Confidence: {signal_data['confidence']}\n"
        f"{signal_data['reasoning']}"
    )
    summary_parts.append(overall_text)
    
    return "\n\n".join(summary_parts)

def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals."""
    data = state["data"]
    insider_trades = data["insider_trades"]
    show_reasoning = state["metadata"]["show_reasoning"]

    # Initialize signals list
    signals = []
    
    # Process insider trades if available
    if insider_trades:
        for trade in insider_trades:
            transaction_shares = trade.get("transaction_shares")
            if transaction_shares:  # Only process if transaction_shares exists and is not None
                if transaction_shares < 0:
                    signals.append("bearish")
                else:
                    signals.append("bullish")

    # Determine overall signal and confidence
    if signals:
        # If we have signals, calculate based on the signals we collected
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")
        total_signals = len(signals)
        
        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"
            
        confidence = max(bullish_signals, bearish_signals) / total_signals if total_signals > 0 else 0.5
        
        reasoning = f"Analyzed {total_signals} insider trades. Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}"
    else:
        # Default values when no insider trades are available
        overall_signal = "neutral"
        confidence = 0.5
        reasoning = "No recent insider trades available. Defaulting to neutral stance."

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }


    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_agent",
    )

    if show_reasoning:
        print("\n==========  Sentiment Analysis Summary  ==========")
        text_summary = generate_sentiment_summary(message_content, insider_trades)
        print(text_summary)
        print("=" * 50)
        
        print("\n==========  Sentiment Analysis Details  ==========")
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")
    
    return {
        "messages": [message],
        "data": data,
    }