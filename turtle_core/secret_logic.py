import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import base64


def simulate_turtle_strategy(df, investment_amount, lookback_period, profit_target, min_buy_gap):
    open_orders = []  # Each order: (buy_date, buy_price)
    total_profit = 0
    buy_signals = 0
    sell_signals = 0
    trade_details = []  # Aggregated trade record per sell event
    investment_tracker = []  # Daily tracking of total investment
    transaction_log = []  # Detailed log of every individual transaction

    df = df.sort_index()
    for i in range(lookback_period, len(df)):
        current_date = df.index[i]
        period_high = df['High'].iloc[i - lookback_period:i].max()
        current_close = df['Close'].iloc[i]

        # Check if we can buy: enforce waiting period
        can_buy = True
        if open_orders:
            last_buy_date = open_orders[-1][0]
            if current_date < (last_buy_date + timedelta(days=min_buy_gap)):
                can_buy = False

        if can_buy and float(df['High'].iloc[i]) >= float(period_high):
            open_orders.append((current_date, float(current_close)))
            buy_signals += 1
            transaction_log.append({
                'Ticker': None,
                'Date': current_date.strftime('%Y-%m-%d'),
                'Event': 'Buy',
                'Price': float(current_close),
                'Quantity': 1,
                'Post_Holdings': len(open_orders),
                'Remarks': 'Buy signal triggered'
            })

        # Track daily investment based on open orders
        investment_tracker.append({
            'date': current_date,
            'investment': len(open_orders) * investment_amount
        })

        # Sell when current_close >= profit_target * avg buy price
        if open_orders:
            avg_buy = sum([order[1] for order in open_orders]) / len(open_orders)
            if float(current_close) >= profit_target * avg_buy:
                profit_per_order = (float(current_close) / avg_buy - 1) * investment_amount
                trade_profit = profit_per_order * len(open_orders)
                total_profit += trade_profit
                sell_signals += len(open_orders)

                trade_record = {
                    'buy_dates': ', '.join([order[0].strftime('%Y-%m-%d') for order in open_orders]),
                    'buy_prices': ', '.join([str(order[1]) for order in open_orders]),
                    'sell_date': current_date.strftime('%Y-%m-%d'),
                    'sell_price': float(current_close),
                    'avg_buy_price': avg_buy,
                    'orders_closed': len(open_orders),
                    'profit': trade_profit
                }
                trade_details.append(trade_record)

                transaction_log.append({
                    'Ticker': None,
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Event': 'Sell',
                    'Price': float(current_close),
                    'Quantity': len(open_orders),
                    'Post_Holdings': 0,
                    'Remarks': f"Sell signal triggered; Avg Buy = {avg_buy:.2f}"
                })

                open_orders = []

    max_investment = max([entry['investment'] for entry in investment_tracker]) if investment_tracker else 0

    return {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'total_profit': total_profit,
        'trades': trade_details,
        'investment_over_time': pd.DataFrame(investment_tracker).set_index('date') if investment_tracker else pd.DataFrame(),
        'transaction_log': pd.DataFrame(transaction_log) if transaction_log else pd.DataFrame(),
        'max_investment': max_investment
    }


def create_investment_chart(ticker, investment_df, trades):
    """
    Returns a Plotly figure of investment over time with vertical lines on sell dates.
    """
    if investment_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=investment_df.index,
        y=investment_df['investment'],
        mode='lines',
        name='Investment (₹)'
    ))
    for trade in trades:
        fig.add_vline(
            x=pd.to_datetime(trade['sell_date']),
            line_width=1,
            line_dash='dash'
        )
    fig.update_layout(
        title=f"Investment Over Time: {ticker}",
        xaxis_title="Date",
        yaxis_title="Investment (₹)",
        height=500
    )
    return fig


def generate_excel_download_link(results):
    """
    Given a dict of results, builds an in-memory Excel file and returns a data URI link.
    """
    buffer = BytesIO()

    # Summary sheet
    summary_df = pd.DataFrame.from_dict(
        {t: {
            'buy_signals': data['buy_signals'],
            'sell_signals': data['sell_signals'],
            'total_profit': data['total_profit'],
            'max_investment': data['max_investment']
        } for t, data in results.items()}, orient='index'
    ).reset_index().rename(columns={'index': 'Ticker'})

    # Trade details
    all_trades = []
    for t, data in results.items():
        for trade in data['trades']:
            trade['Ticker'] = t
            all_trades.append(trade)
    trade_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Transaction log
    all_transactions = []
    for t, data in results.items():
        if not data['transaction_log'].empty:
            df_log = data['transaction_log'].copy()
            df_log['Ticker'] = t
            all_transactions.append(df_log)
    transaction_df = pd.concat(all_transactions, ignore_index=True) if all_transactions else pd.DataFrame()

    # Daily investment
    all_investments = []
    for t, data in results.items():
        if not data['investment_over_time'].empty:
            inv_df = data['investment_over_time'].reset_index()
            inv_df['Ticker'] = t
            all_investments.append(inv_df)
    investment_df = pd.concat(all_investments, ignore_index=True) if all_investments else pd.DataFrame()

    # Write to Excel
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        if not trade_df.empty:
            trade_df.to_excel(writer, sheet_name='TradeDetails', index=False)
        if not transaction_df.empty:
            transaction_df.to_excel(writer, sheet_name='AllTransactions', index=False)
        if not investment_df.empty:
            investment_df.to_excel(writer, sheet_name='DailyInvestment', index=False)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="turtle_strategy_results.xlsx">Download Excel Report</a>'
