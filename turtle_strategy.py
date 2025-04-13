import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import base64
from io import BytesIO

st.set_page_config(layout="wide", page_title="Turtle Trading Strategy Backtester")

st.title("Turtle Trading Strategy Backtester")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")

    mode = st.radio("Select Mode", ["Normal Mode", "Optimization Mode"])

    # In both modes, ticker and duration are required.
    ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS").strip().upper()
    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1),
                               min_value=datetime(2010, 1, 1), max_value=datetime.now())
    end_date = st.date_input("End Date", value=datetime.now(),
                             min_value=datetime(2010, 1, 1), max_value=datetime.now())

    # For Normal Mode, you get to pick all parameters.
    if mode == "Normal Mode":
        st.subheader("Strategy Parameters")
        investment_amount = st.number_input("Investment Amount per Trade (₹)",
                                            min_value=10000, max_value=1000000,
                                            value=100000, step=10000)
        
        # Add columns for slider and input box side by side
        col1, col2 = st.columns([2, 1])
        with col1:
            lookback_period = st.slider("Lookback Period (Days)", min_value=20, max_value=100, value=55)
        with col2:
            lookback_period = st.number_input("", min_value=20, max_value=100, value=lookback_period, label_visibility="collapsed")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            profit_target = st.slider("Profit Target (Multiplier)", min_value=1.0, max_value=1.5, value=1.06, step=0.01,
                                    format="%.2f")
        with col2:
            profit_target = st.number_input("", min_value=1.0, max_value=1.5, value=profit_target, step=0.01, format="%.2f", label_visibility="collapsed")
        
        col1, col2 = st.columns([2, 1])  
        with col1:
            min_buy_gap = st.slider("Minimum Buy Gap (Days)", min_value=0, max_value=60, value=30)
        with col2:
            min_buy_gap = st.number_input("", min_value=0, max_value=60, value=min_buy_gap, step=1, label_visibility="collapsed")
            
    else:
        st.info(
            "In Optimization Mode, only ticker and duration are required. The app will search for optimal parameter values.")
        # Set fixed investment amount (can be adjusted) for optimization.
        investment_amount = 100000
        
        # Allow the user to define search ranges with both sliders and input boxes
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            lookback_range = st.slider("Lookback Period Range (Days)", 20, 100, (30, 70))
        with col2:
            lookback_min = st.number_input("Min", min_value=20, max_value=100, value=lookback_range[0])
        with col3:
            lookback_max = st.number_input("Max", min_value=20, max_value=100, value=lookback_range[1])
        # Update slider if input boxes change
        lookback_range = (lookback_min, lookback_max)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            profit_target_range = st.slider("Profit Target Range (Multiplier)", 1.0, 1.5, (1.03, 1.10), step=0.01)
        with col2:
            profit_min = st.number_input("Min", min_value=1.0, max_value=1.5, value=profit_target_range[0], step=0.01, format="%.2f")
        with col3:
            profit_max = st.number_input("Max", min_value=1.0, max_value=1.5, value=profit_target_range[1], step=0.01, format="%.2f")
        # Update slider if input boxes change
        profit_target_range = (profit_min, profit_max)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            min_buy_gap_range = st.slider("Min Buy Gap Range (Days)", 0, 60, (20, 40))
        with col2:
            gap_min = st.number_input("Min", min_value=0, max_value=60, value=min_buy_gap_range[0])
        with col3:
            gap_max = st.number_input("Max", min_value=0, max_value=60, value=min_buy_gap_range[1])
        # Update slider if input boxes change
        min_buy_gap_range = (gap_min, gap_max)

    run_analysis = st.button("Run Analysis", type="primary")


# Strategy simulation function with adjustable parameters
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

        # Check if we can buy: if open orders exist, enforce waiting until min_buy_gap days have passed.
        can_buy = True
        if open_orders:
            last_buy_date = open_orders[-1][0]
            if current_date < (last_buy_date + timedelta(days=min_buy_gap)):
                can_buy = False

        if can_buy and float(df['High'].iloc[i]) >= float(period_high):
            open_orders.append((current_date, float(current_close)))
            buy_signals += 1
            transaction_log.append({
                'Ticker': None,  # To be filled later
                'Date': current_date.strftime('%Y-%m-%d'),
                'Event': 'Buy',
                'Price': float(current_close),
                'Quantity': 1,
                'Post_Holdings': len(open_orders),
                'Remarks': 'Buy signal triggered'
            })

        # Track daily investment based on current open orders.
        investment_tracker.append({
            'date': current_date,
            'investment': len(open_orders) * investment_amount
        })

        # Sell condition: if open orders exist and current close >= profit_target * average buy price.
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
                    'Ticker': None,  # To be filled later
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
        'investment_over_time': pd.DataFrame(investment_tracker).set_index(
            'date') if investment_tracker else pd.DataFrame(),
        'transaction_log': pd.DataFrame(transaction_log) if transaction_log else pd.DataFrame(),
        'max_investment': max_investment
    }


# Create a Plotly chart for investment over time.
def create_investment_chart(ticker, investment_df, trades):
    if investment_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=investment_df.index,
        y=investment_df['investment'],
        mode='lines',
        name='Investment (₹)',
        line=dict(color='blue', width=2)
    ))
    # Add vertical lines at each sell date.
    for trade in trades:
        fig.add_vline(x=pd.to_datetime(trade['sell_date']), line_width=1, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"Investment Over Time: {ticker}",
        xaxis_title="Date",
        yaxis_title="Investment (₹)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# Generate Excel download link with all results.
def generate_excel_download_link(results):
    buffer = BytesIO()

    summary_df = pd.DataFrame.from_dict(
        {t: {
            'buy_signals': data['buy_signals'],
            'sell_signals': data['sell_signals'],
            'total_profit': data['total_profit'],
            'max_investment': data['max_investment']
        } for t, data in results.items()}, orient='index'
    ).reset_index().rename(columns={'index': 'Ticker'})

    all_trades = []
    for t, data in results.items():
        for trade in data['trades']:
            trade['Ticker'] = t
            all_trades.append(trade)
    trade_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    all_transactions = []
    for t, data in results.items():
        if not data['transaction_log'].empty:
            df_log = data['transaction_log'].copy()
            df_log['Ticker'] = t
            all_transactions.append(df_log)
    transaction_df = pd.concat(all_transactions, ignore_index=True) if all_transactions else pd.DataFrame()

    all_investments = []
    for t, data in results.items():
        if not data['investment_over_time'].empty:
            inv_df = data['investment_over_time'].reset_index()
            inv_df['Ticker'] = t
            all_investments.append(inv_df)
    investment_df = pd.concat(all_investments, ignore_index=True) if all_investments else pd.DataFrame()

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


# Main analysis logic
if run_analysis:
    results = {}
    if mode == "Optimization Mode":
        st.info("Running optimization. Please wait...")
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            st.error(f"No data available for {ticker}")
        else:
            best_obj = -np.inf
            best_params = None
            best_result = None
            # Define search ranges from user inputs.
            lookback_vals = range(lookback_range[0], lookback_range[1] + 1, 5)
            profit_vals = np.arange(profit_target_range[0], profit_target_range[1] + 0.001, 0.01)
            gap_vals = range(min_buy_gap_range[0], min_buy_gap_range[1] + 1, 5)

            total_runs = len(lookback_vals) * len(profit_vals) * len(gap_vals)
            run_count = 0
            progress_bar = st.progress(0)

            for lb in lookback_vals:
                for pt in profit_vals:
                    for gap in gap_vals:
                        res = simulate_turtle_strategy(data, investment_amount, lb, pt, gap)
                        run_count += 1
                        progress_bar.progress(run_count / total_runs)
                        # Objective: maximize profit while minimizing max investment.
                        if res['max_investment'] > 0 and res['sell_signals'] > 0:
                            obj = res['total_profit'] / (res['max_investment'] + 1)
                        else:
                            obj = -np.inf
                        if obj > best_obj:
                            best_obj = obj
                            best_params = {'lookback_period': lb, 'profit_target': pt, 'min_buy_gap': gap}
                            best_result = res
            st.success("Optimization Completed!")
            st.write("### Best Parameters Found")
            st.write(best_params)
            st.write("### Performance")
            st.write({
                "Buy Signals": best_result['buy_signals'],
                "Sell Signals": best_result['sell_signals'],
                "Total Profit (₹)": round(best_result['total_profit'], 2),
                "Max Investment (₹)": best_result['max_investment'],
                "Objective": round(best_obj, 2)
            })
            results[ticker] = best_result
            fig = create_investment_chart(ticker, best_result['investment_over_time'], best_result['trades'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(generate_excel_download_link({ticker: best_result}), unsafe_allow_html=True)
    else:
        # Normal Mode: use user-supplied parameters.
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            st.error(f"No data available for {ticker}")
        else:
            result = simulate_turtle_strategy(data, investment_amount, lookback_period, profit_target, min_buy_gap)
            result['transaction_log']['Ticker'] = ticker
            results[ticker] = result
            st.success("Analysis Completed!")
            summary_df = pd.DataFrame.from_dict({
                ticker: {
                    'Buy Signals': result['buy_signals'],
                    'Sell Signals': result['sell_signals'],
                    'Total Profit (₹)': round(result['total_profit'], 2),
                    'Max Investment (₹)': result['max_investment'],
                    'Return (%)': round((result['total_profit'] / result['max_investment']) * 100, 2) if result[
                                                                                                             'max_investment'] > 0 else 0
                }
            }, orient='index').reset_index().rename(columns={'index': 'Ticker'})

            st.header("Performance Summary")
            st.dataframe(summary_df, use_container_width=True)

            tabs = st.tabs(["Charts", "Trade Details", "Transactions", "Daily Investment"])
            with tabs[0]:
                fig = create_investment_chart(ticker, result['investment_over_time'], result['trades'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with tabs[1]:
                st.header("Aggregated Trade Details")
                if result['trades']:
                    st.dataframe(pd.DataFrame(result['trades']), use_container_width=True)
                else:
                    st.info("No trades executed.")
            with tabs[2]:
                st.header("Transaction Log")
                if not result['transaction_log'].empty:
                    st.dataframe(result['transaction_log'], use_container_width=True)
            with tabs[3]:
                st.header("Daily Investment")
                if not result['investment_over_time'].empty:
                    st.dataframe(result['investment_over_time'], use_container_width=True)

            st.markdown(generate_excel_download_link(results), unsafe_allow_html=True)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Analysis' to begin.")

with st.expander("About the Turtle Trading Strategy"):
    st.markdown("""
    ## Strategy Explanation

    The Turtle Trading Strategy is a breakout system originally taught by Richard Dennis and William Eckhardt.

    **Key Components:**
    - **Lookback Period:** The number of days used to determine the highest high.
    - **Profit Target:** The multiplier (e.g., 1.06 means a 6% gain) at which all positions are sold.
    - **Minimum Buy Gap:** Once a buy occurs, no new buys are triggered until this many days have passed.
    - **Position Sizing:** A fixed investment amount per trade.

    In **Optimization Mode**, the app searches for the combination of lookback period, profit target, and minimum buy gap that maximizes the ratio of total profit to max investment.
    """)
