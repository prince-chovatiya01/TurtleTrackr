# 🐢 Turtle Trading Strategy Backtester 📈

An interactive Streamlit app to backtest and optimize the legendary **Turtle Trading Strategy** using historical data from Yahoo Finance.

---

## 📖 Turtle Trading Strategy

The Turtle Trading Strategy is a **trend-following breakout system** taught by **Richard Dennis** and **William Eckhardt**.

### Strategy Logic

- 📈 **Buy Signal**: When today's price breaks above the highest high of the last `N` days
- 💰 **Sell Signal**: When price reaches a specified multiple (`profit_target`) of the average buy price
- ⏳ **Cool-down Period**: Wait `min_buy_gap` days between buy signals
- 🧮 **Position Sizing**: Fixed ₹ amount per trade

---

## 🧠 Optimization Mode

In **Optimization Mode**, the app finds the best combination of:

- 🔁 **Lookback Period**: e.g., 20–70 days
- 🎯 **Profit Target**: e.g., 1.03–1.10
- 📆 **Minimum Buy Gap**: e.g., 10–40 days

### Objective Function

Objective = Total Profit / (Max Investment + 1)


This helps maximize returns while controlling capital usage.

---

## 📊 Output Includes

- 📈 **Investment-over-time Plot** (with red markers at sell dates)
- 📘 **Aggregated Trade Details** – Buy/sell events grouped
- 📄 **Transaction Log** – Per-transaction breakdown
- 💰 **Daily Capital Tracker** – Track how much is invested over time
- 📥 **Excel Export** – Download `.xlsx` with all outputs

---

## 💼 Example Tickers

| Market     | Example Tickers                      |
|------------|--------------------------------------|
| NSE India  | `RELIANCE.NS`, `TCS.NS`, `INFY.NS`   |
| US Stocks  | `AAPL`, `MSFT`, `GOOGL`, `TSLA`      |

> 🔍 Data is fetched live from Yahoo Finance using the `yfinance` library.

---

## 🛠 Built With

- [**Streamlit**](https://streamlit.io/) – UI/Frontend
- [**yfinance**](https://github.com/ranaroussi/yfinance) – Stock data fetching
- [**pandas**](https://pandas.pydata.org/) – Data processing
- [**plotly**](https://plotly.com/) – Interactive plotting
- [**xlsxwriter**](https://xlsxwriter.readthedocs.io/) – Excel generation

---

## ✅ To-Do / Future Improvements

- [ ] Add stop-loss logic
- [ ] Include transaction fees
- [ ] Batch backtesting for multiple tickers
- [ ] Real-time email alerts or notifications
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces

---

## 📜 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

Inspired by the original **Turtle Traders** experiment. Strategy logic has been adapted and simplified for educational and retail use.

---
