# binance-oi-scanner-
A Python Telegram bot that scans Binance USDT futures for abnormal open interest growth.
- 📊 **Market-wide scan** of top USDT futures on Binance:
  - OI (contracts) growth ≥ 50% over the last 30 days
  - Price growth ≤ 50% over the same period
  - Average daily volume ≥ 5M USD
- 🔔 **Autoscan every hour (H1)**:
  - Sends only **new** signals (per symbol), no duplicates
- 🔍 **Raw (per-symbol) analysis**:
  - Send `ETH`, `BTC`, `SOL`, `DOTUSDT`, `BNB/USDT`, etc.
  - Bot returns raw metrics for price, volume and OI (no filters)
- 🔀 **Adjustable sorting modes**:
  - By OI growth in contracts
  - By OI growth in USD
  - By price growth
  - By average daily volume
