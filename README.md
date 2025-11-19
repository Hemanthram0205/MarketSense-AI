🧠 MarketSense AI — Indian Stock Analysis & Prediction App

🔗 Live Demo: Add your Streamlit link here


📄 Overview

MarketSense AI is a powerful Streamlit web app designed for NSE/BSE stock analysis, real-time metrics, technical indicators, and AI-powered price prediction.
It integrates live market data using yFinance, processes it with Pandas, and presents insights using an elegant India-themed UI.

🎯 Features
🔍 Data & Stock Selection

Choose from popular NSE stocks or enter any .NS / .BO symbol manually

Smart suggestions and helpful symbol guide

📊 Market Data & Metrics

Real-time close price

Day high & low

52-week high & low

Average volume

Sector, industry, and company information

📈 Charts & Visualizations

Interactive price history chart

Daily volume bar chart

Clean trend visualization

📉 Technical Indicators

10-Day Moving Average (MA)

50-Day Moving Average (MA)

Automatic bullish / bearish signals

Volatility-based confidence scoring

🔮 AI-Powered Prediction

Trend-based forecast for 7–90 days

Predicted price with % change

Clear Buy / Hold / Sell recommendation

📥 Data Export

Download last 10-day trading data as CSV

⚡ Performance

10-minute smart caching for API optimization

Rate-limit handling

Smooth UI with responsive layout

⚙️ How It Works

User selects a stock (e.g., TCS.NS).

App fetches market data using yfinance.

Calculates key metrics and technical indicators.

Generates interactive charts.

Applies a simple linear regression trend model to predict future prices.

Displays prediction summary + investment recommendation.

User can download recent data in CSV format.

🧰 Tech Stack
Component	Technology Used
Language	Python
Framework	Streamlit
Market Data API	yFinance
Data Processing	Pandas, NumPy
Caching	st.cache_data
Charts	Streamlit Charts (Line, Bar)
Prediction Model	Linear Trend Model
Deployment	Streamlit Cloud / Local / Heroku
🚀 Why MarketSense AI?

India-focused stock analysis (NSE/BSE)

Clean, intuitive dashboard

Great for students, traders, and analysts

Demonstrates real-world analytics workflow:
Data → EDA → Indicators → Prediction → Export

Easy to scale with ML models like LSTM, Prophet, XGBoost
