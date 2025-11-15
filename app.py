# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indian Stock Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Indian theme
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #FF9933;
        padding-bottom: 1rem;
    }
    h2 {
        color: #138808;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stock-header {
        background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Indian Stock Symbols Dictionary
POPULAR_STOCKS = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HUL": "HINDUNILVR.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "M&M": "M&M.NS",
    "Titan": "TITAN.NS",
    "Coal India": "COALINDIA.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Tech Mahindra": "TECHM.NS",
    "UltraTech": "ULTRACEMCO.NS",
    "Nestle India": "NESTLEIND.NS",
    "HCL Tech": "HCLTECH.NS",
    "JSW Steel": "JSWSTEEL.NS"
}

# Title with Indian flag colors
st.markdown("""
    <div class='stock-header' style='text-align: center;'>
        <h1 style='color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            🇮🇳 Indian Stock Market Prediction App
        </h1>
        <p style='color: white; margin-top: 0.5rem; font-size: 1.2rem;'>NSE/BSE Stock Analysis & Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock selection method
    selection_method = st.radio(
        "Select Stock By:",
        ["Popular Stocks", "Enter Symbol Manually"],
        help="Choose from popular Indian stocks or enter a symbol manually"
    )
    
    if selection_method == "Popular Stocks":
        stock_name = st.selectbox(
            "Select Stock",
            options=list(POPULAR_STOCKS.keys()),
            help="Choose from popular NSE stocks"
        )
        stock_symbol = POPULAR_STOCKS[stock_name]
        display_name = stock_name
    else:
        manual_symbol = st.text_input(
            "Enter Stock Symbol",
            value="RELIANCE.NS",
            help="Enter NSE symbol with .NS (e.g., TCS.NS) or BSE symbol with .BO (e.g., TCS.BO)"
        ).upper()
        stock_symbol = manual_symbol
        display_name = manual_symbol.replace('.NS', '').replace('.BO', '')
    
    st.info("💡 **NSE stocks**: Add .NS (e.g., TCS.NS)\n\n**BSE stocks**: Add .BO (e.g., TCS.BO)")
    
    # Date range
    st.subheader("📅 Historical Data Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Prediction period
    st.subheader("🔮 Prediction Settings")
    prediction_days = st.slider(
        "Days to Predict",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of days to predict into the future"
    )
    
    # Model selection
    st.subheader("🤖 Prediction Model")
    model_type = st.selectbox(
        "Select Algorithm",
        ["Linear Regression", "Polynomial Regression", "Moving Average"],
        help="Choose the prediction algorithm"
    )
    
    if model_type == "Polynomial Regression":
        poly_degree = st.slider("Polynomial Degree", 2, 5, 3)
    
    predict_button = st.button("🔮 Predict Stock Price", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📚 University Project")
    st.markdown("**Indian Stock Market Analysis**")
    st.markdown("*Using AI & Machine Learning*")

# Helper Functions
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    """Load stock data from Yahoo Finance for Indian stocks"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end)
        
        if df.empty:
            return None, None
        
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def calculate_metrics(df):
    """Calculate stock metrics"""
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = current_price - prev_price
    pct_change = (change / prev_price) * 100
    
    high_52w = df['High'].tail(252).max()
    low_52w = df['Low'].tail(252).min()
    avg_volume = df['Volume'].tail(30).mean()
    
    # Calculate moving averages
    ma_50 = df['Close'].tail(50).mean()
    ma_200 = df['Close'].tail(200).mean() if len(df) >= 200 else None
    
    return {
        'current_price': current_price,
        'change': change,
        'pct_change': pct_change,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'avg_volume': avg_volume,
        'ma_50': ma_50,
        'ma_200': ma_200
    }

def create_candlestick_chart(df, symbol_name):
    """Create candlestick chart with volume and moving averages"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol_name} Price Chart', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#138808',
            decreasing_line_color='#FF0000'
        ),
        row=1, col=1
    )
    
    # Add 50-day MA
    if len(df) >= 50:
        ma_50 = df['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma_50,
                name='MA 50',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    # Add 200-day MA
    if len(df) >= 200:
        ma_200 = df['Close'].rolling(window=200).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma_200,
                name='MA 200',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Volume bars
    colors = ['#FF0000' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#138808' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol_name} Stock Price Analysis',
        yaxis_title='Price (INR ₹)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def prepare_data_for_prediction(df):
    """Prepare data for ML prediction"""
    df = df.copy()
    df['Days'] = np.arange(len(df))
    return df

def predict_linear_regression(df, days):
    """Predict using Linear Regression"""
    # Prepare data
    X = np.array(df['Days']).reshape(-1, 1)
    y = np.array(df['Close'])
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    last_day = df['Days'].iloc[-1]
    future_days = np.array([last_day + i for i in range(1, days + 1)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # Calculate confidence intervals (simple estimation)
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    
    upper_bound = predictions + (2 * std_error)
    lower_bound = predictions - (2 * std_error)
    
    return predictions, upper_bound, lower_bound

def predict_polynomial_regression(df, days, degree=3):
    """Predict using Polynomial Regression"""
    from sklearn.preprocessing import PolynomialFeatures
    
    # Prepare data
    X = np.array(df['Days']).reshape(-1, 1)
    y = np.array(df['Close'])
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict future
    last_day = df['Days'].iloc[-1]
    future_days = np.array([last_day + i for i in range(1, days + 1)]).reshape(-1, 1)
    future_days_poly = poly_features.transform(future_days)
    predictions = model.predict(future_days_poly)
    
    # Calculate confidence intervals
    residuals = y - model.predict(X_poly)
    std_error = np.std(residuals)
    
    upper_bound = predictions + (2 * std_error)
    lower_bound = predictions - (2 * std_error)
    
    return predictions, upper_bound, lower_bound

def predict_moving_average(df, days, window=30):
    """Predict using Moving Average"""
    # Calculate moving average
    ma = df['Close'].rolling(window=window).mean()
    trend = ma.iloc[-1] - ma.iloc[-window]
    daily_trend = trend / window
    
    # Simple linear projection
    predictions = []
    last_price = df['Close'].iloc[-1]
    
    for i in range(1, days + 1):
        predicted_price = last_price + (daily_trend * i)
        predictions.append(predicted_price)
    
    predictions = np.array(predictions)
    
    # Calculate confidence intervals based on historical volatility
    volatility = df['Close'].pct_change().std() * df['Close'].iloc[-1]
    
    upper_bound = predictions + (2 * volatility * np.sqrt(np.arange(1, days + 1)))
    lower_bound = predictions - (2 * volatility * np.sqrt(np.arange(1, days + 1)))
    
    return predictions, upper_bound, lower_bound

def create_prediction_chart(df, predictions, upper_bound, lower_bound, symbol_name, days):
    """Create prediction chart"""
    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'{symbol_name} - Price Prediction ({days} days)',
        xaxis_title='Date',
        yaxis_title='Price (INR ₹)',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Main App Logic
if predict_button:
    with st.spinner(f'📊 Loading data for {display_name}...'):
        # Load data
        df, info = load_stock_data(stock_symbol, start_date, end_date)
        
        if df is None or df.empty:
            st.error(f"❌ Could not load data for {stock_symbol}. Please check the symbol and try again.")
            st.info("💡 Make sure to add .NS for NSE stocks or .BO for BSE stocks")
        else:
            # Reset index to have Date as a column
            df_display = df.reset_index()
            
            # Company Information Header
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("### 🏢 Company")
                company_name = info.get('longName', display_name) if info else display_name
                st.markdown(f"**{company_name}**")
            
            with col2:
                st.markdown("### 📍 Exchange")
                exchange = "NSE" if ".NS" in stock_symbol else "BSE" if ".BO" in stock_symbol else "N/A"
                st.markdown(f"**{exchange}**")
            
            with col3:
                st.markdown("### 🏭 Sector")
                sector = info.get('sector', 'N/A') if info else 'N/A'
                st.markdown(f"**{sector}**")
            
            with col4:
                st.markdown("### 🏢 Industry")
                industry = info.get('industry', 'N/A') if info else 'N/A'
                st.markdown(f"**{industry}**")
            
            st.markdown("---")
            
            # Calculate and display metrics
            metrics = calculate_metrics(df_display.set_index('Date'))
            
            st.subheader("📊 Key Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{metrics['current_price']:.2f}",
                    f"{metrics['change']:+.2f} ({metrics['pct_change']:+.2f}%)",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "52 Week High",
                    f"₹{metrics['high_52w']:.2f}"
                )
            
            with col3:
                st.metric(
                    "52 Week Low",
                    f"₹{metrics['low_52w']:.2f}"
                )
            
            with col4:
                st.metric(
                    "50 Day MA",
                    f"₹{metrics['ma_50']:.2f}"
                )
            
            with col5:
                volume_lakhs = metrics['avg_volume'] / 100000
                st.metric(
                    "Avg Volume (30d)",
                    f"{volume_lakhs:.2f}L"
                )
            
            st.markdown("---")
            
            # Historical data chart
            st.subheader("📈 Historical Price Analysis")
            fig_historical = create_candlestick_chart(df, display_name)
            st.plotly_chart(fig_historical, use_container_width=True)
            
            # Technical Analysis
            st.markdown("---")
            st.subheader("📊 Technical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price vs Moving Averages
                current = metrics['current_price']
                ma50 = metrics['ma_50']
                
                if current > ma50:
                    st.success(f"✅ Price is **above** 50-day MA (₹{ma50:.2f}) - **Bullish Signal**")
                else:
                    st.warning(f"⚠️ Price is **below** 50-day MA (₹{ma50:.2f}) - **Bearish Signal**")
            
            with col2:
                # 52-week range
                week_52_range = ((current - metrics['low_52w']) / (metrics['high_52w'] - metrics['low_52w'])) * 100
                st.info(f"📍 Stock is at **{week_52_range:.1f}%** of its 52-week range")
            
            st.markdown("---")
            
            # Predictions
            st.subheader(f"🔮 Price Predictions using {model_type}")
            
            with st.spinner('🤖 Generating predictions using Machine Learning...'):
                try:
                    # Prepare data
                    df_pred = prepare_data_for_prediction(df_display.set_index('Date').reset_index())
                    
                    # Get predictions based on selected model
                    if model_type == "Linear Regression":
                        predictions, upper_bound, lower_bound = predict_linear_regression(df_pred, prediction_days)
                    elif model_type == "Polynomial Regression":
                        predictions, upper_bound, lower_bound = predict_polynomial_regression(df_pred, prediction_days, poly_degree)
                    else:  # Moving Average
                        predictions, upper_bound, lower_bound = predict_moving_average(df_pred, prediction_days)
                    
                    # Create and display prediction chart
                    fig_pred = create_prediction_chart(
                        df,
                        predictions,
                        upper_bound,
                        lower_bound,
                        display_name,
                        prediction_days
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Forecast statistics
                    st.markdown("---")
                    st.subheader("📊 Prediction Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        predicted_price = predictions[-1]
                        st.metric(
                            "Predicted Price",
                            f"₹{predicted_price:.2f}",
                            help=f"Predicted price after {prediction_days} days"
                        )
                    
                    with col2:
                        predicted_change = predicted_price - metrics['current_price']
                        predicted_pct = (predicted_change / metrics['current_price']) * 100
                        st.metric(
                            "Expected Change",
                            f"{predicted_pct:+.2f}%",
                            f"₹{predicted_change:+.2f}",
                            delta_color="normal"
                        )
                    
                    with col3:
                        st.metric(
                            "Upper Bound",
                            f"₹{upper_bound[-1]:.2f}",
                            help="95% confidence upper limit"
                        )
                    
                    with col4:
                        st.metric(
                            "Lower Bound",
                            f"₹{lower_bound[-1]:.2f}",
                            help="95% confidence lower limit"
                        )
                    
                    # Investment Recommendation
                    st.markdown("---")
                    st.subheader("💡 AI Recommendation")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if predicted_pct > 5:
                            st.success(f"""
                            ### 🟢 STRONG BUY Signal
                            - Predicted growth: **{predicted_pct:.2f}%**
                            - Expected return: **₹{predicted_change:.2f}** per share
                            - Confidence interval: ₹{lower_bound[-1]:.2f} - ₹{upper_bound[-1]:.2f}
                            - Model used: **{model_type}**
                            """)
                        elif predicted_pct > 0:
                            st.info(f"""
                            ### 🔵 BUY Signal
                            - Predicted growth: **{predicted_pct:.2f}%**
                            - Expected return: **₹{predicted_change:.2f}** per share
                            - Confidence interval: ₹{lower_bound[-1]:.2f} - ₹{upper_bound[-1]:.2f}
                            - Model used: **{model_type}**
                            """)
                        elif predicted_pct > -5:
                            st.warning(f"""
                            ### 🟡 HOLD Signal
                            - Predicted change: **{predicted_pct:.2f}%**
                            - Expected change: **₹{predicted_change:.2f}** per share
                            - Confidence interval: ₹{lower_bound[-1]:.2f} - ₹{upper_bound[-1]:.2f}
                            - Model used: **{model_type}**
                            """)
                        else:
                            st.error(f"""
                            ### 🔴 SELL Signal
                            - Predicted decline: **{predicted_pct:.2f}%**
                            - Expected loss: **₹{predicted_change:.2f}** per share
                            - Confidence interval: ₹{lower_bound[-1]:.2f} - ₹{upper_bound[-1]:.2f}
                            - Model used: **{model_type}**
                            """)
                    
                    with col2:
                        # Risk meter
                        uncertainty = upper_bound[-1] - lower_bound[-1]
                        risk_pct = (uncertainty / predicted_price) * 100
                        
                        st.metric("Risk Level", f"{risk_pct:.1f}%")
                        
                        if risk_pct < 10:
                            st.success("Low Risk ✅")
                        elif risk_pct < 20:
                            st.info("Medium Risk ⚠️")
                        else:
                            st.warning("High Risk ⚡")
                    
                    # Detailed forecast table
                    st.markdown("---")
                    st.subheader("📋 Detailed Daily Predictions")
                    
                    # Generate future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='D')
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates.strftime('%d-%m-%Y'),
                        'Predicted Price (₹)': predictions.round(2),
                        'Lower Bound (₹)': lower_bound.round(2),
                        'Upper Bound (₹)': upper_bound.round(2)
                    })
                    
                    # Add day-wise change
                    forecast_df['Change (₹)'] = forecast_df['Predicted Price (₹)'].diff().round(2)
                    forecast_df['Change (%)'] = (forecast_df['Change (₹)'] / forecast_df['Predicted Price (₹)'].shift(1) * 100).round(2)
                    
                    st.dataframe(
                        forecast_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Download buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name=f"{display_name}_forecast_{datetime.now().strftime('%d%m%Y')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download historical data
                        hist_csv = df_display.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Historical Data",
                            data=hist_csv,
                            file_name=f"{display_name}_historical_{datetime.now().strftime('%d%m%Y')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Create analysis report
                        report = f"""
STOCK ANALYSIS REPORT
{'='*60}
Stock: {company_name}
Symbol: {stock_symbol}
Exchange: {exchange}
Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

CURRENT METRICS
{'='*60}
Current Price: ₹{metrics['current_price']:.2f}
Daily Change: ₹{metrics['change']:.2f} ({metrics['pct_change']:.2f}%)
52 Week High: ₹{metrics['high_52w']:.2f}
52 Week Low: ₹{metrics['low_52w']:.2f}
50 Day MA: ₹{metrics['ma_50']:.2f}

PREDICTION ({prediction_days} days using {model_type})
{'='*60}
Predicted Price: ₹{predicted_price:.2f}
Expected Change: ₹{predicted_change:.2f} ({predicted_pct:.2f}%)
Upper Bound: ₹{upper_bound[-1]:.2f}
Lower Bound: ₹{lower_bound[-1]:.2f}
Risk Level: {risk_pct:.1f}%

RECOMMENDATION
{'='*60}
{'STRONG BUY' if predicted_pct > 5 else 'BUY' if predicted_pct > 0 else 'HOLD' if predicted_pct > -5 else 'SELL'}

Disclaimer: This is an AI-generated prediction for educational purposes only.
Not financial advice. Please do your own research before investing.

Generated by Indian Stock Prediction App
University Project - Machine Learning Application
Model: {model_type}
"""
                        st.download_button(
                            label="📄 Download Analysis Report",
                            data=report,
                            file_name=f"{display_name}_report_{datetime.now().strftime('%d%m%Y')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    # Landing page
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%); 
                padding: 2rem; border-radius: 1rem; margin: 2rem 0;'>
        <h2 style='color: white; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
            Welcome to Indian Stock Market Analysis & Prediction System! 🚀
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🎯 Features
        
        - 📊 **Real-time NSE/BSE Data** from Yahoo Finance
        - 📈 **Interactive Charts** with candlesticks and volume
        - 🤖 **Multiple ML Models** 
          - Linear Regression
          - Polynomial Regression
          - Moving Average
        - 📉 **Technical Analysis** with moving averages
        - 💡 **Investment Recommendations** based on predictions
        - 💾 **Export Data** to CSV and generate reports
        - 🇮🇳 **Indian Currency** (₹ INR) formatting
        
        ## 🏆 Top Indian Stocks
        
        ### IT Sector
        - **TCS** - Tata Consultancy Services
        - **Infosys** - Infosys Limited
        - **Wipro** - Wipro Limited
        - **HCL Tech** - HCL Technologies
        - **Tech Mahindra** - Tech Mahindra Ltd
        
        ### Banking & Finance
        - **HDFC Bank** - HDFC Bank Limited
        - **ICICI Bank** - ICICI Bank Limited
        - **SBI** - State Bank of India
        - **Axis Bank** - Axis Bank Limited
        - **Kotak Bank** - Kotak Mahindra Bank
        """)
    
    with col2:
        st.markdown("""
        ## 📖 How to Use
        
        1. **Select a Stock**
           - Choose from popular stocks OR
           - Enter symbol manually (e.g., RELIANCE.NS)
        
        2. **Set Date Range**
           - Choose historical data period
           - More data = better predictions
        
        3. **Configure Prediction**
           - Select prediction period (7-90 days)
           - Choose ML algorithm
           - Adjust model parameters
        
        4. **Click "Predict Stock Price"**
           - View comprehensive analysis
           - Get AI-powered recommendations
           - Download reports and data
        
        ## 💡 Stock Symbol Format
        
        - **NSE Stocks**: Add `.NS` suffix
          - Example: `TCS.NS`, `RELIANCE.NS`
        
        - **BSE Stocks**: Add `.BO` suffix
          - Example: `TCS.BO`, `RELIANCE.BO`
        
        ## 📚 ML Models Available
        
        1. **Linear Regression**
           - Best for: Steady trends
           - Speed: Fast ⚡
           
        2. **Polynomial Regression**
           - Best for: Complex patterns
           - Speed: Medium ⚡⚡
           
        3. **Moving Average**
           - Best for: Short-term predictions
           - Speed: Very Fast ⚡⚡⚡
        """)
    
    st.markdown("---")
    
    # Popular stocks quick access
    st.subheader("🔥 Quick Access - Popular Stocks")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    quick_stocks = [
        ("Reliance", "RELIANCE.NS"),
        ("TCS", "TCS.NS"),
        ("Infosys", "INFY.NS"),
        ("HDFC Bank", "HDFCBANK.NS"),
        ("ITC", "ITC.NS")
    ]
    
    for idx, (col, (name, symbol)) in enumerate(zip([col1, col2, col3, col4, col5], quick_stocks)):
        with col:
            if st.button(f"📊 {name}", use_container_width=True, key=f"quick_{idx}"):
                st.info(f"Select '{name}' from sidebar and click 'Predict Stock Price'")
    
    st.markdown("---")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Important Disclaimer**
    
    This application is developed for **educational and academic purposes only** as part of a university project.
    
    - Stock market predictions are based on historical data and ML algorithms
    - Past performance does not guarantee future results
    - This is **NOT financial advice**
    - Always consult with a qualified financial advisor before making investment decisions
    - The developers are not responsible for any financial losses incurred
    
    **Please do your own research and invest responsibly!**
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
        <p style='margin: 0.5rem 0;'><b>🎓 University Project - Indian Stock Market Prediction System</b></p>
        <p style='margin: 0.5rem 0;'>Made with ❤️ using Streamlit | Data from Yahoo Finance | ML Models: scikit-learn</p>
        <p style='margin: 0.5rem 0;'>🇮🇳 Customized for NSE/BSE with INR Currency</p>
        <p style='margin: 0.5rem 0; font-size: 0.8rem;'>For Educational Purposes Only | Not Financial Advice</p>
    </div>
    """, unsafe_allow_html=True)
