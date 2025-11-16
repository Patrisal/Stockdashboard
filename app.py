import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Smart Buy Signal Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .buy-signal {background-color: #1a4d2e; padding: 20px; border-radius: 10px; border: 2px solid #4ade80;}
    .hold-signal {background-color: #4d4d1a; padding: 20px; border-radius: 10px; border: 2px solid #fbbf24;}
    .sell-signal {background-color: #4d1a1a; padding: 20px; border-radius: 10px; border: 2px solid #ef4444;}
</style>
""", unsafe_allow_html=True)

sia = SentimentIntensityAnalyzer()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, period=20, std=2):
    sma = data['Close'].rolling(window=period).mean()
    std_dev = data['Close'].rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def calculate_moving_averages(data):
    ma20 = data['Close'].rolling(window=20).mean()
    ma50 = data['Close'].rolling(window=50).mean()
    ma200 = data['Close'].rolling(window=200).mean()
    return ma20, ma50, ma200

def calculate_volume_spike(data, period=20):
    avg_volume = data['Volume'].rolling(window=period).mean()
    return data['Volume'] / avg_volume

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')
        return data if not data.empty else None
    except:
        return None

def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0.0, [], "No news"
        
        sentiments = []
        news_list = []
        
        for article in news[:10]:
            title = article.get('title', '')
            sentiment = sia.polarity_scores(title)['compound']
            sentiments.append(sentiment)
            
            label = "🟢 Positive" if sentiment >= 0.05 else "🟡 Neutral" if sentiment >= -0.05 else "🔴 Negative"
            
            news_list.append({
                'title': title,
                'publisher': article.get('publisher', 'Unknown'),
                'sentiment': label,
                'score': sentiment,
                'link': article.get('link', '')
            })
        
        avg = np.mean(sentiments) if sentiments else 0.0
        
        if avg >= 0.15:
            trend = "Very Positive 📈"
        elif avg >= 0.05:
            trend = "Positive 📊"
        elif avg <= -0.15:
            trend = "Very Negative 📉"
        elif avg <= -0.05:
            trend = "Negative 📊"
        else:
            trend = "Neutral ➖"
        
        return avg, news_list, trend
    except:
        return 0.0, [], "Error"

def calculate_technical_score(data, ticker):
    if data is None or len(data) < 200:
        return 0, ["Insufficient data"]
    
    score = 0
    signals = []
    latest = data.iloc[-1]
    
    rsi = calculate_rsi(data)
    latest_rsi = rsi.iloc[-1]
    
    if latest_rsi < 30:
        score += 20
        signals.append(f"🟢 RSI Oversold ({latest_rsi:.1f})")
    elif latest_rsi < 40:
        score += 12
        signals.append(f"🟢 RSI Low ({latest_rsi:.1f})")
    elif latest_rsi > 70:
        score -= 10
        signals.append(f"🔴 RSI Overbought ({latest_rsi:.1f})")
    else:
        score += 5
        signals.append(f"🟡 RSI Neutral ({latest_rsi:.1f})")
    
    macd, signal_line = calculate_macd(data)
    if macd.iloc[-1] > signal_line.iloc[-1]:
        score += 8
        signals.append("🟢 MACD Positive")
    else:
        score += 2
        signals.append("🔴 MACD Negative")
    
    ma20, ma50, ma200 = calculate_moving_averages(data)
    price = latest['Close']
    
    ma_score = 0
    if price > ma20.iloc[-1]:
        ma_score += 3
    if price > ma50.iloc[-1]:
        ma_score += 3
    if price > ma200.iloc[-1]:
        ma_score += 4
    
    score += ma_score
    signals.append(f"{'🟢' if ma_score >= 8 else '🟡' if ma_score >= 5 else '🔴'} MA Score: {ma_score}/10")
    
    upper, middle, lower = calculate_bollinger_bands(data)
    bb_pos = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if upper.iloc[-1] != lower.iloc[-1] else 0.5
    
    if bb_pos < 0.2:
        score += 10
        signals.append("🟢 Near Lower BB")
    elif bb_pos > 0.8:
        score -= 5
        signals.append("🔴 Near Upper BB")
    else:
        score += 3
        signals.append("🟡 Mid BB Range")
    
    vol_ratio = calculate_volume_spike(data)
    latest_vol = vol_ratio.iloc[-1]
    
    if latest_vol > 1.5:
        score += 5
        signals.append(f"🟢 High Volume ({latest_vol:.1f}x)")
    else:
        score += 1
        signals.append(f"🟡 Normal Volume ({latest_vol:.1f}x)")
    
    return max(0, min(60, score)), signals

def calculate_sentiment_score(sentiment_avg):
    if sentiment_avg >= 0.5:
        return 30 + (sentiment_avg - 0.5) * 20
    elif sentiment_avg >= 0.15:
        return 20 + (sentiment_avg - 0.15) * 28.6
    elif sentiment_avg >= 0.05:
        return 10 + (sentiment_avg - 0.05) * 100
    elif sentiment_avg >= -0.05:
        return 10 + sentiment_avg * 100
    else:
        return max(-20, -20 + (sentiment_avg + 0.5) * 28.6)

st.title("📈 Smart Buy Signal Dashboard")
st.markdown("### Technical Analysis + Market Sentiment = Smart Trading")

st.sidebar.header("⚙️ Configuration")
default_tickers = "AAPL,MSFT,GOOGL,TSLA,NVDA"
ticker_input = st.sidebar.text_input("Enter 5 Stock Tickers", value=default_tickers)
tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

if len(tickers) != 5:
    st.sidebar.warning(f"Please enter exactly 5 tickers. You entered {len(tickers)}.")
    tickers = default_tickers.split(',')

st.sidebar.success(f"Tracking: {', '.join(tickers)}")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

st.sidebar.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

with st.spinner("📊 Fetching data..."):
    summary_data = []
    
    for ticker in tickers:
        st.markdown("---")
        st.markdown(f"## {ticker}")
        
        data = get_stock_data(ticker)
        if data is None or len(data) < 200:
            st.error(f"Unable to fetch data for {ticker}")
            continue
        
        ma20, ma50, ma200 = calculate_moving_averages(data)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(data)
        rsi = calculate_rsi(data)
        
        sentiment_avg, news_list, sentiment_trend = get_news_sentiment(ticker)
        tech_score, tech_signals = calculate_technical_score(data, ticker)
        sent_score = calculate_sentiment_score(sentiment_avg)
        buy_score = max(0, min(100, tech_score + sent_score))
        
        if buy_score >= 70:
            rec = "🟢 STRONG BUY"
            rec_class = "buy-signal"
        elif buy_score >= 55:
            rec = "🟢 BUY"
            rec_class = "buy-signal"
        elif buy_score >= 40:
            rec = "🟡 HOLD"
            rec_class = "hold-signal"
        else:
            rec = "🔴 SELL"
            rec_class = "sell-signal"
        
        price = data['Close'].iloc[-1]
        change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Price", f"${price:.2f}", f"{change:+.2f}%")
        col2.metric("RSI", f"{rsi.iloc[-1]:.1f}")
        col3.metric("Tech", f"{tech_score:.0f}/60")
        col4.metric("Sentiment", f"{sent_score:.0f}/40")
        col5.metric("Buy Score", f"{buy_score:.0f}/100")
        
        st.markdown(f'<div class="{rec_class}"><h2 style="text-align:center;margin:0;">{rec}</h2><h3 style="text-align:center;margin:10px 0 0 0;">Score: {buy_score:.0f}/100</h3></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Signals", "📊 Technical", "📰 News"])
        
        with tab1:
            st.markdown("### Technical Signals")
            for signal in tech_signals:
                st.markdown(f"- {signal}")
        
        with tab2:
            st.markdown("### Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("20-Day MA", f"${ma20.iloc[-1]:.2f}")
            col2.metric("50-Day MA", f"${ma50.iloc[-1]:.2f}")
            col3.metric("200-Day MA", f"${ma200.iloc[-1]:.2f}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Upper BB", f"${upper_bb.iloc[-1]:.2f}")
            col2.metric("Middle BB", f"${middle_bb.iloc[-1]:.2f}")
            col3.metric("Lower BB", f"${lower_bb.iloc[-1]:.2f}")
        
        with tab3:
            st.markdown(f"### Sentiment: {sentiment_trend}")
            st.metric("Avg Sentiment", f"{sentiment_avg:.3f}")
            for news in news_list[:10]:
                with st.expander(f"{news['sentiment']} {news['title'][:70]}..."):
                    st.markdown(f"**{news['title']}**")
                    st.markdown(f"Publisher: {news['publisher']} | Score: {news['score']:.3f}")
        
        summary_data.append({
            'Ticker': ticker,
            'Price': f"${price:.2f}",
            'Change': f"{change:+.2f}%",
            'RSI': f"{rsi.iloc[-1]:.1f}",
            'Tech': f"{tech_score:.0f}",
            'Sent': f"{sent_score:.0f}",
            'Score': f"{buy_score:.0f}",
            'Signal': rec
        })
    
    if summary_data:
        st.markdown("---")
        st.markdown("## 📋 Summary")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'><p><strong>Disclaimer:</strong> Educational purposes only. Not financial advice.</p></div>", unsafe_allow_html=True)
