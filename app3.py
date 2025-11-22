import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ephem
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict

# --- CONFIGURATION ---
st.set_page_config(page_title="Astro-Volatility Analyzer", layout="wide")

# --- 1. ASTROLOGY ENGINE ---
planet_names = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
           'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

def get_planet_state(check_date):
    obs = ephem.Observer()
    obs.date = check_date
    state = {}
    for name in planet_names:
        body = getattr(ephem, name)()
        body.compute(obs)
        lon = np.degrees(ephem.Ecliptic(body).lon)
        state[name] = {'lon': lon, 'sign_idx': int(lon / 30)}
    return state

def get_aspects(state_dict):
    found = set()
    names = list(state_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            l1, l2 = state_dict[p1]['lon'], state_dict[p2]['lon']
            diff = abs(l1 - l2)
            if diff > 180: diff = 360 - diff
            if abs(diff - 0) < 3.0: found.add(f"{p1} conj {p2}")
            elif abs(diff - 180) < 3.0: found.add(f"{p1} opp {p2}")
            elif abs(diff - 120) < 3.0: found.add(f"{p1} tri {p2}")
            elif abs(diff - 90) < 3.0: found.add(f"{p1} sq {p2}")
    return found

def get_daily_events(current_date, prev_state):
    # --- FIX: ROBUST DATE HANDLING ---
    # Handles both Pandas Timestamp (from Dataframe) and Python Datetime (from Future Scanner)
    if hasattr(current_date, 'to_pydatetime'):
        d = current_date.to_pydatetime()
    else:
        d = current_date

    curr_state = get_planet_state(d)
    
    events = []
    
    # 1. Ingress
    for name in planet_names:
        if curr_state[name]['sign_idx'] != prev_state[name]['sign_idx']:
            new_sign = zodiacs[curr_state[name]['sign_idx'] % 12]
            events.append(f"{name} enters {new_sign}")
            
    # 2. New Aspects
    asp_curr = get_aspects(curr_state)
    asp_prev = get_aspects(prev_state)
    new_asps = list(asp_curr - asp_prev)
    events.extend(new_asps)
    
    return events, curr_state

# --- 2. FUTURE PROJECTION ENGINE ---
def find_future_dates(event_name, limit=5):
    """
    Parses the event name and scans the future to find the next occurrences.
    """
    future_dates = []
    
    # Start from Tomorrow
    start_date = datetime.now() + timedelta(days=1)
    end_limit = start_date + timedelta(days=365*5) # 5 Year Cap
    
    current_scan = start_date
    prev_state = get_planet_state(start_date - timedelta(days=1))
    
    # Scan day by day
    while current_scan < end_limit and len(future_dates) < limit:
        day_events, curr_state = get_daily_events(current_scan, prev_state)
        
        if event_name in day_events:
            future_dates.append(current_scan.date())
            
        prev_state = curr_state
        current_scan += timedelta(days=1)
        
    return future_dates

# --- 3. PERFORMANCE ENGINE (ABSOLUTE RETURN) ---
def analyze_market_performance(df, lookahead_days):
    event_returns = defaultdict(list)
    event_dates = defaultdict(list)
    
    # Handle Timestamp for start date
    start_date = df.index[0]
    if hasattr(start_date, 'to_pydatetime'):
        start_py = start_date.to_pydatetime()
    else:
        start_py = start_date

    prev_state = get_planet_state(start_py - timedelta(days=1))
    
    my_bar = st.progress(0, text="Calculating Absolute Movements...")
    total_days = len(df)
    limit_index = total_days - lookahead_days
    
    for i in range(total_days):
        date = df.index[i]
        price_today = df['Close'].iloc[i]
        
        daily_events_list, curr_state = get_daily_events(date, prev_state)
        prev_state = curr_state
        
        if i < limit_index:
            price_future = df['Close'].iloc[i + lookahead_days]
            pct_change = ((price_future - price_today) / price_today) * 100
            
            for event in daily_events_list:
                event_returns[event].append(pct_change)
                event_dates[event].append({'Date': date, 'Return': pct_change, 'Price': price_today})
        else:
            for event in daily_events_list:
                event_dates[event].append({'Date': date, 'Return': None, 'Price': price_today})

        if i % 50 == 0: my_bar.progress((i / total_days))
        
    my_bar.empty()
    return event_returns, event_dates

# --- 4. UI & LOGIC ---
st.title("📈 Astro-Volatility Analyzer")
st.markdown("**Calculation Mode:** Absolute Return (Magnitude).")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value="^NSEI")
    period_map = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
    period_sel = st.selectbox("History Length", options=list(period_map.keys()))
    
    st.divider()
    st.subheader("Performance Logic")
    lookahead = st.number_input("Check Movement After (Days)", min_value=1, value=7, 
                                help="Calculates the % Return X days after the event occurs.")
    
    if st.button("Calculate Performance"):
        st.session_state['run_perf'] = True

if st.session_state.get('run_perf', False):
    try:
        df = yf.download(ticker, period=period_map[period_sel], progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            st.error("No Data Found.")
        else:
            # RUN ANALYSIS
            event_returns_dict, event_dates_dict = analyze_market_performance(df, lookahead)
            
            # AGGREGATE DATA
            summary_data = []
            for event, returns in event_returns_dict.items():
                if not returns: continue
                
                abs_returns = [abs(r) for r in returns]
                avg_abs_ret = np.mean(abs_returns)
                median_abs_ret = np.median(abs_returns)
                
                occurrences = len(returns)
                positive_trades = sum(1 for r in returns if r > 0)
                win_rate = (positive_trades / occurrences) * 100
                
                summary_data.append({
                    "Event": event,
                    "Avg Move (Abs) %": avg_abs_ret,
                    "Win Rate %": win_rate,
                    "Occurrences": occurrences,
                    "Median Move (Abs) %": median_abs_ret
                })
            
            summary_df = pd.DataFrame(summary_data).sort_values(by="Avg Move (Abs) %", ascending=False)
            
            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["📊 Market Averages", "🔍 Event Drill-Down & Forecast", "🧠 Strategic Insights"])
            
            with tab1:
                st.subheader(f"Average Magnitude of Move ({lookahead}-Day Horizon)")
                st.dataframe(
                    summary_df.style.format({
                        "Avg Move (Abs) %": "{:.2f}%", 
                        "Win Rate %": "{:.1f}%",
                        "Median Move (Abs) %": "{:.2f}%"
                    }).background_gradient(cmap='Blues', subset=['Avg Move (Abs) %']),
                    use_container_width=True
                )
                
            with tab2:
                st.subheader("Deep Dive: Analyze Specific Event")
                event_list = summary_df['Event'].tolist()
                selected_event = st.selectbox("Select Astro Event", event_list)
                
                if selected_event:
                    # 1. HISTORICAL DATA
                    details = event_dates_dict[selected_event]
                    detail_df = pd.DataFrame(details)
                    detail_df['Abs Return'] = detail_df['Return'].abs()
                    
                    stats = summary_df[summary_df['Event'] == selected_event].iloc[0]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Magnitude", f"{stats['Avg Move (Abs) %']:.2f}%")
                    c2.metric("Directional Bias (Win Rate)", f"{stats['Win Rate %']:.1f}%")
                    c3.metric("Total Hits", len(detail_df))
                    
                    st.markdown(f"### 📜 History of '{selected_event}'")
                    st.dataframe(detail_df.style.format({"Return": "{:.2f}%", "Price": "{:.2f}", "Abs Return": "{:.2f}%"}), use_container_width=True)
                    
                    # 2. FUTURE PROJECTION
                    st.markdown("---")
                    st.markdown(f"### 🔮 Upcoming Dates: {selected_event}")
                    with st.spinner(f"Scanning future dates for {selected_event}..."):
                        future_dates = find_future_dates(selected_event)
                        
                        if future_dates:
                            future_df = pd.DataFrame(future_dates, columns=["Next Occurrence Date"])
                            st.table(future_df)
                        else:
                            st.warning("No occurrences found in the next 5 years (Event might be very rare).")
                    
                    # Chart
                    st.markdown("---")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'], name='Price', opacity=0.3))
                    fig.add_trace(go.Scatter(
                        x=detail_df['Date'], y=detail_df['Price'],
                        mode='markers', name=selected_event,
                        marker=dict(color='purple', size=10, symbol='diamond')
                    ))
                    fig.update_layout(title=f"Chart: Occurrences of {selected_event}", height=500)
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Automated Insights")
                min_obs = 3
                sig_df = summary_df[summary_df['Occurrences'] >= min_obs]
                
                if sig_df.empty:
                    st.warning("Not enough data points (min 3) to generate reliable insights.")
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.success(f"**🚀 Top Bullish Signals**")
                        st.caption("High Win Rate (>60%) + High Impact")
                        bulls = sig_df[sig_df['Win Rate %'] >= 60].sort_values("Avg Move (Abs) %", ascending=False).head(5)
                        st.table(bulls[['Event', 'Avg Move (Abs) %', 'Win Rate %']])
                        
                    with col_b:
                        st.error(f"**🔻 Top Bearish Signals**")
                        st.caption("Low Win Rate (<40%) + High Impact")
                        bears = sig_df[sig_df['Win Rate %'] <= 40].sort_values("Avg Move (Abs) %", ascending=False).head(5)
                        st.table(bears[['Event', 'Avg Move (Abs) %', 'Win Rate %']])

    except Exception as e:
        st.error(f"Error: {e}")