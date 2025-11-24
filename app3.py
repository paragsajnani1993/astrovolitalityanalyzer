import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ephem
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict

# --- CONFIGURATION ---
st.set_page_config(page_title="Astro-Swing Analyzer (Mumbai)", layout="wide")

# --- 1. ASTROLOGY ENGINE (MUMBAI LOCALIZED) ---
planet_names = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
           'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

def get_planet_state(check_date):
    # Mumbai Location
    obs = ephem.Observer()
    obs.lat = '19.0760'
    obs.lon = '72.8777'
    obs.elevation = 0
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
    # Robust date handling
    if hasattr(current_date, 'to_pydatetime'):
        d = current_date.to_pydatetime()
    else:
        d = current_date

    curr_state = get_planet_state(d)
    events = []
    
    # Ingress
    for name in planet_names:
        if curr_state[name]['sign_idx'] != prev_state[name]['sign_idx']:
            new_sign = zodiacs[curr_state[name]['sign_idx'] % 12]
            events.append(f"{name} enters {new_sign}")
            
    # New Aspects
    asp_curr = get_aspects(curr_state)
    asp_prev = get_aspects(prev_state)
    new_asps = list(asp_curr - asp_prev)
    events.extend(new_asps)
    
    return events, curr_state

# --- 2. FUTURE PROJECTION ENGINE ---
def find_future_dates(event_name, limit=5):
    future_dates = []
    start_date = datetime.now() + timedelta(days=1)
    end_limit = start_date + timedelta(days=365*5) 
    
    current_scan = start_date
    prev_state = get_planet_state(start_date - timedelta(days=1))
    
    while current_scan < end_limit and len(future_dates) < limit:
        day_events, curr_state = get_daily_events(current_scan, prev_state)
        if event_name in day_events:
            future_dates.append(current_scan.date())
        prev_state = curr_state
        current_scan += timedelta(days=1)
        
    return future_dates

# --- 3. PERFORMANCE ENGINE (SWING HIGH/LOW) ---
def analyze_market_performance(df, lookahead_days):
    """
    Calculates Max Swing High and Max Swing Low within the window.
    """
    # Key = Event, Value = List of results
    swing_highs = defaultdict(list)
    swing_lows = defaultdict(list)
    event_details = defaultdict(list)
    
    start_date = df.index[0]
    if hasattr(start_date, 'to_pydatetime'):
        start_py = start_date.to_pydatetime()
    else:
        start_py = start_date

    prev_state = get_planet_state(start_py - timedelta(days=1))
    
    my_bar = st.progress(0, text="Scanning Max Swings in Window...")
    total_days = len(df)
    limit_index = total_days - lookahead_days
    
    for i in range(total_days):
        date = df.index[i]
        price_trigger = df['Close'].iloc[i] # Baseline is Event Day Close
        
        daily_events_list, curr_state = get_daily_events(date, prev_state)
        prev_state = curr_state
        
        if i < limit_index:
            # Create a window slice from i+1 to i+lookahead
            window = df.iloc[i+1 : i+1+lookahead_days]
            
            # Find Extremes in the window
            max_price = window['High'].max()
            min_price = window['Low'].min()
            
            # Calculate % Swing from Trigger Close
            pct_swing_high = ((max_price - price_trigger) / price_trigger) * 100
            pct_swing_low = ((min_price - price_trigger) / price_trigger) * 100
            
            for event in daily_events_list:
                swing_highs[event].append(pct_swing_high)
                swing_lows[event].append(pct_swing_low)
                
                event_details[event].append({
                    'Date': date,
                    'Trigger Price': price_trigger,
                    'Window High': max_price,
                    'Window Low': min_price,
                    'Swing High %': pct_swing_high,
                    'Swing Low %': pct_swing_low
                })
        else:
            # Recent events (no full window data)
            for event in daily_events_list:
                event_details[event].append({
                    'Date': date, 'Trigger Price': price_trigger, 
                    'Window High': None, 'Window Low': None, 
                    'Swing High %': None, 'Swing Low %': None
                })

        if i % 50 == 0: my_bar.progress((i / total_days))
        
    my_bar.empty()
    return swing_highs, swing_lows, event_details

# --- 4. UI & LOGIC ---
st.title("📈 Astro-Swing Analyzer")
st.markdown("""
**Metric:** Maximum Profit Potential.
* **Swing High:** The max % gain achievable within the window (Long potential).
* **Swing Low:** The max % drop achievable within the window (Short potential).
""")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value="^NSEI")
    period_map = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
    period_sel = st.selectbox("History Length", options=list(period_map.keys()))
    
    st.divider()
    st.subheader("Performance Logic")
    lookahead = st.number_input("Lookahead Window (Days)", min_value=1, value=7, 
                                help="Scans for the Highest High and Lowest Low in this window.")
    
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
            highs_dict, lows_dict, details_dict = analyze_market_performance(df, lookahead)
            
            # AGGREGATE DATA
            summary_data = []
            for event in highs_dict.keys():
                h_list = highs_dict[event]
                l_list = lows_dict[event]
                
                if not h_list: continue
                
                occurrences = len(h_list)
                
                # Averages
                avg_high = np.mean(h_list)
                avg_low = np.mean(l_list)
                
                # "Reliability" (Win Rate logic adapted for Swings)
                # How often did it swing at least +1%?
                reliable_upside = sum(1 for x in h_list if x > 1.0)
                upside_prob = (reliable_upside / occurrences) * 100
                
                # How often did it swing at least -1%?
                reliable_downside = sum(1 for x in l_list if x < -1.0)
                downside_prob = (reliable_downside / occurrences) * 100
                
                summary_data.append({
                    "Event": event,
                    "Avg Swing High %": avg_high,
                    "Avg Swing Low %": avg_low,
                    "Occurrences": occurrences,
                    "Upside Prob (>1%)": upside_prob,
                    "Downside Prob (<-1%)": downside_prob
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["📊 Swing Averages", "🔍 Event Drill-Down & Forecast", "🧠 Strategic Insights"])
            
            with tab1:
                st.subheader(f"Average Max Excursion ({lookahead}-Day Window)")
                st.caption("Green = Max profit for Longs. Red = Max profit for Shorts.")
                
                st.dataframe(
                    summary_df.style.format({
                        "Avg Swing High %": "{:.2f}%", 
                        "Avg Swing Low %": "{:.2f}%",
                        "Upside Prob (>1%)": "{:.1f}%",
                        "Downside Prob (<-1%)": "{:.1f}%"
                    }).background_gradient(cmap='RdYlGn', subset=['Avg Swing High %', 'Avg Swing Low %']),
                    use_container_width=True
                )
                
            with tab2:
                st.subheader("Deep Dive: Analyze Specific Event")
                event_list = summary_df['Event'].tolist()
                selected_event = st.selectbox("Select Astro Event", event_list)
                
                if selected_event:
                    # 1. HISTORICAL DATA
                    details = details_dict[selected_event]
                    detail_df = pd.DataFrame(details)
                    
                    stats = summary_df[summary_df['Event'] == selected_event].iloc[0]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Upside Potential", f"{stats['Avg Swing High %']:.2f}%")
                    c2.metric("Avg Downside Potential", f"{stats['Avg Swing Low %']:.2f}%")
                    c3.metric("Total Hits", len(detail_df))
                    
                    st.markdown(f"### 📜 History of '{selected_event}'")
                    st.dataframe(detail_df.style.format({
                        "Trigger Price": "{:.2f}", "Window High": "{:.2f}", "Window Low": "{:.2f}",
                        "Swing High %": "{:.2f}%", "Swing Low %": "{:.2f}%"
                    }), use_container_width=True)
                    
                    # 2. FUTURE PROJECTION
                    st.markdown("---")
                    st.markdown(f"### 🔮 Upcoming Dates: {selected_event}")
                    with st.spinner(f"Scanning future dates for {selected_event}..."):
                        future_dates = find_future_dates(selected_event)
                        if future_dates:
                            future_df = pd.DataFrame(future_dates, columns=["Next Occurrence Date"])
                            st.table(future_df)
                        else:
                            st.warning("No occurrences found in the next 5 years.")
                    
                    # Chart
                    st.markdown("---")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'], name='Price', opacity=0.3))
                    fig.add_trace(go.Scatter(
                        x=detail_df['Date'], y=detail_df['Trigger Price'],
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
                        st.success(f"**🚀 Best Long Opportunities**")
                        st.caption("Sorted by Avg Swing High %")
                        bulls = sig_df.sort_values("Avg Swing High %", ascending=False).head(5)
                        st.table(bulls[['Event', 'Avg Swing High %', 'Occurrences']])
                        
                    with col_b:
                        st.error(f"**🔻 Best Short Opportunities**")
                        st.caption("Sorted by Avg Swing Low % (Most Negative)")
                        bears = sig_df.sort_values("Avg Swing Low %", ascending=True).head(5)
                        st.table(bears[['Event', 'Avg Swing Low %', 'Occurrences']])
                        
                    st.info("Note: 'Swing High' is the max profit if you bought. 'Swing Low' is the max profit if you shorted.")

    except Exception as e:
        st.error(f"Error: {e}")
