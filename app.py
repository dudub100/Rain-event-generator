import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfinv
import itur
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# ==========================================
# Caching Heavy Computations & API Calls
# ==========================================
@st.cache_data(show_spinner=False)
def get_coordinates(city_name):
    """Fetches latitude and longitude for a given city name using Geopy."""
    geolocator = Nominatim(user_agent="mmw_rain_fade_simulator")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None, None, None

@st.cache_data(show_spinner=False)
def calculate_link_statistics(lat, lon, freq_GHz, distance_km, availability=0.01):
    """Calculates ITU-R P.837 and P.530 rain statistics."""
    # 1. ITU-R P.837: Rain rate
    R_001 = itur.models.itu837.rainfall_rate(lat, lon, p=availability)
    
    # 2. ITU-R P.837: Probability of rain
    try:
        P_rain = float(itur.models.itu837.rainfall_probability(lat, lon)) / 100.0
    except AttributeError:
        P_rain = 0.05 # Fallback if specific grid data is missing
        
    # 3. ITU-R P.530: Terrestrial Path Attenuation 
    # Note: P.530 requires an elevation angle (el), which is 0 for terrestrial links
    A_001 = itur.models.itu530.rain_attenuation(lat, lon, distance_km, freq_GHz, el=0, p=availability, tau=90)
    
    # Return standard float values to keep Streamlit cache happy
    return float(R_001.value), float(A_001.value), float(P_rain)


@st.cache_data
def convert_df_to_csv(df):
    """Converts a pandas DataFrame to a utf-8 encoded CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# Core Stochastic Synthesizer (P.1853)
# ==========================================
def synthesize_p1853_time_series(A_001, P_rain, duration_hours=72, Ts=1.0):
    """Generates synthetic rain attenuation utilizing a filtered Markov process."""
    # beta defines the terrestrial time dynamics (fade slope)
    beta = 2e-4 
    
    P_norm = 0.0001 / P_rain
    x_001 = np.sqrt(2) * erfinv(1 - 2 * P_norm)
    
    sigma = 0.5 * x_001 
    m = np.log(A_001) - sigma * x_001
    
    N_samples = int(duration_hours * 3600 / Ts)
    n_t = np.random.normal(0, 1, N_samples)
    
    rho = np.exp(-beta * Ts)
    X_t = np.zeros(N_samples)
    
    for k in range(1, N_samples):
        X_t[k] = rho * X_t[k-1] + np.sqrt(1 - rho**2) * n_t[k]
        
    Y_t = np.exp(m + sigma * X_t)
    
    threshold_idx = int(N_samples * (1 - P_rain))
    if threshold_idx >= N_samples:
        A_offset = 0
    else:
        Y_sorted = np.sort(Y_t)
        A_offset = Y_sorted[threshold_idx]
        
    A_rain = np.maximum(0, Y_t - A_offset)
    time_axis = np.arange(N_samples) * Ts / 3600.0 
    
    return time_axis, A_rain

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="ITU-R Synthetic Rain Fade Generator", layout="wide")
st.title("📡 Synthetic Rain Fade Time-Series Generator")
st.markdown("Generates physical RSL drop signatures based on **ITU-R P.1853**, **P.837**, and **P.530** to train anomaly detection models for Point-to-Point links.")

# Sidebar Configuration
st.sidebar.header("1. Link Configuration")
city_input = st.sidebar.text_input("City/Location", value="Tel Aviv, Israel")
freq_GHz = st.sidebar.number_input("Frequency (GHz)", min_value=1.0, max_value=100.0, value=18.0, step=1.0)
distance_km = st.sidebar.number_input("Link Length (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)

st.sidebar.header("2. Simulation Parameters")
duration_hours = st.sidebar.number_input("Duration of each Series (Hours)", min_value=1, max_value=720, value=72)
n_series = st.sidebar.number_input("Number of Time Series (N)", min_value=1, max_value=100, value=5)

# Main Execution Logic
if st.sidebar.button("Generate Simulation", type="primary"):
    with st.spinner(f"Geocoding {city_input}..."):
        lat, lon, address = get_coordinates(city_input)
        
    if lat is None:
        st.error(f"Could not find coordinates for '{city_input}'. Please try a different name.")
    else:
        st.success(f"**Location Found:** {address} (Lat: {lat:.4f}, Lon: {lon:.4f})")
        
        with st.spinner("Querying ITU-R environmental baselines..."):
            R_001, A_001, P_rain = calculate_link_statistics(lat, lon, freq_GHz, distance_km)
            
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Target Rain Rate ($R_{0.01}$)", value=f"{R_001:.2f} mm/hr")
        col2.metric(label="Max Attenuation Threshold ($A_{0.01}$)", value=f"{A_001:.2f} dB")
        col3.metric(label="Probability of Rain ($P_{rain}$)", value=f"{P_rain*100:.2f}%")

        # Synthesize N Time Series
        st.subheader(f"Generated {n_series} Attenuation Signatures")
        progress_bar = st.progress(0)
        
        df_export = pd.DataFrame()
        
        # Plotting Setup
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(-A_001, color='red', linestyle='--', label=f'99.99% Availability Limit (-{A_001:.1f} dB)')

        for i in range(n_series):
            time_hrs, attenuation_dB = synthesize_p1853_time_series(A_001, P_rain, duration_hours)
            
            # Save the first time_axis just once
            if i == 0:
                df_export['Time_Hours'] = time_hrs
                
            col_name = f"Series_{i+1}_dB"
            df_export[col_name] = -attenuation_dB # Save as RSL drop (negative)
            
            # Plot only the first few to avoid completely cluttered charts
            if i < 5:
                ax.plot(time_hrs, -attenuation_dB, linewidth=1, alpha=0.8, label=f"Series {i+1}")
                
            progress_bar.progress((i + 1) / n_series)
            
        # Finalize Plot
        ax.set_title(f"Synthetic ITU-R P.1853 Rain Fade Dynamics\n{freq_GHz} GHz over {distance_km} km at {city_input}")
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("RSL Attenuation (dB)")
        ax.grid(True, alpha=0.3)
        if n_series <= 5:
            ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # CSV Export Preparation
        st.subheader("Data Export")
        st.dataframe(df_export.head(10), use_container_width=True)
        csv = convert_df_to_csv(df_export)
        
        st.download_button(
            label="⬇️ Download All Series as CSV",
            data=csv,
            file_name=f"rain_fade_sim_{freq_GHz}GHz_{distance_km}km.csv",
            mime="text/csv",
            type="primary"
        )
