# app.py

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import requests
import warnings
from datetime import datetime

# Suppress specific FutureWarnings related to frequency codes
warnings.simplefilter(action='ignore', category=FutureWarning)

def fetch_ons_data(dataset_id, series_id):
    url = f"https://api.ons.gov.uk/timeseries/{series_id}/dataset/{dataset_id}/data"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        data = response.json()
        
        # Ensure 'years' key exists in the response
        if 'years' not in data:
            st.error("Error: 'years' data not found in the response.")
            return None
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['years'])
        
        # Ensure 'date' and 'value' columns exist in the DataFrame
        if 'date' not in df.columns or 'value' not in df.columns:
            st.error("Error: 'date' or 'value' columns missing in the data.")
            return None
        
        # Convert 'date' and 'value' to appropriate types
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'])
        
        # Return the DataFrame with 'date' as the index
        return df.set_index('date')['value']
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
        return None
    except Exception as e:
        st.error(f"Error fetching data from ONS API: {e}")
        return None

def generate_mock_energy_data(start_year=2010, end_year=2023):
    years = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='YE')
    np.random.seed(42)
    electricity_prices = 100 + np.cumsum(np.random.normal(5, 2, len(years)))
    gas_prices = 80 + np.cumsum(np.random.normal(3, 1, len(years)))
    return pd.DataFrame({
        'ds': years,
        'electricity_price': electricity_prices,
        'gas_price': gas_prices
    })

# Title and Description
st.set_page_config(page_title="Bodycote Forecast Dashboard", layout="wide")
st.title("📈 Bodycote Forecast Dashboard")
st.markdown("""
This interactive dashboard provides forecasts of Bodycote's revenue and energy costs over the next few years using historical data and the Prophet forecasting model.
""")

# Sidebar Inputs
st.sidebar.header("🔧 Configuration")
forecast_period = st.sidebar.slider("Forecasting Period (Years)", min_value=1, max_value=10, value=5)
energy_cost_percentage = st.sidebar.number_input("Energy Cost as Percentage of Revenue (%)", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
efficiency_improvement_rate = st.sidebar.number_input("Annual Efficiency Improvement Rate (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

# Create a list of cumulative improvements over the forecast period
efficiency_improvements = [(efficiency_improvement_rate * (year + 1)) for year in range(forecast_period)]

# Data Fetching and Preparation
@st.cache_data
def load_data():
    # Fetch energy price data
    electricity_prices = fetch_ons_data("mm22", "d7du")
    gas_prices = fetch_ons_data("mm22", "d7dt")

    if electricity_prices is None or gas_prices is None:
        st.warning("Using mock energy price data due to API fetch failure.")
        energy_prices = generate_mock_energy_data()
    else:
        energy_prices = pd.DataFrame({
            'ds': electricity_prices.index,
            'electricity_price': electricity_prices.values,
            'gas_price': gas_prices.values
        }).reset_index(drop=True)

    # Extract Bodycote financial data
    bodycote_data = pd.DataFrame({
        'Year': [2019, 2020, 2021, 2022, 2023],
        'Revenue': [719.7, 598.0, 615.8, 743.6, 802.5],
        'Headline_Operating_Profit': [134.9, 75.3, 94.8, 112.2, 127.6],
        'Headline_Operating_Margin': [18.7, 12.6, 15.4, 15.1, 15.9],
        'Free_Cash_Flow': [133.5, 106.1, 105.0, 84.0, 122.5],
        'ROCE': [20.0, 9.8, 11.8, 13.3, 14.8],
        'Basic_Headline_EPS': [52.1, 27.8, 37.0, 42.7, 48.4],
        'Dividend_Per_Share': [19.3, 19.4, 20.0, 21.3, 22.7],
        'Net_Cash': [21.6, -21.9, 1.9, -33.4, 12.6]
    })

    return energy_prices, bodycote_data

energy_prices, bodycote_data = load_data()

# Forecasting
@st.cache_data
def perform_forecasting(energy_prices, bodycote_data):
    # Prepare Bodycote data for Prophet
    bodycote_prophet_data = bodycote_data.rename(columns={'Year': 'ds', 'Revenue': 'y'})
    bodycote_prophet_data['ds'] = pd.to_datetime(bodycote_prophet_data['ds'].astype(str) + '-12-31')

    # Forecast Bodycote's revenue
    revenue_model = Prophet(yearly_seasonality=True)
    revenue_model.fit(bodycote_prophet_data)
    future_revenue = revenue_model.make_future_dataframe(periods=forecast_period, freq='YE')
    revenue_forecast = revenue_model.predict(future_revenue)

    # Forecast energy prices
    electricity_model = Prophet(yearly_seasonality=True)
    electricity_model.fit(energy_prices[['ds', 'electricity_price']].rename(columns={'electricity_price': 'y'}))
    gas_model = Prophet(yearly_seasonality=True)
    gas_model.fit(energy_prices[['ds', 'gas_price']].rename(columns={'gas_price': 'y'}))

    future_energy = electricity_model.make_future_dataframe(periods=forecast_period, freq='YE')
    electricity_forecast = electricity_model.predict(future_energy)
    gas_forecast = gas_model.predict(future_energy)

    # Combine forecasts
    combined_forecast = pd.merge(revenue_forecast[['ds', 'yhat']], 
                                 electricity_forecast[['ds', 'yhat']], 
                                 on='ds', suffixes=('_revenue', '_electricity'))
    combined_forecast = pd.merge(combined_forecast, 
                                 gas_forecast[['ds', 'yhat']], 
                                 on='ds', suffixes=('', '_gas'))

    # Rename columns to ensure consistent naming
    combined_forecast = combined_forecast.rename(columns={
        'yhat_revenue': 'revenue_forecast',
        'yhat_electricity': 'electricity_price_forecast',
        'yhat': 'gas_price_forecast'
    })

    # Estimate energy costs (using user input percentage of revenue)
    combined_forecast['estimated_energy_cost'] = combined_forecast['revenue_forecast'] * (energy_cost_percentage / 100)

    return combined_forecast

combined_forecast = perform_forecasting(energy_prices, bodycote_data)

# Potential Savings Calculation
def calculate_savings(combined_forecast):
    future_energy_costs = combined_forecast[combined_forecast['ds'] > '2023-12-31']['estimated_energy_cost'].values
    baseline_cost = future_energy_costs[0]

    # Ensure efficiency improvements are in decimal form
    efficiency_improvements_decimal = [improvement / 100 for improvement in efficiency_improvements]

    # Assume efficiency improvements lead to gradual cost reductions
    optimized_costs = [cost * (1 - improvement) for cost, improvement in zip(future_energy_costs, efficiency_improvements_decimal)]

    total_baseline_cost = sum(future_energy_costs)
    total_optimized_cost = sum(optimized_costs)
    total_savings = total_baseline_cost - total_optimized_cost
    percentage_savings = (total_savings / total_baseline_cost) * 100

    # Prepare savings data
    savings_data = pd.DataFrame({
        'Year': combined_forecast[combined_forecast['ds'] > '2023-12-31']['ds'].dt.year,
        'Baseline Cost (£m)': future_energy_costs,
        'Optimized Cost (£m)': optimized_costs,
        'Savings (£m)': future_energy_costs - optimized_costs,
        'Cumulative Improvement (%)': efficiency_improvements[:forecast_period]
    })

    return total_baseline_cost, total_optimized_cost, total_savings, percentage_savings, savings_data

total_baseline_cost, total_optimized_cost, total_savings, percentage_savings, savings_data = calculate_savings(combined_forecast)

# Dashboard Layout
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Detailed Forecasts", "💡 Insights & Savings"])

with tab1:
    st.header("Overview")

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Next Year Revenue Forecast (£m)", value=f"{combined_forecast.loc[combined_forecast['ds'] == combined_forecast['ds'].max(), 'revenue_forecast'].values[0]:.1f}")
    with col2:
        st.metric(label="Next Year Estimated Energy Cost (£m)", value=f"{combined_forecast.loc[combined_forecast['ds'] == combined_forecast['ds'].max(), 'estimated_energy_cost'].values[0]:.1f}")
    with col3:
        st.metric(label="Total Potential Savings (£m)", value=f"{total_savings:.1f}")

    # Revenue and Energy Cost Forecast Plot
    fig_overview = px.line(
        combined_forecast, x='ds', y=['revenue_forecast', 'estimated_energy_cost'],
        labels={'value': 'Amount (£m)', 'ds': 'Year', 'variable': 'Metric'},
        title='Revenue Forecast vs Estimated Energy Cost'
    )
    st.plotly_chart(fig_overview, use_container_width=True)

with tab2:
    st.header("Detailed Forecasts")

    # Revenue Forecast Plot
    st.subheader("Revenue Forecast")
    fig_revenue = px.line(
        combined_forecast, x='ds', y='revenue_forecast',
        labels={'revenue_forecast': 'Revenue (£m)', 'ds': 'Year'},
        title='Revenue Forecast'
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

    # Energy Prices Forecast Plot
    st.subheader("Energy Prices Forecast")
    fig_energy_prices = px.line(
        combined_forecast, x='ds', y=['electricity_price_forecast', 'gas_price_forecast'],
        labels={'value': 'Price', 'ds': 'Year', 'variable': 'Energy Type'},
        title='Electricity and Gas Price Forecast'
    )
    st.plotly_chart(fig_energy_prices, use_container_width=True)

    # Estimated Energy Cost Plot
    st.subheader("Estimated Energy Cost")
    fig_energy_cost = px.bar(
        combined_forecast, x='ds', y='estimated_energy_cost',
        labels={'estimated_energy_cost': 'Estimated Energy Cost (£m)', 'ds': 'Year'},
        title='Estimated Energy Cost Over Time'
    )
    st.plotly_chart(fig_energy_cost, use_container_width=True)

with tab3:
    st.header("Insights & Savings")

    # Savings Overview
    st.subheader("Potential Savings from Energy Optimization")
    st.markdown(f"""
    - **Total baseline energy cost over {forecast_period} years**: £{total_baseline_cost:.1f} million
    - **Total optimized energy cost over {forecast_period} years**: £{total_optimized_cost:.1f} million
    - **Total potential savings**: £{total_savings:.1f} million
    - **Percentage savings**: {percentage_savings:.1f}%
    """)

    # Yearly Savings Breakdown
    st.subheader("Yearly Savings Breakdown")
    st.write(savings_data)

    # Savings Plot
    fig_savings = go.Figure()
    fig_savings.add_trace(go.Bar(
        x=savings_data['Year'], y=savings_data['Baseline Cost (£m)'],
        name='Baseline Cost (£m)', marker_color='indianred'
    ))
    fig_savings.add_trace(go.Bar(
        x=savings_data['Year'], y=savings_data['Optimized Cost (£m)'],
        name='Optimized Cost (£m)', marker_color='lightsalmon'
    ))
    fig_savings.update_layout(
        barmode='group',
        xaxis_title='Year',
        yaxis_title='Cost (£m)',
        title='Baseline vs Optimized Energy Costs'
    )
    st.plotly_chart(fig_savings, use_container_width=True)

    # Conclusion
    st.subheader("Conclusion")
    st.markdown(f"""
    By implementing an annual energy efficiency improvement rate of **{efficiency_improvement_rate}%** over the next **{forecast_period} years**, Bodycote could potentially save **£{total_savings:.1f} million** on energy costs, representing a **{percentage_savings:.1f}%** reduction from the baseline forecast.

    These savings could significantly enhance profitability and provide a competitive advantage in the market.
    """)

# Additional Data and Forecast Components
st.markdown("---")
st.header("📄 Additional Data")

with st.expander("Show Bodycote Financial Data"):
    st.write(bodycote_data)

with st.expander("Show Energy Prices Data"):
    st.write(energy_prices)

with st.expander("Show Combined Forecast Data"):
    st.write(combined_forecast)

# Footer
st.markdown("---")
st.markdown("© 2024 Bodycote Forecast Dashboard")