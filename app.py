# app.py - COVID-19 Global Data Tracker
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import time
import sys
import traceback
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Global Data Tracker",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🦠 COVID-19 Global Data Tracker")
st.markdown("""
This dashboard provides interactive visualizations and analysis of COVID-19 trends across the globe,
including cases, deaths, recoveries, and vaccination progress.
""")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")

# Create a placeholder for progress updates
progress_placeholder = st.empty()
data_load_state = st.text('Loading data...')
progress_bar = st.progress(0)


@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_data():
    """Load the latest COVID-19 data from Our World in Data."""
    try:
        logger.info("Starting data download process...")
        progress_placeholder.text(
            "Step 1/5: Initiating data download from Our World in Data...")
        progress_bar.progress(10)

        url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        logger.info(f"Downloading data from {url}")

        # Add timeout and show progress as it downloads
        start_time = time.time()

        try:
            df = pd.read_csv(url)
            logger.info(
                f"Data download completed in {time.time() - start_time:.2f} seconds")
            progress_placeholder.text(
                f"Step 2/5: Download complete! Downloaded {df.shape[0]} rows and {df.shape[1]} columns")
            progress_bar.progress(30)
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            progress_placeholder.text(f"Error downloading data: {e}")
            # Try alternate approach
            import requests
            logger.info("Trying alternate download method...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open('covid_data_temp.csv', 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                logger.info("Alternate download complete, reading data...")
                df = pd.read_csv('covid_data_temp.csv')
                logger.info(
                    f"Data loaded with alternate method: {df.shape[0]} rows")
            else:
                raise Exception(
                    f"Failed to download data: HTTP {response.status_code}")

        # Basic data cleaning
        logger.info("Step 3/5: Converting date column to datetime...")
        progress_placeholder.text("Step 3/5: Processing dates...")
        progress_bar.progress(50)

        df['date'] = pd.to_datetime(df['date'])

        # Calculate additional metrics
        logger.info("Step 4/5: Calculating additional metrics...")
        progress_placeholder.text(
            "Step 4/5: Calculating additional metrics (death rates, vaccination percentages)...")
        progress_bar.progress(70)

        # Show which columns are available in the data
        logger.info(f"Available columns: {df.columns.tolist()}")

        # Safe division to avoid divide by zero errors
        # Add check for columns before calculating
        if 'total_cases' in df.columns and 'total_deaths' in df.columns:
            logger.info("Calculating death rate...")
            df['death_rate'] = np.where(df['total_cases'] > 0,
                                      (df['total_deaths'] /
                                       df['total_cases']) * 100,
                                      0)
        else:
            logger.warning(
                "Missing required columns for death rate calculation")

        if 'population' in df.columns and 'people_vaccinated' in df.columns:
            logger.info("Calculating vaccination percentage...")
            df['vax_percentage'] = np.where(df['population'] > 0,
                                          (df['people_vaccinated'] /
                                           df['population']) * 100,
                                          0)
        else:
            logger.warning(
                "Missing required columns for vaccination percentage calculation")

        if 'population' in df.columns and 'people_fully_vaccinated' in df.columns:
            logger.info("Calculating fully vaccinated percentage...")
            df['fully_vax_percentage'] = np.where(df['population'] > 0,
                                               (df['people_fully_vaccinated'] /
                                                df['population']) * 100,
                                               0)
        else:
            logger.warning(
                "Missing required columns for fully vaccinated percentage calculation")

        if 'population' in df.columns and 'total_cases' in df.columns:
            logger.info("Calculating cases per million...")
            df['cases_per_million'] = np.where(df['population'] > 0,
                                             (df['total_cases'] /
                                              (df['population'] / 1000000)),
                                             0)
        else:
            logger.warning(
                "Missing required columns for cases per million calculation")

        if 'population' in df.columns and 'total_deaths' in df.columns:
            logger.info("Calculating deaths per million...")
            df['deaths_per_million'] = np.where(df['population'] > 0,
                                              (df['total_deaths'] /
                                               (df['population'] / 1000000)),
                                              0)
        else:
            logger.warning(
                "Missing required columns for deaths per million calculation")

        logger.info("Step 5/5: Data processing complete!")
        progress_placeholder.text("Step 5/5: Data processing complete!")
        progress_bar.progress(100)

        # Show some stats about the data
        logger.info(f"Data load complete. Shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        logger.error(traceback.format_exc())
        progress_placeholder.text(f"⚠️ Error loading data: {e}")
        # Return a minimal dataframe with error message
        return pd.DataFrame({'error': ['Data loading failed. Please try refreshing the page.']})


# Load the data with extended timeout
try:
    with st.spinner("Loading COVID-19 data... This might take up to 60 seconds."):
        logger.info("Calling load_data function...")
        start_time = time.time()
        df = load_data()
        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.2f} seconds")
        data_load_state.text(f"✅ Data loaded! ({load_time:.1f} seconds)")

        # Check if data loaded successfully
        if 'error' in df.columns:
            st.error("Failed to load COVID data. Please try refreshing the page.")
            st.stop()

except Exception as e:
    logger.error(f"Exception during data loading: {e}")
    logger.error(traceback.format_exc())
    st.error(f"Error loading data: {e}")
    st.stop()

# Clear progress indicators after successful load
progress_bar.empty()

# Show quick data stats
st.sidebar.info(f"""
**Data Stats:**
- Records: {df.shape[0]:,}
- Countries: {df['location'].nunique()}
- Date Range: {df['date'].min().date()} to {df['date'].max().date()}
""")

# Get list of countries and continents for filters
try:
    logger.info("Preparing country filters...")
    countries = sorted(df['location'].unique())
    continents = sorted(df[df['continent'].notna()]['continent'].unique())
except Exception as e:
    logger.error(f"Error preparing filters: {e}")
    st.error(f"Error preparing country filters: {e}")
    countries = []
    continents = []

# Default countries to display
default_countries = ['United States', 'India',
    'Brazil', 'United Kingdom', 'South Africa', 'Kenya']
available_defaults = [c for c in default_countries if c in countries]
if not available_defaults:
    if countries:
        available_defaults = [countries[0]]
    else:
        st.error("No country data available")
        st.stop()

# Sidebar filters
st.sidebar.subheader("Filter Data")

# Date range selector
try:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    # Default to last 180 days but check if we have that much data
    default_start = max_date - pd.Timedelta(days=180)
    if default_start < min_date:
        default_start = min_date

    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [default_start, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Convert to datetime for filtering
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
except Exception as e:
    logger.error(f"Error setting date range: {e}")
    st.sidebar.error(f"Error with date selection: {e}")
    # Set some defaults
    start_date = df['date'].min()
    end_date = df['date'].max()

# Country selection
st.sidebar.subheader("Select Countries")
view_option = st.sidebar.radio(
    "View Options",
    ["Default Countries", "Select by Continent", "Custom Selection"]
)

try:
    if view_option == "Default Countries":
        selected_countries = available_defaults
    elif view_option == "Select by Continent":
        if continents:
            selected_continent = st.sidebar.selectbox(
                "Select Continent", continents)
            continent_countries = sorted(
                df[df['continent'] == selected_continent]['location'].unique())
            selected_countries = st.sidebar.multiselect(
                "Select Countries",
                continent_countries,
                default=continent_countries[:5] if len(
                    continent_countries) > 5 else continent_countries
            )
        else:
            st.sidebar.warning("Continent data not available")
            selected_countries = available_defaults
    else:  # Custom Selection
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            countries,
            default=available_defaults
        )
except Exception as e:
    logger.error(f"Error with country selection: {e}")
    st.sidebar.error(f"Error with country selection: {e}")
    selected_countries = available_defaults

# Ensure at least one country is selected
if not selected_countries:
    st.warning("Please select at least one country to display data.")
    selected_countries = available_defaults

# Filter data based on selections
try:
    logger.info(
        f"Filtering data for {len(selected_countries)} countries and date range {start_date} to {end_date}")
    filtered_df = df[(df['location'].isin(selected_countries)) &
                    (df['date'] >= start_date) &
                    (df['date'] <= end_date)]

    logger.info(f"Filtered data shape: {filtered_df.shape}")
    if filtered_df.empty:
        st.warning(
            "No data available for selected filters. Adjusting selection...")
        # Try to provide some fallback data
        filtered_df = df[df['location'].isin(selected_countries)].head(100)
        if filtered_df.empty:
            st.error("No data available. Please try different selections.")
except Exception as e:
    logger.error(f"Error filtering data: {e}")
    st.error(f"Error filtering data: {e}")
    # Try to recover with simpler filter
    try:
        filtered_df = df.head(100)  # Just take the first 100 rows as fallback
    except:
        st.error("Cannot process data. Please reload the page.")
        st.stop()

# Overview metrics
st.header("📊 Global Overview")

# Create a row with metric cards using columns
col1, col2, col3, col4 = st.columns(4)

# Get latest global data
try:
    logger.info("Getting global metrics...")
    world_data = df[df['location'] == 'World']
    if not world_data.empty:
        latest_global = world_data.sort_values('date').iloc[-1]

        # Display key metrics
        with col1:
            if 'total_cases' in latest_global and pd.notna(latest_global['total_cases']):
                new_cases_delta = latest_global['new_cases'] if 'new_cases' in latest_global and pd.notna(
                    latest_global['new_cases']) else None
                st.metric(
                    "Total Cases",
                    f"{latest_global['total_cases']:,.0f}",
                    f"{new_cases_delta:+,.0f} new" if new_cases_delta is not None else None
                )
            else:
                st.metric("Total Cases", "Data unavailable")

        with col2:
            if 'total_deaths' in latest_global and pd.notna(latest_global['total_deaths']):
                new_deaths_delta = latest_global['new_deaths'] if 'new_deaths' in latest_global and pd.notna(
                    latest_global['new_deaths']) else None
                st.metric(
                    "Total Deaths",
                    f"{latest_global['total_deaths']:,.0f}",
                    f"{new_deaths_delta:+,.0f} new" if new_deaths_delta is not None else None
                )
            else:
                st.metric("Total Deaths", "Data unavailable")

        with col3:
            if 'total_deaths' in latest_global and 'total_cases' in latest_global and \
               pd.notna(latest_global['total_deaths']) and pd.notna(latest_global['total_cases']) and \
               latest_global['total_cases'] > 0:
                death_rate = latest_global['total_deaths'] / \
                    latest_global['total_cases'] * 100
                st.metric("Global Death Rate", f"{death_rate:.2f}%")
            else:
                st.metric("Global Death Rate", "Data unavailable")

        with col4:
            try:
                if 'people_vaccinated' in latest_global and 'population' in latest_global and \
                   pd.notna(latest_global['people_vaccinated']) and pd.notna(latest_global['population']) and \
                   latest_global['population'] > 0:
                    vax_pct = latest_global['people_vaccinated'] / \
                        latest_global['population'] * 100
                    st.metric("Global Vaccination Rate", f"{vax_pct:.2f}%")
                else:
                    st.metric("Global Vaccination Rate", "Data unavailable")
            except Exception as e:
                logger.error(f"Error calculating vaccination rate: {e}")
                st.metric("Global Vaccination Rate", "Error in calculation")
    else:
        logger.warning("No global data available")
        for col in [col1, col2, col3, col4]:
            with col:
                st.metric("Data", "Not available")

except Exception as e:
    logger.error(f"Error displaying global metrics: {e}")
    st.error(f"Error displaying global overview: {e}")

# Main analysis and visualization tabs
try:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Cases Analysis", "Deaths Analysis", "Vaccination Progress", "Country Comparison"])

    with tab1:
        st.header("COVID-19 Cases Analysis")
        logger.info("Rendering cases analysis tab")

        # Total cases over time line chart
        st.subheader("Total Cases Over Time")

        fig_total_cases = go.Figure()

        for country in selected_countries:
            country_data = filtered_df[filtered_df['location'] == country]
            if not country_data.empty:
                fig_total_cases.add_trace(go.Scatter(
                    x=country_data['date'],
                    y=country_data['total_cases'],
                    mode='lines',
                    name=country
                ))

        fig_total_cases.update_layout(
            title="Cumulative COVID-19 Cases",
            xaxis_title="Date",
            yaxis_title="Total Cases",
            height=500
        )

        st.plotly_chart(fig_total_cases, use_container_width=True)

        # Daily new cases
        st.subheader("Daily New Cases")

        # Use 7-day rolling average for smoother visualization
        fig_new_cases = go.Figure()

        for country in selected_countries:
            country_data = filtered_df[filtered_df['location'] == country]
            if not country_data.empty and 'new_cases' in country_data.columns:
                try:
                    # Calculate 7-day rolling average
                    rolling_avg = country_data['new_cases'].rolling(
                        window=7).mean()
                    fig_new_cases.add_trace(go.Scatter(
                        x=country_data['date'],
                        y=rolling_avg,
                        mode='lines',
                        name=f"{country} (7-day avg)"
                    ))
                except Exception as e:
                    logger.error(
                        f"Error calculating rolling average for {country}: {e}")

        fig_new_cases.update_layout(
            title="Daily New COVID-19 Cases (7-day Rolling Average)",
            xaxis_title="Date",
            yaxis_title="New Cases",
            height=500
        )

        st.plotly_chart(fig_new_cases, use_container_width=True)

        # Cases per million
        st.subheader("Cases per Million Population")

        try:
            # Get the latest data for each country
            latest_data = filtered_df.sort_values(
                'date').groupby('location').last().reset_index()

            if 'cases_per_million' in latest_data.columns:
                fig_cases_per_million = px.bar(
                    latest_data.sort_values(
                        'cases_per_million', ascending=False),
                    x='location',
                    y='cases_per_million',
                    color='location',
                    title="COVID-19 Cases per Million by Country (Latest Data)"
                )

                fig_cases_per_million.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Cases per Million",
                    height=500
                )

                st.plotly_chart(fig_cases_per_million,
                                use_container_width=True)
            else:
                st.warning("Cases per million data is not available")
        except Exception as e:
            logger.error(f"Error rendering cases per million chart: {e}")
            st.error(f"Error rendering chart: {e}")

    with tab2:
        st.header("COVID-19 Deaths Analysis")
        logger.info("Rendering deaths analysis tab")

        # Total deaths over time
        st.subheader("Total Deaths Over Time")

        try:
            fig_total_deaths = go.Figure()

            for country in selected_countries:
                country_data = filtered_df[filtered_df['location'] == country]
                if not country_data.empty and 'total_deaths' in country_data.columns:
                    fig_total_deaths.add_trace(go.Scatter(
                        x=country_data['date'],
                        y=country_data['total_deaths'],
                        mode='lines',
                        name=country
                    ))

            fig_total_deaths.update_layout(
                title="Cumulative COVID-19 Deaths",
                xaxis_title="Date",
                yaxis_title="Total Deaths",
                height=500
            )

            st.plotly_chart(fig_total_deaths, use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering total deaths chart: {e}")
            st.error(f"Error rendering chart: {e}")

        # Daily new deaths
        st.subheader("Daily New Deaths")

        try:
            fig_new_deaths = go.Figure()

            for country in selected_countries:
                country_data = filtered_df[filtered_df['location'] == country]
                if not country_data.empty and 'new_deaths' in country_data.columns:
                    # Calculate 7-day rolling average
                    rolling_avg = country_data['new_deaths'].rolling(
                        window=7).mean()
                    fig_new_deaths.add_trace(go.Scatter(
                        x=country_data['date'],
                        y=rolling_avg,
                        mode='lines',
                        name=f"{country} (7-day avg)"
                    ))

            fig_new_deaths.update_layout(
                title="Daily New COVID-19 Deaths (7-day Rolling Average)",
                xaxis_title="Date",
                yaxis_title="New Deaths",
                height=500
            )

            st.plotly_chart(fig_new_deaths, use_container_width=True)
        except Exception as e:
            logger.error(f"Error rendering new deaths chart: {e}")
            st.error(f"Error rendering chart: {e}")

        # Death rate
        st.subheader("Death Rate Analysis")

        try:
            if 'death_rate' in latest_data.columns:
                fig_death_rate = px.bar(
                    latest_data.sort_values('death_rate', ascending=False),
                    x='location',
                    y='death_rate',
                    color='location',
                    title="COVID-19 Death Rate by Country (Latest Data)"
                )

                fig_death_rate.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Death Rate (%)",
                    height=500
                )

                st.plotly_chart(fig_death_rate, use_container_width=True)
            else:
                st.warning("Death rate data is not available")
        except Exception as e:
            logger.error(f"Error rendering death rate chart: {e}")
            st.error(f"Error rendering chart: {e}")

        # Deaths per million
        st.subheader("Deaths per Million Population")

        try:
            if 'deaths_per_million' in latest_data.columns:
                fig_deaths_per_million = px.bar(
                    latest_data.sort_values(
                        'deaths_per_million', ascending=False),
                    x='location',
                    y='deaths_per_million',
                    color='location',
                    title="COVID-19 Deaths per Million by Country (Latest Data)"
                )

                fig_deaths_per_million.update_layout(
                    xaxis_title="Country",
                    yaxis_title="Deaths per Million",
                    height=500
                )

                st.plotly_chart(fig_deaths_per_million,
                                use_container_width=True)
            else:
                st.warning("Deaths per million data is not available")
        except Exception as e:
            logger.error(f"Error rendering deaths per million chart: {e}")
            st.error(f"Error rendering chart: {e}")

    with tab3:
        st.header("Vaccination Progress")
        logger.info("Rendering vaccination progress tab")

        # Filter for rows with vaccination data
        try:
            vax_data = filtered_df.dropna(subset=['total_vaccinations'])

            if vax_data.empty:
                st.warning(
                    "No vaccination data available for the selected countries and time period.")
            else:
                # Total vaccinations over time
                st.subheader("Total Vaccinations Over Time")

                fig_total_vax = go.Figure()

                for country in selected_countries:
                    country_data = vax_data[vax_data['location'] == country]
                    if not country_data.empty:
                        fig_total_vax.add_trace(go.Scatter(
                            x=country_data['date'],
                            y=country_data['total_vaccinations'],
                            mode='lines',
                            name=country
                        ))

                fig_total_vax.update_layout(
                    title="Cumulative COVID-19 Vaccinations",
                    xaxis_title="Date",
                    yaxis_title="Total Vaccinations",
                    height=500
                )

                st.plotly_chart(fig_total_vax, use_container_width=True)

                # Vaccination percentage over time
                st.subheader(
                    "Vaccination Coverage Over Time (% of Population)")

                fig_vax_pct = go.Figure()

                for country in selected_countries:
                    country_data = vax_data[vax_data['location'] == country]
                    if not country_data.empty and 'people_vaccinated' in country_data.columns and 'population' in country_data.columns:
                        # Calculate percentage of population vaccinated
                        country_data['vax_percentage'] = (
                            country_data['people_vaccinated'] / country_data['population']) * 100
                        fig_vax_pct.add_trace(go.Scatter(
                            x=country_data['date'],
                            y=country_data['vax_percentage'],
                            mode='lines',
                            name=country
                        ))

                fig_vax_pct.update_layout(
                    title="Percentage of Population Vaccinated",
                    xaxis_title="Date",
                    yaxis_title="Population Vaccinated (%)",
                    height=500,
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig_vax_pct, use_container_width=True)

                # Fully vs. Partially vaccinated (latest data)
                st.subheader(
                    "Vaccination Status: Fully vs. Partially Vaccinated")

                # Get latest vaccination data
                latest_vax = vax_data.sort_values('date').groupby(
                    'location').last().reset_index()

                # Prepare data for stacked bar chart
                vax_status_data = []

                for country in latest_vax['location'].unique():
                    country_data = latest_vax[latest_vax['location'] == country]

                    if not country_data.empty and pd.notna(country_data['people_vaccinated'].values[0]) and pd.notna(country_data['people_fully_vaccinated'].values[0]):
                        fully_vax = country_data['people_fully_vaccinated'].values[0]
                        partially_vax = country_data['people_vaccinated'].values[0] - fully_vax
                        population = country_data['population'].values[0]

                        fully_vax_pct = (fully_vax / population) * \
                                         100 if population > 0 else 0
                        partially_vax_pct = (
                            partially_vax / population) * 100 if population > 0 else 0

                        vax_status_data.append({
                            'country': country,
                            'fully_vaccinated': fully_vax_pct,
                            'partially_vaccinated': partially_vax_pct
                        })

                if vax_status_data:
                    vax_status_df = pd.DataFrame(vax_status_data)

                    # Create stacked bar chart
                    fig_vax_status = go.Figure()

                    fig_vax_status.add_trace(go.Bar(
                        x=vax_status_df['country'],
                        y=vax_status_df['fully_vaccinated'],
                        name='Fully Vaccinated',
                        marker_color='darkgreen'
                    ))

                    fig_vax_status.add_trace(go.Bar(
                        x=vax_status_df['country'],
                        y=vax_status_df['partially_vaccinated'],
                        name='Partially Vaccinated',
                        marker_color='lightgreen'
                    ))

                    fig_vax_status.update_layout(
                        title="Vaccination Status by Country (Latest Data)",
                        xaxis_title="Country",
                        yaxis_title="Population (%)",
                        barmode='stack',
                        height=500,
                        yaxis=dict(range=[0, 100])
                    )

                    st.plotly_chart(fig_vax_status, use_container_width=True)
                else:
                    st.info(
                        "Insufficient vaccination data for the selected countries.")
        except Exception as e:
            logger.error(f"Error rendering vaccination charts: {e}")
            st.error(f"Error rendering vaccination data: {e}")

    with tab4:
        st.header("Country Comparison Dashboard")
        logger.info("Rendering country comparison tab")

        # Select metrics for comparison
        st.subheader("Compare Countries by Key Metrics")

        try:
            # Get latest data for comparison
            latest_data = filtered_df.sort_values(
                'date').groupby('location').last().reset_index()

            # Choose metrics for scatter plot
            col1, col2 = st.columns(2)

            # Get available metrics
            numeric_columns = [col for col in latest_data.columns if
                          pd.api.types.is_numeric_dtype(latest_data[col]) and
                          col not in ['iso_code', 'date'] and
                          not col.startswith('Unnamed')]

            metric_options = [
                "total_cases_per_million",
                "total_deaths_per_million",
                "vax_percentage",
                "death_rate",
                "gdp_per_capita",
                "hospital_beds_per_thousand"
            ]

            # Filter to only available metrics
            available_metrics = [
                m for m in metric_options if m in numeric_columns]

            if len(available_metrics) < 2:
                st.warning(
                    "Insufficient metrics available for comparison. Using available numeric columns.")
                available_metrics = numeric_columns[:min(
                    5, len(numeric_columns))]

            with col1:
                x_metric = st.selectbox(
                    "X-Axis Metric",
                    available_metrics,
                    index=0,
                    format_func=lambda x: x.replace("_", " ").title()
                )

            with col2:
                y_metric = st.selectbox(
                    "Y-Axis Metric",
                    available_metrics,
                    index=min(1, len(available_metrics)-1),
                    format_func=lambda x: x.replace("_", " ").title()
                )

            # The problematic section with corrected indentation
            # Create scatter plot
            if x_metric in latest_data.columns and y_metric in latest_data.columns:
                fig_scatter = px.scatter(
                    latest_data,
                    x=x_metric,
                    y=y_metric,
                    color="location",
                    size="population" if "population" in latest_data.columns else None,
                    hover_name="location",
                    log_x=st.checkbox("Log Scale for X-Axis", value=False),
                    log_y=st.checkbox("Log Scale for Y-Axis", value=False),
                    size_max=60
                )

                fig_scatter.update_layout(
                    title=f"Country Comparison: {x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}",
                    xaxis_title=x_metric.replace("_", " ").title(),
                    yaxis_title=y_metric.replace("_", " ").title(),
                    height=600
                )

                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Selected metrics not available for comparison.")

            # Correlation heatmap for selected countries
            st.subheader("Correlation Between COVID-19 Metrics")

            # Select relevant numeric columns for correlation
            numeric_cols = [
                'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                'total_cases_per_million', 'total_deaths_per_million',
                'reproduction_rate', 'icu_patients', 'hosp_patients',
                'total_tests', 'new_tests', 'total_vaccinations',
                'people_vaccinated', 'people_fully_vaccinated'
            ]

            # Filter to only columns that exist in the data
            available_cols = [
                col for col in numeric_cols if col in filtered_df.columns]

            if available_cols:
                # Create correlation matrix
                correlation_df = filtered_df[available_cols].corr()
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    correlation_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix of COVID-19 Metrics"
                )
                
                fig_heatmap.update_layout(height=600)
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Insufficient data for correlation analysis.")
        except Exception as e:
            logger.error(f"Error in country comparison tab: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Error generating country comparison: {e}")
except Exception as e:
    logger.error(f"Error creating visualization tabs: {e}")
    logger.error(traceback.format_exc())
    st.error(f"Error creating visualization tabs: {e}")

# World Map visualization
try:
   st.header("🗺️ Global COVID-19 Map")
   logger.info("Rendering world map")
   
   # Get the latest data for each country
   latest_global_data = df.sort_values('date').groupby('location').last().reset_index()
   
   # Check if we have the required data
   if 'iso_code' not in latest_global_data.columns:
       st.warning("Cannot create map visualization: Missing country codes (iso_code)")
   else:
       # Map metric selection
       available_map_metrics = [m for m in [
           "total_cases_per_million",
           "total_deaths_per_million",
           "vax_percentage",
           "death_rate"
       ] if m in latest_global_data.columns]
       
       if not available_map_metrics:
           st.warning("No suitable metrics available for map visualization")
       else:
           map_metric = st.selectbox(
               "Select Metric to Visualize on Map",
               available_map_metrics,
               format_func=lambda x: {
                   "total_cases_per_million": "Cases per Million",
                   "total_deaths_per_million": "Deaths per Million",
                   "vax_percentage": "Vaccination Rate (%)",
                   "death_rate": "Death Rate (%)"
               }.get(x, x.replace("_", " ").title())
           )
           
           # Create choropleth map
           try:
               if map_metric == "vax_percentage" and 'vax_percentage' not in latest_global_data.columns:
                   # Calculate vaccination percentage for mapping
                   logger.info("Calculating vaccination percentage for map...")
                   latest_global_data['vax_percentage'] = latest_global_data.apply(
                       lambda row: (row['people_vaccinated'] / row['population']) * 100 
                       if pd.notna(row.get('people_vaccinated', None)) and pd.notna(row.get('population', None)) and row['population'] > 0 
                       else np.nan, 
                       axis=1
                   )
                   
               # Filter out rows with missing values
               map_data = latest_global_data.dropna(subset=[map_metric, 'iso_code'])
               
               if map_data.empty:
                   st.warning(f"No data available for {map_metric} on the map")
               else:
                   fig_map = px.choropleth(
                       map_data,
                       locations="iso_code",
                       color=map_metric,
                       hover_name="location",
                       projection="natural earth",
                       color_continuous_scale="Viridis" if map_metric == "vax_percentage" else "Reds",
                       title=f"Global COVID-19 Map: {map_metric.replace('_', ' ').title()}",
                       height=600
                   )
                   
                   fig_map.update_layout(
                       coloraxis_colorbar=dict(
                           title=map_metric.replace("_", " ").title()
                       )
                   )
                   
                   st.plotly_chart(fig_map, use_container_width=True)
           except Exception as e:
               logger.error(f"Error creating map visualization: {e}")
               logger.error(traceback.format_exc())
               st.error(f"Error creating map visualization: {e}")
               st.info("Some countries might lack the required data for mapping.")
except Exception as e:
   logger.error(f"Error in map section: {e}")
   logger.error(traceback.format_exc())
   st.error(f"Error generating map visualization: {e}")

# Key Insights Section
try:
   st.header("📝 Key Insights")
   logger.info("Generating insights")
   
   # Generate insights based on the data
   try:
       # Calculate statistics for insights
       latest_data = filtered_df.sort_values('date').groupby('location').last().reset_index()
       
       insights_text = []
       
       # Find country with highest case rate
       if 'cases_per_million' in latest_data.columns:
           highest_case_idx = latest_data['cases_per_million'].idxmax()
           highest_case_country = latest_data.loc[highest_case_idx]['location']
           highest_case_rate = latest_data['cases_per_million'].max()
           
           insights_text.append(f"**Highest Case Rate:** {highest_case_country} has the highest number of cases per million ({highest_case_rate:,.2f}).")
       
       # Find country with highest death rate
       if 'death_rate' in latest_data.columns:
           highest_death_idx = latest_data['death_rate'].idxmax()
           highest_death_country = latest_data.loc[highest_death_idx]['location']
           highest_death_rate = latest_data['death_rate'].max()
           
           insights_text.append(f"**Highest Death Rate:** {highest_death_country} has the highest death rate among selected countries ({highest_death_rate:.2f}%).")
       
       # Find country with highest vaccination rate (if data available)
       vax_data = latest_data.dropna(subset=['vax_percentage']) if 'vax_percentage' in latest_data.columns else pd.DataFrame()
       
       if not vax_data.empty:
           highest_vax_idx = vax_data['vax_percentage'].idxmax()
           highest_vax_country = vax_data.loc[highest_vax_idx]['location']
           highest_vax_rate = vax_data['vax_percentage'].max()
           
           lowest_vax_idx = vax_data['vax_percentage'].idxmin()
           lowest_vax_country = vax_data.loc[lowest_vax_idx]['location']
           lowest_vax_rate = vax_data['vax_percentage'].min()
           
           insights_text.append(f"**Highest Vaccination Rate:** {highest_vax_country} leads with {highest_vax_rate:.2f}% of population vaccinated.")
           insights_text.append(f"**Lowest Vaccination Rate:** {lowest_vax_country} has the lowest rate at {lowest_vax_rate:.2f}%.")
       
       # Display insights
       col1, col2 = st.columns(2)
       
       with col1:
           st.markdown("### Case and Death Insights")
           case_insights = [insight for insight in insights_text if "Case Rate" in insight or "Death Rate" in insight]
           if case_insights:
               for insight in case_insights:
                   st.markdown(f"- {insight}")
           else:
               st.markdown("Insufficient data for case and death insights.")
       
       with col2:
           st.markdown("### Vaccination Insights")
           vax_insights = [insight for insight in insights_text if "Vaccination Rate" in insight]
           if vax_insights:
               for insight in vax_insights:
                   st.markdown(f"- {insight}")
           else:
               st.markdown("Insufficient vaccination data available for the selected countries.")
       
       # Find country with fastest growth in cases (last 30 days)
       try:
           # Get data for the last 30 days
           cutoff_date = filtered_df['date'].max() - pd.Timedelta(days=30)
           recent_df = filtered_df[filtered_df['date'] >= cutoff_date]
           
           # Calculate growth rate for each country
           growth_data = []
           
           for country in selected_countries:
               country_data = recent_df[recent_df['location'] == country]
               if len(country_data) > 15 and 'total_cases' in country_data.columns:  # Ensure we have enough data points
                   start_cases = country_data.sort_values('date').iloc[0]['total_cases']
                   end_cases = country_data.sort_values('date').iloc[-1]['total_cases']
                   
                   if pd.notna(start_cases) and pd.notna(end_cases) and start_cases > 0:
                       growth_rate = ((end_cases - start_cases) / start_cases) * 100
                       growth_data.append({
                           'country': country,
                           'growth_rate': growth_rate
                       })
           
           if growth_data:
               growth_df = pd.DataFrame(growth_data)
               fastest_idx = growth_df['growth_rate'].idxmax()
               fastest_growth_country = growth_df.loc[fastest_idx]['country']
               fastest_growth_rate = growth_df['growth_rate'].max()
               
               st.markdown(f"""
               ### Growth Trend Insight
               
               - **Fastest Case Growth:** {fastest_growth_country} had the fastest growth in cases over the last 30 days ({fastest_growth_rate:.2f}%).
               """)
       except Exception as e:
           logger.error(f"Could not calculate recent growth trends: {e}")
           logger.error(traceback.format_exc())
   
   except Exception as e:
       logger.error(f"Error generating insights: {e}")
       logger.error(traceback.format_exc())
       st.error(f"Error generating insights: {e}")
       st.info("Some data might be missing for proper insight generation.")
except Exception as e:
   logger.error(f"Error in insights section: {e}")
   logger.error(traceback.format_exc())
   st.error(f"Error generating insights section: {e}")

# Footer with data source information
st.markdown("""
---
### Data Sources
- COVID-19 data is sourced from [Our World in Data](https://ourworldindata.org/coronavirus)
- Last updated: """ + df['date'].max().strftime('%B %d, %Y'))

# Add download button for the filtered data
try:
   st.sidebar.header("Download Data")
   csv = filtered_df.to_csv(index=False).encode('utf-8')
   st.sidebar.download_button(
       label="Download CSV",
       data=csv,
       file_name=f"covid19_data_{datetime.now().strftime('%Y%m%d')}.csv",
       mime="text/csv"
   )
except Exception as e:
   logger.error(f"Error creating download button: {e}")
   st.sidebar.warning("Download option unavailable")

# Add system info
st.sidebar.markdown("---")
st.sidebar.subheader("Debug Info")
st.sidebar.info(f"""
- App version: 1.0.1
- Python: {sys.version.split()[0]}
- Pandas: {pd.__version__}
- Streamlit: {st.__version__}
- Data load time: {load_time:.2f}s
- Records: {df.shape[0]:,}
- Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB
""")

# Add a button to refresh data
if st.sidebar.button("Force Refresh Data"):
   st.experimental_rerun()

# Additional debug information expandable section
with st.sidebar.expander("Advanced Debug Info"):
   st.write("Data Column Types:")
   col_types = {}
   for col in df.columns:
       col_types[col] = str(df[col].dtype)
   
   st.json(col_types)
   
   st.write("Missing Values Summary:")
   missing_counts = df[df.columns].isnull().sum().to_dict()
   missing_pct = {col: f"{count / len(df) * 100:.1f}%" for col, count in missing_counts.items() if count > 0}
   
   if missing_pct:
       st.json(missing_pct)
   else:
       st.write("No missing values")
   
   st.write("Log Level Control:")
   log_level = st.selectbox(
       "Set Log Level",
       ["DEBUG", "INFO", "WARNING", "ERROR"],
       index=1
   )
   
   if st.button("Apply Log Level"):
       logger.setLevel(getattr(logging, log_level))
       st.success(f"Log level set to {log_level}")
       
# Show app status (hidden by default)
st.markdown("""
<style>
   #app-status {
       visibility: hidden;
       position: fixed;
       bottom: 10px;
       right: 10px;
       z-index: 999;
       background: rgba(255, 255, 255, 0.7);
       padding: 5px 10px;
       border-radius: 5px;
       font-size: 12px;
   }
   .stApp:hover #app-status {
       visibility: visible;
   }
</style>
<div id="app-status">App running normally</div>
""", unsafe_allow_html=True)
