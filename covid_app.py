import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data
@st.cache_data
def load_data():
    url = "COVID_FINAL_DATA.xlsx"  # Replace with your actual data URL
    df = pd.read_excel(url)
    
    # Ensure numeric columns are treated as numeric
    numeric_columns = ['Cumulative', 'Expired', 'Discharged']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
    
    # Standardize date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%d-%b-%Y')  # Coerce invalid formats to NaT
    return df

df = load_data()

# Title of the app
st.title("COVID-19 Data Visualization for Pakistan")

# Sidebar for user input
st.sidebar.header('User Input Features')
selected_region = st.sidebar.selectbox('Select Region', df['Region'].unique())
date_range = st.sidebar.date_input('Select Date Range', [df['Date'].min(), df['Date'].max()])

# Convert date_range to datetime64[ns]
date_range = [pd.to_datetime(date) for date in date_range]

# Filter data based on user input
filtered_data = df[(df['Region'] == selected_region) & (df['Date'].between(date_range[0], date_range[1]))]

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_data)

# Summary Statistics
st.subheader('Summary Statistics')
total_cases = df['Cumulative'].sum()
total_deaths = df['Expired'].sum()
total_recoveries = df['Discharged'].sum()
active_cases = total_cases - total_deaths - total_recoveries

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", total_cases)
col2.metric("Total Deaths", total_deaths)
col3.metric("Total Recoveries", total_recoveries)
col4.metric("Active Cases", active_cases)

# Display summary in HTML using f-string
st.markdown(f"""
    <div style="background-color:#f0f0f5;padding:10px;border-radius:10px;">
        <h3>COVID-19 Summary</h3>
        <p><strong>Total Cases:</strong> {total_cases}</p>
        <p><strong>Total Deaths:</strong> {total_deaths}</p>
        <p><strong>Total Recoveries:</strong> {total_recoveries}</p>
        <p><strong>Active Cases:</strong> {active_cases}</p>
    </div>
""", unsafe_allow_html=True)

# Plotting the evolution of COVID-19 results
st.subheader('Evolution of COVID-19 Results')
evolution = df.groupby('Date').sum(numeric_only=True)[['Cumulative', 'Expired', 'Discharged']]
evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100
evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

fig = go.Figure()
fig.add_trace(go.Scatter(x=evolution.index, y=evolution['Cumulative'], mode='lines', name='Cumulative'))
fig.add_trace(go.Scatter(x=evolution.index, y=evolution['Expired'], mode='lines', name='Expired'))
fig.add_trace(go.Scatter(x=evolution.index, y=evolution['Discharged'], mode='lines', name='Discharged'))
fig.update_layout(title='Evolution of COVID-19 Results', xaxis_title='Date', yaxis_title='Number of Patients')
st.plotly_chart(fig)

# Heatmap for Regional Data
st.subheader('Heatmap of COVID-19 Cases by Region')
heatmap_data = df.pivot_table(index='Date', columns='Region', values='Cumulative', aggfunc='sum')
fig = px.imshow(heatmap_data, labels=dict(x="Region", y="Date", color="Cases"),
                title="Heatmap of COVID-19 Cases by Region")
st.plotly_chart(fig)

# Daily New Cases and Deaths
st.subheader('Daily New Cases and Deaths')
df['New Cases'] = df.groupby('Region')['Cumulative'].diff().fillna(0)
df['New Deaths'] = df.groupby('Region')['Expired'].diff().fillna(0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['New Cases'], mode='lines', name='New Cases'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['New Deaths'], mode='lines', name='New Deaths'))
fig.update_layout(title='Daily New Cases and Deaths', xaxis_title='Date', yaxis_title='Count')
st.plotly_chart(fig)

# Recovery vs Death Rate Over Time
st.subheader('Recovery vs Death Rate Over Time')
df['Recovery Rate'] = (df['Discharged'] / df['Cumulative']) * 100
df['Death Rate'] = (df['Expired'] / df['Cumulative']) * 100

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Recovery Rate'], mode='lines', name='Recovery Rate'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Death Rate'], mode='lines', name='Death Rate'))
fig.update_layout(title='Recovery vs Death Rate Over Time', xaxis_title='Date', yaxis_title='Rate (%)')
st.plotly_chart(fig)

# Regional Comparison (Bar Chart Race)
st.subheader('Regional Comparison Over Time')
regional_data = df.groupby(['Date', 'Region']).sum(numeric_only=True).reset_index()
fig = px.bar(regional_data, x='Cumulative', y='Region', animation_frame='Date', 
             orientation='h', title='Regional Comparison Over Time')
st.plotly_chart(fig)

# Map Visualization (if latitude and longitude data is available)
st.subheader('COVID-19 Cases by Region on Map')
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    fig = px.scatter_geo(df, lat='Latitude', lon='Longitude', color='Cumulative',
                         hover_name='Region', size='Cumulative', projection='natural earth',
                         title='COVID-19 Cases by Region on Map')
    st.plotly_chart(fig)
else:
    st.warning("Latitude and Longitude data is required for map visualization.")

# Predictive Analytics
st.subheader('COVID-19 Case Prediction')
if st.checkbox('Show Predictive Analytics'):
    # Prepare data for prediction
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Cumulative'].values

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    future_days = st.slider('Predict for how many future days?', 1, 30, 7)
    future_X = np.array(range(len(df), len(df) + future_days)).reshape(-1, 1)
    predictions = model.predict(future_X)

    # Plot predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=y, mode='lines', name='Actual Cases'))
    fig.add_trace(go.Scatter(x=pd.date_range(df['Date'].iloc[-1], periods=future_days), 
                  y=predictions, mode='lines', name='Predicted Cases'))
    fig.update_layout(title='COVID-19 Case Prediction', xaxis_title='Date', yaxis_title='Cases')
    st.plotly_chart(fig)

# Downloadable Reports
st.sidebar.subheader('Download Reports')
if st.sidebar.button('Download Filtered Data as CSV'):
    filtered_data.to_csv('filtered_data.csv', index=False)
    st.sidebar.success('File downloaded successfully!')

# Customizable Themes
st.sidebar.subheader('Theme Customization')
theme = st.sidebar.selectbox('Select Theme', ['Light', 'Dark'])
if theme == 'Dark':
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
