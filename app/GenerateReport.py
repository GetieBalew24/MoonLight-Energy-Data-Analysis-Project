import streamlit as st
import sys
import os
# Get the path of the project directory
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# Add the src directory to the Python path
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)
# Import necessary functions from the eda.py script
from eda import *

# Load the data
st.title("Solar Radiation Data Analysis Dashboard")
file_path = '../data/benin-malanville.csv'

@st.cache
def get_data(filepath):
    return load_data(filepath)

df = get_data(file_path)

# Sidebar for navigation
st.sidebar.header("EDA Analysis Report")
options = st.sidebar.selectbox("Select an EDA Report Analysis", 
                               ["Summary Statistics", 
                                "Data Quality Check", 
                                "Time Series Analysis", 
                                "Correlation Analysis", 
                                "Wind Analysis", 
                                "Temperature Analysis", 
                                "Histograms", 
                                "Box Plots", 
                                "Scatter Plots"])

# Display the selected analysis
if options == "Summary Statistics":
    st.subheader("Summary Statistics")
    st.write(summary_statistics(df))

elif options == "Data Quality Check":
    st.subheader("Data Quality Check")
    missing, negative = data_quality_check(df)
    st.write("Missing Values:")
    st.write(missing)
    st.write("Negative Values in GHI, DNI, DHI:")
    st.write(negative)

elif options == "Time Series Analysis":
    st.subheader("Time Series Analysis")
    time_series = time_series_analysis(df)
    st.pyplot(time_series) 

elif options == "Correlation Analysis":
    st.subheader("Correlation Analysis")
    cor_anl=correlation_analysis(df)
    st.pyplot(cor_anl)

elif options == "Wind Analysis":
    st.subheader("Wind Analysis")
    wind_anl=wind_analysis(df)
    st.pyplot(wind_anl)

elif options == "Temperature Analysis":
    st.subheader("Temperature Analysis")
    temp_anl=temperature_analysis(df)
    st.pyplot(temp_anl)

elif options == "Histograms":
    st.subheader("Histograms")
    hist=histograms(df)
    st.pyplot(hist)

elif options == "Box Plots":
    st.subheader("Box Plots")
    box_plt=box_plots(df)
    st.pyplot(box_plt)

elif options == "Scatter Plots":
    st.subheader("Scatter Plots")
    scatter_plot=scatter_plots(df)
    st.pyplot(scatter_plot)
