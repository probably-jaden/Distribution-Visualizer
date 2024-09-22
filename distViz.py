import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    norm, gamma, beta, expon, cauchy, chi2, lognorm, erlang, pareto, fisk,
    ncx2, nct, kstwo, powerlaw, weibull_min
)

st.title('Probability Distribution Explorer')
st.markdown("---")

# Define tick marks for sliders
tick_marks = [-100, -50, -10, -5, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 5, 10, 50, 100]
positive_tick_marks = [0.25, 0.5, 1, 2, 3, 5, 10, 50, 100]  # For parameters that must be positive

def safe_stats(dist):
    try:
        mean = dist.mean()
        if not np.isfinite(mean):
            mean = 'undefined'
    except:
        mean = 'undefined'
    try:
        std_dev = dist.std()
        if not np.isfinite(std_dev):
            std_dev = 'undefined'
    except:
        std_dev = 'undefined'
    try:
        skewness = dist.stats(moments='s')
        if not np.isfinite(skewness):
            skewness = 'undefined'
    except:
        skewness = 'undefined'
    return mean, std_dev, skewness

# Select input method
input_method = st.radio('Select input method for parameters', ['Sliders', 'Numeric Inputs'])

# Select distribution
distribution = st.selectbox('Select a distribution', [
    'Normal', 'Gamma', 'Beta', 'Exponential', 'Weibull', 'Cauchy', 'Chi-Square',
    'Log-Normal', 'Erlang', 'Pareto', 'Log-Logistic', 'Noncentral Chi-Square',
    'Noncentral t', 'Kolmogorov-Smirnov', 'Power'
])

# Input parameters and compute statistics
if distribution == 'Normal':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            mean_input = st.select_slider('Mean (μ)', options=tick_marks, value=0.0)
        with col2:
            std_dev_input = st.select_slider('Standard Deviation (σ)', options=positive_tick_marks, value=1.0)
    else:
        with col1:
            mean_input = st.number_input('Mean (μ)', value=0.0)
        with col2:
            std_dev_input = st.number_input('Standard Deviation (σ)', min_value=0.0, value=1.0)
    dist = norm(loc=mean_input, scale=std_dev_input)
    x = np.linspace(mean_input - 4*std_dev_input, mean_input + 4*std_dev_input, 500)
elif distribution == 'Gamma':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            shape = st.select_slider('Shape (k)', options=positive_tick_marks, value=2.0)
        with col2:
            scale = st.select_slider('Scale (θ)', options=positive_tick_marks, value=2.0)
    else:
        with col1:
            shape = st.number_input('Shape (k)', min_value=0.0, value=2.0)
        with col2:
            scale = st.number_input('Scale (θ)', min_value=0.0, value=2.0)
    dist = gamma(a=shape, scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
# ... (rest of the distribution handling code remains the same)
elif distribution == 'Cauchy':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            loc = st.select_slider('Location (x₀)', options=tick_marks, value=0.0)
        with col2:
            scale = st.select_slider('Scale (γ)', options=positive_tick_marks, value=1.0)
    else:
        with col1:
            loc = st.number_input('Location (x₀)', value=0.0)
        with col2:
            scale = st.number_input('Scale (γ)', min_value=0.0, value=1.0)
    dist = cauchy(loc=loc, scale=scale)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
# ... (rest of your code remains the same)

st.markdown("---")

# Compute statistics
mean, std_dev, skewness = safe_stats(dist)

# Check if the statistics are finite numbers
if np.isfinite(mean):
    disp_mean = round(mean, 2)
else:
    disp_mean = 'undefined'

if np.isfinite(std_dev):
    disp_std_dev = round(std_dev, 2)
else:
    disp_std_dev = 'undefined'

if np.isfinite(skewness):
    disp_skewness = round(skewness, 2)
else:
    disp_skewness = 'undefined'

# Display statistics side by side with bars above and below
tab1, tab2 = st.tabs(["PDFs", "Likelihood"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Mean</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>{disp_mean}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Standard Deviation</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>{disp_std_dev}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Skewness</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'>{disp_skewness}</div>", unsafe_allow_html=True)
    st.markdown("---")


    # ... (rest of your plotting code remains the same)
