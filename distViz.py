import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, beta

st.title('Probability Distribution Explorer')

# Select distribution
distribution = st.selectbox('Select a distribution', ['Normal', 'Gamma', 'Beta'])

# Define tick marks for sliders
tick_marks = [-100, -50, -10, -5, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 5, 10, 50, 100]
positive_tick_marks = [0.25, 0.5, 1, 2, 3, 5, 10, 50, 100]  # For parameters that must be positive

# Input parameters side by side with sliders on log scale
if distribution == 'Normal':
    col1, col2 = st.columns(2)
    with col1:
        mean = st.select_slider('Mean (μ)', options=tick_marks, value=0.0)
    with col2:
        std_dev = st.select_slider('Standard Deviation (σ)', options=positive_tick_marks, value=1.0)
    dist = norm(loc=mean, scale=std_dev)
    x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 500)
elif distribution == 'Gamma':
    col1, col2 = st.columns(2)
    with col1:
        shape = st.select_slider('Shape (k)', options=positive_tick_marks, value=2.0)
    with col2:
        scale = st.select_slider('Scale (θ)', options=positive_tick_marks, value=2.0)
    dist = gamma(a=shape, scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Beta':
    col1, col2 = st.columns(2)
    with col1:
        a = st.select_slider('Alpha (α)', options=positive_tick_marks, value=2.0)
    with col2:
        b = st.select_slider('Beta (β)', options=positive_tick_marks, value=5.0)
    dist = beta(a=a, b=b)
    x = np.linspace(0, 1, 500)
else:
    st.write('Please select a distribution.')

# Compute PDF, CDF, Reliability function, and Hazard function
pdf = dist.pdf(x)
cdf = dist.cdf(x)
reliability = 1 - cdf  # Reliability function
# Avoid division by zero for hazard function
with np.errstate(divide='ignore', invalid='ignore'):
    hazard = np.divide(pdf, reliability)
    hazard[reliability == 0] = np.nan  # Handle division by zero

# Define common style parameters
title_fontsize = 18
label_fontsize = 15
line_width = 3.5
background_color = '#f5f5f5'  # Very faint grey

# Define a minimalistic style function
def apply_minimal_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.grid(False)
    ax.set_facecolor(background_color)
    ax.xaxis.set_tick_params(pad=10)  # Spacing between tick labels and axis
    ax.yaxis.set_tick_params(pad=10)  # Spacing between tick labels and axis

col1, col2 = st.columns(2)
with col1:
    # Plot PDF
    fig_pdf, ax_pdf = plt.subplots()
    ax_pdf.plot(x, pdf, linewidth=line_width)
    ax_pdf.set_title('Probability Density', fontsize=title_fontsize)
    ax_pdf.set_xlabel('x', fontsize=label_fontsize)
    ax_pdf.set_ylabel('Probability Density', fontsize=label_fontsize)
    apply_minimal_style(ax_pdf)
    ax_pdf.grid(False)  # Remove grid outlines
    st.pyplot(fig_pdf)

with col2:
    # Plot CDF
    fig_cdf, ax_cdf = plt.subplots()
    ax_cdf.plot(x, cdf, color='orange', linewidth=line_width)
    ax_cdf.set_title('Cumulative Density', fontsize=title_fontsize)
    ax_cdf.set_xlabel('x', fontsize=label_fontsize)
    ax_cdf.set_ylabel('Cumulative Probability', fontsize=label_fontsize)
    apply_minimal_style(ax_cdf)
    ax_cdf.grid(False)  # Remove grid outlines
    st.pyplot(fig_cdf)

# Plot Reliability and Hazard functions side by side
col3, col4 = st.columns(2)
with col3:
    # Plot Reliability function
    fig_rel, ax_rel = plt.subplots()
    ax_rel.plot(x, reliability, color='green', linewidth=line_width)
    ax_rel.set_title('Reliability Function', fontsize=title_fontsize)
    ax_rel.set_xlabel('x', fontsize=label_fontsize)
    ax_rel.set_ylabel('Reliability', fontsize=label_fontsize)
    apply_minimal_style(ax_rel)
    ax_rel.grid(False)  # Remove grid outlines
    st.pyplot(fig_rel)

with col4:
    # Plot Hazard function
    fig_haz, ax_haz = plt.subplots()
    ax_haz.plot(x, hazard, color='red', linewidth=line_width)
    ax_haz.set_title('Hazard Function', fontsize=title_fontsize)
    ax_haz.set_xlabel('x', fontsize=label_fontsize)
    ax_haz.set_ylabel('Hazard Rate', fontsize=label_fontsize)
    apply_minimal_style(ax_haz)
    ax_haz.grid(False)  # Remove grid outlines
    st.pyplot(fig_haz)
