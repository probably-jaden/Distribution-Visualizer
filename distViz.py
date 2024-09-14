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
    except:
        mean = 'undefined'
    try:
        std_dev = dist.std()
    except:
        std_dev = 'undefined'
    try:
        skewness = dist.stats(moments='s')
    except:
        skewness = 'undefined'
    return mean, std_dev, skewness

# Select distribution
distribution = st.selectbox('Select a distribution', [
    'Normal', 'Gamma', 'Beta', 'Exponential', 'Weibull', 'Cauchy', 'Chi-Square',
    'Log-Normal', 'Erlang', 'Pareto', 'Log-Logistic', 'Noncentral Chi-Square',
    'Noncentral t', 'Kolmogorov-Smirnov', 'Power'
])

# Input parameters and compute statistics
if distribution == 'Normal':
    col1, col2 = st.columns(2)
    with col1:
        mean_input = st.select_slider('Mean (μ)', options=tick_marks, value=0.0)
    with col2:
        std_dev_input = st.select_slider('Standard Deviation (σ)', options=positive_tick_marks, value=1.0)
    dist = norm(loc=mean_input, scale=std_dev_input)
    x = np.linspace(mean_input - 4*std_dev_input, mean_input + 4*std_dev_input, 500)
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
elif distribution == 'Exponential':
    scale = st.select_slider('Scale (λ)', options=positive_tick_marks, value=1.0)
    dist = expon(scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Cauchy':
    col1, col2 = st.columns(2)
    with col1:
        loc = st.select_slider('Location (x₀)', options=tick_marks, value=0.0)
    with col2:
        scale = st.select_slider('Scale (γ)', options=positive_tick_marks, value=1.0)
    dist = cauchy(loc=loc, scale=scale)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
elif distribution == 'Chi-Square':
    df = st.select_slider('Degrees of Freedom (k)', options=positive_tick_marks, value=2.0)
    dist = chi2(df=df)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Log-Normal':
    col1, col2 = st.columns(2)
    with col1:
        mu = st.select_slider('Mean of log (μ)', options=tick_marks, value=0.0)
    with col2:
        sigma = st.select_slider('Standard Deviation of log (σ)', options=positive_tick_marks, value=1.0)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Erlang':
    col1, col2 = st.columns(2)
    with col1:
        k = st.select_slider('Shape (k)', options=positive_tick_marks, value=2.0)
    with col2:
        scale = st.select_slider('Scale (θ)', options=positive_tick_marks, value=1.0)
    dist = erlang(a=int(k), scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Pareto':
    b = st.select_slider('Shape (b)', options=positive_tick_marks, value=2.0)
    dist = pareto(b=b)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
elif distribution == 'Log-Logistic':
    c = st.select_slider('Shape (c)', options=positive_tick_marks, value=2.0)
    dist = fisk(c=c)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Noncentral Chi-Square':
    col1, col2 = st.columns(2)
    with col1:
        df = st.select_slider('Degrees of Freedom (k)', options=positive_tick_marks, value=2.0)
    with col2:
        nc = st.select_slider('Noncentrality (λ)', options=positive_tick_marks, value=1.0)
    dist = ncx2(df=df, nc=nc)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Noncentral t':
    col1, col2 = st.columns(2)
    with col1:
        df = st.select_slider('Degrees of Freedom (k)', options=positive_tick_marks, value=10.0)
    with col2:
        nc = st.select_slider('Noncentrality (δ)', options=tick_marks, value=1.0)
    dist = nct(df=df, nc=nc)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
elif distribution == 'Kolmogorov-Smirnov':
    n = st.select_slider('Sample Size (n)', options=positive_tick_marks, value=10.0)
    dist = kstwo(n=int(n))
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Power':
    a = st.select_slider('Shape (a)', options=positive_tick_marks, value=2.0)
    dist = powerlaw(a=a)
    x = np.linspace(0, 1, 500)
elif distribution == 'Weibull':
    col1, col2 = st.columns(2)
    with col1:
        c = st.slider('Shape (c)', min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    with col2:
        scale = st.slider('Scale (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    dist = weibull_min(c=c, scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
else:
    st.write('Please select a distribution.')

# Compute statistics
mean, std_dev, skewness = safe_stats(dist)
disp_mean = round(mean, 2) if mean != 'undefined' else mean
disp_std_dev = round(std_dev, 2) if std_dev != 'undefined' else std_dev
disp_skewness = round(skewness, 2) if skewness != 'undefined' else skewness

# Display statistics side by side with bars above and below
st.markdown("---")
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
background_color = '#fafafa'  # Very faint grey

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

# Plot with dashed lines for mean and ±1 std_dev
def plot_with_dashed_lines(ax, x, y, title, xlabel, ylabel, color):
    # Apply minimal style
    apply_minimal_style(ax)
    # Plot the main line
    ax.plot(x, y, color=color, linewidth=line_width)

    # Define light sky blue color
    light_sky_blue = '#4169E1'

    # Compute y-values at mean and mean±std_dev
    y_mean = np.interp(mean, x, y)
    y_plus_std = np.interp(mean + std_dev, x, y)
    y_minus_std = np.interp(mean - std_dev, x, y)

    # Plot vertical dashed lines from y=0 to the corresponding y-values
    ax.vlines(mean, ymin=0, ymax=y_mean, linestyle='--', color=light_sky_blue, linewidth=2.1, label=f'Mean ({mean})')
    ax.vlines(mean + std_dev, ymin=0, ymax=y_plus_std, linestyle='--', color=light_sky_blue, linewidth=1.4, label=f'+1 SD ({mean + std_dev:.2f})')
    ax.vlines(mean - std_dev, ymin=0, ymax=y_minus_std, linestyle='--', color=light_sky_blue, linewidth=1.4, label=f'-1 SD ({mean - std_dev:.2f})')

    # Set title and labels
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    # Remove grid if desired
    ax.grid(False)

# Display plots in two columns
col1, col2 = st.columns(2)
with col1:
    # Plot PDF
    fig_pdf, ax_pdf = plt.subplots()
    plot_with_dashed_lines(ax_pdf, x, pdf, 'Probability Density', 'x', 'PDF', 'blue')
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
