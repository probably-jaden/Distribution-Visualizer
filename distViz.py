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
    'Normal', 'Gamma', 'Beta', 'Exponential', 'Weibull', 'Chi-Square',
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
elif distribution == 'Beta':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            a = st.select_slider('Alpha (α)', options=positive_tick_marks, value=2.0)
        with col2:
            b = st.select_slider('Beta (β)', options=positive_tick_marks, value=5.0)
    else:
        with col1:
            a = st.number_input('Alpha (α)', min_value=0.0, value=2.0)
        with col2:
            b = st.number_input('Beta (β)', min_value=0.0, value=5.0)
    dist = beta(a=a, b=b)
    x = np.linspace(0, 1, 500)
elif distribution == 'Exponential':
    if input_method == 'Sliders':
        scale = st.select_slider('Scale (λ)', options=positive_tick_marks, value=1.0)
    else:
        scale = st.number_input('Scale (λ)', min_value=0.0, value=1.0)
    dist = expon(scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
# elif distribution == 'Cauchy':
#     col1, col2 = st.columns(2)
#     if input_method == 'Sliders':
#         with col1:
#             loc = st.select_slider('Location (x₀)', options=tick_marks, value=0.0)
#         with col2:
#             scale = st.select_slider('Scale (γ)', options=positive_tick_marks, value=1.0)
#     else:
#         with col1:
#             loc = st.number_input('Location (x₀)', value=0.0)
#         with col2:
#             scale = st.number_input('Scale (γ)', min_value=0.0, value=1.0)
#     dist = cauchy(loc=loc, scale=scale)
#     x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
elif distribution == 'Chi-Square':
    if input_method == 'Sliders':
        df = st.select_slider('Degrees of Freedom (k)', options=positive_tick_marks, value=2.0)
    else:
        df = st.number_input('Degrees of Freedom (k)', min_value=1, value=2, step=1, format='%d')
    dist = chi2(df=df)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Log-Normal':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            mu = st.select_slider('Mean of log (μ)', options=tick_marks, value=0.0)
        with col2:
            sigma = st.select_slider('Standard Deviation of log (σ)', options=positive_tick_marks, value=1.0)
    else:
        with col1:
            mu = st.number_input('Mean of log (μ)', value=0.0)
        with col2:
            sigma = st.number_input('Standard Deviation of log (σ)', min_value=0.0, value=1.0)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Erlang':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            k = st.select_slider('Shape (k)', options=positive_tick_marks, value=2.0)
        with col2:
            scale = st.select_slider('Scale (θ)', options=positive_tick_marks, value=1.0)
    else:
        with col1:
            k = st.number_input('Shape (k)', min_value=1, value=2, step=1, format='%d')
        with col2:
            scale = st.number_input('Scale (θ)', min_value=0.0, value=1.0)
    dist = erlang(a=int(k), scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Pareto':
    if input_method == 'Sliders':
        b = st.select_slider('Shape (b)', options=positive_tick_marks, value=2.0)
    else:
        b = st.number_input('Shape (b)', min_value=0.0, value=2.0)
    dist = pareto(b=b)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
elif distribution == 'Log-Logistic':
    if input_method == 'Sliders':
        c = st.select_slider('Shape (c)', options=positive_tick_marks, value=2.0)
    else:
        c = st.number_input('Shape (c)', min_value=0.0, value=2.0)
    dist = fisk(c=c)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Noncentral Chi-Square':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            df = st.select_slider('Degrees of Freedom (k)', options=positive_tick_marks, value=2.0)
        with col2:
            nc = st.select_slider('Noncentrality (λ)', options=positive_tick_marks, value=1.0)
    else:
        with col1:
            df = st.number_input('Degrees of Freedom (k)', min_value=1, value=2, step=1, format='%d')
        with col2:
            nc = st.number_input('Noncentrality (λ)', min_value=0.0, value=1.0)
    dist = ncx2(df=df, nc=nc)
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Noncentral t':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            df = st.select_slider('Degrees of Freedom (k)', options=positive_tick_marks, value=10.0)
        with col2:
            nc = st.select_slider('Noncentrality (δ)', options=tick_marks, value=1.0)
    else:
        with col1:
            df = st.number_input('Degrees of Freedom (k)', min_value=1, value=10, step=1, format='%d')
        with col2:
            nc = st.number_input('Noncentrality (δ)', value=1.0)
    dist = nct(df=df, nc=nc)
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 500)
elif distribution == 'Kolmogorov-Smirnov':
    if input_method == 'Sliders':
        n = st.select_slider('Sample Size (n)', options=positive_tick_marks, value=10.0)
    else:
        n = st.number_input('Sample Size (n)', min_value=1, value=10, step=1, format='%d')
    dist = kstwo(n=int(n))
    x = np.linspace(0, dist.ppf(0.99), 500)
elif distribution == 'Power':
    if input_method == 'Sliders':
        a = st.select_slider('Shape (a)', options=positive_tick_marks, value=2.0)
    else:
        a = st.number_input('Shape (a)', min_value=0.0, value=2.0)
    dist = powerlaw(a=a)
    x = np.linspace(0, 1, 500)
elif distribution == 'Weibull':
    col1, col2 = st.columns(2)
    if input_method == 'Sliders':
        with col1:
            c = st.slider('Shape (c)', min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        with col2:
            scale = st.slider('Scale (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    else:
        with col1:
            c = st.number_input('Shape (c)', min_value=0.1, value=1.5)
        with col2:
            scale = st.number_input('Scale (λ)', min_value=0.1, value=1.0)
    dist = weibull_min(c=c, scale=scale)
    x = np.linspace(0, dist.ppf(0.99), 500)
else:
    st.write('Please select a distribution.')

st.markdown("---")

# Compute statistics
mean, std_dev, skewness = safe_stats(dist)
disp_mean = round(mean, 2) if mean != 'undefined' else mean
disp_std_dev = round(std_dev, 2) if std_dev != 'undefined' else std_dev
#disp_skewness = round(skewness, 2) if skewness != 'undefined' else skewness
disp_skewness = round(mean, 2) if mean != 'undefined' else mean

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

with tab2:
    st.header("Exponential Distribution")

    # Input for lambda parameter
    if input_method == 'Sliders':
        lambda_param = st.slider(
            "Select the lambda (rate) parameter",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
    else:
        lambda_param = st.number_input(
            "Enter the lambda (rate) parameter",
            min_value=0.1,
            value=1.0,
            step=0.1
        )

    # Input for number of samples
    if input_method == 'Sliders':
        num_samples = st.slider(
            "Select the number of samples",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
    else:
        num_samples = st.number_input(
            "Enter the number of samples",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            format='%d'
        )

    # Generate random samples
    samples = np.random.exponential(scale=1 / lambda_param, size=int(num_samples))
    st.write(
        f"Generated {int(num_samples)} samples from Exponential distribution with lambda = {lambda_param}"
    )

    # Compute the likelihood function for a range of lambda values
    sum_samples = np.sum(samples)
    n = int(num_samples)
    lambda_values = np.linspace(0.1, 10, 1000)
    log_likelihoods = n * np.log(lambda_values) - lambda_values * sum_samples

    # Plot the log-likelihood function
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, log_likelihoods, label='Log-Likelihood')
    plt.axvline(x=lambda_param, color='red', linestyle='--', label=f'Chosen λ = {lambda_param}')
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Function for Exponential Distribution')
    plt.legend()
    st.pyplot(plt)

    # Compute and display the log-likelihood at the chosen lambda
    log_likelihood_chosen = n * np.log(lambda_param) - lambda_param * sum_samples
    st.write(
        f"The log-likelihood at λ = {lambda_param} is {log_likelihood_chosen:.4f}"
    )

    # Show the maximum likelihood estimate (MLE) of lambda
    mle_lambda = n / sum_samples
    st.write(f"The maximum likelihood estimate (MLE) of λ is: {mle_lambda:.4f}")
