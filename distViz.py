import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, beta

st.title('Probability Distribution Explorer')

# # Select distribution
# distribution = st.selectbox('Select a distribution', ['Normal', 'Gamma', 'Beta'])

# # Input parameters
# if distribution == 'Normal':
#     mean = st.number_input('Mean (μ)', value=0.0)
#     std_dev = st.number_input('Standard Deviation (σ)', value=1.0, min_value=0.0)
#     dist = norm(loc=mean, scale=std_dev)
#     x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 500)
# elif distribution == 'Gamma':
#     shape = st.number_input('Shape (k)', value=2.0, min_value=0.0)
#     scale = st.number_input('Scale (θ)', value=2.0, min_value=0.0)
#     dist = gamma(a=shape, scale=scale)
#     x = np.linspace(0, dist.ppf(0.99), 500)
# elif distribution == 'Beta':
#     a = st.number_input('Alpha (α)', value=2.0, min_value=0.0)
#     b = st.number_input('Beta (β)', value=5.0, min_value=0.0)
#     dist = beta(a=a, b=b)
#     x = np.linspace(0, 1, 500)
# else:
#     st.write('Please select a distribution.')

