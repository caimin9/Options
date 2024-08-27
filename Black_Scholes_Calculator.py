import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from scipy.stats import norm
import streamlit as st

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price

# the title of my ui
st.title('Black-Scholes Option Pricing Model')
S = st.number_input('Stock Price', min_value=0.01, max_value=100000.0, value=100.0)
K = st.number_input('Strike Price', min_value=0.01, max_value=100000.0, value=100.0)
T = st.number_input('Time to Maturity', min_value=0.01, max_value=10.0, value=1.0)
r = st.number_input('Risk-Free Rate', min_value=0.0, max_value=1.0, value=0.05)
sigma = st.number_input('Volatility', min_value=0.0, max_value=1.0, value=0.2)
option_type = st.selectbox('Option Type', ['call', 'put'])
option_price = black_scholes(S, K, T, r, sigma, option_type)

st.write(f'The price of the {option_type} option is {option_price:.2f}')

x = np.linspace(0.01, 10, 100)
y = [black_scholes(S, K, t, r, sigma, option_type) for t in x]

# Plot the graph
fig, ax = plt.subplots()
ax.plot(x, y, label=f'{option_type.capitalize()} Option Price')
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Option Price')
ax.legend()

# Display the graph in Streamlit
st.pyplot(fig)
