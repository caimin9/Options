import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from scipy.stats import norm
import streamlit as st
import yfinance as yf
import random
from random import sample

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price

# Greek Calculations
def delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return theta / 365  # Return per day

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # Return as percentage

def rho(S, K, T, r, sigma, option_type):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Return as percentage
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Return as percentage


def monte_carlo_option_pricing(S, K, T, r, sigma, option_type, simulations=10000):
    np.random.seed(42)
    payoff = []
    for _ in range(simulations):
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal(0, 1))
        if option_type == 'call':
            payoff.append(max(0, ST - K))
        elif option_type == 'put':
            payoff.append(max(0, K - ST))
    
    option_price = np.exp(-r * T) * np.mean(payoff)
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
delta_val = delta(S, K, T, r, sigma, option_type)
gamma_val = gamma(S, K, T, r, sigma)
theta_val = theta(S, K, T, r, sigma, option_type)
vega_val = vega(S, K, T, r, sigma)
rho_val = rho(S, K, T, r, sigma, option_type)
mc_price = monte_carlo_option_pricing(S, K, T, r, sigma, option_type)

    #Outputs
st.write(f'Monte Carlo Estimated Option Price: {mc_price:.2f}')
st.write(f'The price of the {option_type} option is {option_price:.2f}')
st.write(f'Delta: {delta_val:.4f}')
st.write(f'Gamma: {gamma_val:.4f}')
st.write(f'Theta (per day): {theta_val:.4f}')
st.write(f'Vega (per % volatility): {vega_val:.4f}')
st.write(f'Rho (per % rate): {rho_val:.4f}')

x = np.linspace(0.01, 10, 100)
y = [black_scholes(S, K, t, r, sigma, option_type) for t in x]

# Plot the graph
fig, ax = plt.subplots()
ax.plot(x, y, label=f'{option_type.capitalize()} Option Price')
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Option Price')
ax.legend()


#Plotting the Delta and Gamma
time_range = np.linspace(0.01, T, 100)
delta_vals = [delta(S, K, t, r, sigma, option_type) for t in time_range]
gamma_vals = [gamma(S, K, t, r, sigma) for t in time_range]

fig, ax = plt.subplots()
ax.plot(time_range, delta_vals, label='Delta')
ax.plot(time_range, gamma_vals, label='Gamma')
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Value')
ax.legend()
# Display the graph in Streamlit
st.pyplot(fig)
