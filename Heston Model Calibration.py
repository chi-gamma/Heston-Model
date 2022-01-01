# import the relevant packages
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import brute, fmin, fsolve
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

#====================================SPY Call Options===================================#

tic = 'SPY'
obj = yf.Ticker(tic) # ticker object
expDates = obj.options
today = datetime.datetime.now()
total = []
for ttm in expDates:
    df = obj.option_chain(ttm)[0] # call options
    exp = df.contractSymbol.iloc[0][len(tic):len(tic)+6]
    exp = '20' + exp[0:2] + '-' + exp[2:4] + '-' + exp[4:]
    df['exp'] = exp # Expiry date on the contract symbol
    df['exp'] = pd.to_datetime(df.exp)
    df['ttm'] = ((df['exp'] - today) / pd.offsets.Day(1))/365
    total.append(df)
df = pd.concat(total)
df = df.dropna()
df = df.reset_index()

#=======================================================================================#

def bsm(S0, K, T, r, sigma, opType):
    d1 = (np.log(S0/K) + ((r + 0.5*sigma**2)*T)) / (sigma*np.sqrt(T))
    d2 = d1 - (sigma*np.sqrt(T))
    c = (S0 * norm.cdf(d1,0,1)) - (K*np.exp(-r*T)*norm.cdf(d2,0,1))
    p = c - S0 + (K*np.exp(-r*T)) # put-call parity
    if opType == 'call':
        return c
    else:
        return p
    
#=======================================================================================#

# Function to determine the implied volatilitiy of an option price
def imp_vol(S0, K, T, r, prc, opType): # opType = 'call' or 'put'
    
    # Difference between Price using an estimate volatility and actual price
    def difference(sigma):
        prc_est = bsm(S0, K, T, r, sigma, opType)
        return prc_est - prc
    
    sigma_est = np.sqrt(2*np.pi/T) * (prc/S0) # a good initial guess of the IV
    
    iv = fsolve(difference, sigma_est)[0]
    return iv

#=======================================================================================#

# Calculate the call price from the Heston Model
def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    int_value = quad(lambda u: H93_int_func(u, S0, K, T, r, kappa_v,
    theta_v, sigma_v, rho, v0), 0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K)
    / np.pi * int_value)
    return call_value

def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    char_func_value = H93_char_func(u - 1j * 0.5, T, r, kappa_v,
    theta_v, sigma_v, rho, v0)
    int_func_value = 1 / (u ** 2 + 0.25) \
    * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value

def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v)
    ** 2 - sigma_v ** 2 * (-u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) \
    / (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r * u * 1j * T + (c1 / sigma_v ** 2)
    * ((kappa_v - rho * sigma_v * u * 1j + c2) * T
    - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2
    * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value

#=================================Heston Error Function===================================#

i = 0
min_MSE = 500
def H93_error_function(p0):    
    global i, min_MSE
    global S0, df
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    #===================calibrating to the implied volatility==============================#
    se = []
    for row, option in df.iterrows():
        K = option['strike']
        T = option['ttm']
        r = 0.35/100
        prc = (option['bid'] + option['ask']) / 2
        model_value = H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)
        se.append((model_value - prc) ** 2)
    #======================================================================================#
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    i += 1
    return MSE

#===============================Heston Calibration Function================================#

def H93_calibration_full():
    ''' Calibrates H93 stochastic volatility model to market quotes. '''
    p0 = brute(H93_error_function,
        ((2.5, 10.6, 5.0), # kappa_v
        (0.01, 0.051, 0.01), # theta_v
        (0.005, 0.251, 0.1), # sigma_v
        (-0.75, 0.01, 0.25), # rho
        (0.01, 0.031, 0.01)), # v0
        finish=None)
    opt = fmin(H93_error_function, p0, xtol=0.000001, ftol=0.000001, maxiter=750, maxfun=900)
    return opt

#===============================Run Heston Calibration=====================================#
S0 = obj.history('1d').iloc[-1].Open # current asset price
opt = H93_calibration_full() # Run the Heston Model calibration
kappa_v, theta_v, sigma_v, rho, v0 = tuple(opt) # Heston calibrated parameters
print("\nThe calibrated Heston Model Parameters are:" + 
      "\nrate of reversion: " + str(kappa_v) +
      "\nlong run variance: "  + str(theta_v) +
      "\nvolatility of volatility: " + str(sigma_v) +
      "\ninitial variance: " + str(v0) +
      "\ncorrelation: " + str(rho))

#=======================calculating Heston model volatilities=============================#

hesImpVol = np.zeros(len(df))
for k, option in df.iterrows():
    r = 0.35/100
    T = option['ttm']
    K = option['strike']
    hesCall = H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)# heston price
    iv = imp_vol(S0, K, T, r, hesCall, 'call');
    hesImpVol[k] = iv


#================================computing the RMSE========================================#

mktIV = df.impliedVolatility.values # market implied volatility
hesRmse = np.sqrt(((hesImpVol - mktIV) ** 2).mean())
print("\nThe RMSE for the Heston Model is: ", hesRmse)

#================================Plotting the Volatility Surfaces==========================#
# Market Volatillity Surface
x = df['ttm']
y= df['strike']
z= df['impliedVolatility']
fig = plt.figure(figsize=(20,15))
ax = fig.gca(projection='3d', azim=-10, elev=10)
ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none', antialiased=True)
ax.set_ylabel('Maturity')
ax.set_xlabel('\nStrike')
ax.set_zlabel('Implied Volatility')
plt.show()

# Heston Model volatility surface
z= hesImpVol
fig = plt.figure(figsize=(20,15))
ax = fig.gca(projection='3d', azim=-10, elev=10)
ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none', antialiased=True)
ax.set_ylabel('Maturity')
ax.set_xlabel('\nStrike')
ax.set_zlabel('Implied Volatility')
plt.show()
