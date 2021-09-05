# import the relevant packages
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import brute, fmin, fsolve, least_squares
from scipy.integrate import quad
from scipy.stats import norm

#=====================================SPY Call Options=====================================#
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
#==========================================================================================#
def vega(S0,K,T,r,sigma):
    ''' Returns Vega of option. '''
    d1 = ((np.log(S0/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
    vega = S0 * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    return vega

def callValue(S0,K,T,r,sigma):
    ''' Returns option value using Black-Scholes. '''
    d1 = ((np.log(S0/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
    d2 = ((np.log(S0/K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
    value = (S0*norm.cdf(d1, 0.0, 1.0)-K*np.exp(-r*T)*norm.cdf(d2, 0.0, 1.0))
    return value

def imp_vol(S,K,T,r,C0, sigma_est):
    ''' Returns implied volatility given option price. '''
    def difference(sigma):
        callPrice = callValue(S,K,T,r,sigma)
        return callPrice - C0
    iv = fsolve(difference, sigma_est)[0]
    return iv
#============================Black-Scholes Calibration Function============================#
def bsm_evaluate(sigma_est):
    r = 0.03
    diffs = np.zeros(len(df))
    for j, row in df.iterrows():
        C0 = (row['bid'] + row['ask']) / 2
        K = row['strike']
        T = row['ttm']
        bsmValue = callValue(S0,K,T,r,sigma_est)
        diffs[j] = C0 - bsmValue
    return diffs
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
        IV = option['impliedVolatility'] # market implied Volatility
        r = 0.03
        sigma_est = 3.5
        model_value = H93_call_value(S0, K, T, 0, kappa_v, theta_v, sigma_v, rho, v0)
        model_IV = imp_vol(S0,K,T,r,model_value, sigma_est)
        vg = vega(S0,K,T,r,IV) # call vega
        se.append(((model_IV - IV) * vg) ** 2)
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
#====================Calculate the call price from the Heston Model========================#
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
#===============================Run Heston Calibration=====================================#
S0 = obj.history('1d').iloc[-1].Close # current asset price
opt = H93_calibration_full() # Run the Heston Model calibration
kappa_v, theta_v, sigma_v, rho, v0 = tuple(opt) # Heston calibrated parameters
print("\nThe calibrated Heston Model Parameters are:" + 
      "\nrate of reversion: " + str(kappa_v) +
      "\nlong run variance: "  + str(theta_v) +
      "\nvolatility of volatility: " + str(sigma_v) +
      "\ninitial variance: " + str(v0) +
      "\ncorrelation: " + str(rho))
#==============================Run Black-Scholes Calibration===============================#
sigma_est = 1.5
plsq = least_squares(bsm_evaluate, sigma_est, bounds=(0.0, 10.0))
bsmParams = plsq.x[0]
print("\nThe calibrated Black-Scholes volatility is: " + str(bsmParams))
#=======================calculating Heston model volatilities=============================#
hesImpVol = np.zeros(len(df))
for k, option in df.iterrows():
    sigma_est = 4.4
    r = 0.03
    T = option['ttm']
    K = option['strike']
    hesCall = H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)# heston price
    # calculate the implied volatility of the Heston Price
    iv = imp_vol(S0,K,T,r,hesCall, sigma_est)
    hesImpVol[k] = iv
#======================calculating Black-Scholes model volatilities========================#
blackImpVol = np.zeros(len(df)) # constant black-scholes calibrated volatility
for k, option in df.iterrows():
    sigma_est = bsmParams # set the estimate to the calibrated volatility
    r = 0.03
    T = option['ttm']
    K = option['strike']
    bsmCall = callValue(S0,K,T,r,bsmParams) # Calculate the Black-Scholes Price
    iv = imp_vol(S0,K,T,r,bsmCall, sigma_est) # Calculate the implied volatility of BSM price
    blackImpVol[k] = iv
#================================computing the RMSE========================================#
mktIV = df.impliedVolatility.values # market implied volatility
hesRmse = np.sqrt(((hesImpVol - mktIV) ** 2).mean())
blackRmse = np.sqrt(((blackImpVol - mktIV) ** 2).mean())
print("\nThe RMSE for the Heston Model is: ", hesRmse)
print("\nThe RMSE for the Black-Scholes Model is: ", blackRmse)
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

# Black-Scholes volatility surface
z = blackImpVol
fig = plt.figure(figsize=(20,15))
ax = fig.gca(projection='3d', azim=-10, elev=10)
ax.plot_trisurf(x, y, z, cmap='plasma', edgecolor='none', antialiased=True)
ax.set_ylabel('Maturity')
ax.set_xlabel('\nStrike')
ax.set_zlabel('Implied Volatility')
plt.show()

