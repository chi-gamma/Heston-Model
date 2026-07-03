import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import differential_evolution, brentq
from scipy.stats import norm
from scipy.special import ndtr


# =========================
# Black-Scholes
# =========================

def black_scholes_price(S, K, T, r, q, vol, phi):
    """
    European Black-Scholes price.

    phi =  1 for call
    phi = -1 for put
    """
    
    sqrt_T = np.sqrt(T)

    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    return phi * (
        S * np.exp(-q * T) * ndtr(phi * d1)
        - K * np.exp(-r * T) * ndtr(phi * d2)
    )


def black_scholes_vega(S, K, T, r, q, vol):
    sqrt_T = np.sqrt(T)

    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrt_T)

    return S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T


def implied_volatility(
    price,
    S,
    K,
    T,
    r,
    q,
    phi,
    vol_low=1.0e-6,
    vol_high=5.0,
):
    """
    Black-Scholes implied volatility using Brent's method.
    """

    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    lower_bound = max(phi * (discounted_spot - discounted_strike), 0.0)
    upper_bound = discounted_spot if phi == 1 else discounted_strike

    tol = 1.0e-10

    if price < lower_bound - tol:
        return np.nan

    if price > upper_bound + tol:
        return np.nan

    price = min(max(price, lower_bound), upper_bound)

    def objective(vol):
        return black_scholes_price(S, K, T, r, q, vol, phi) - price

    try:
        return brentq(
            objective,
            vol_low,
            vol_high,
            xtol=1.0e-8,
            rtol=1.0e-8,
            maxiter=100,
        )
    except ValueError:
        return np.nan


vectorized_implied_vol = np.vectorize(implied_volatility)
    

# =========================
# Heston characteristic function
# =========================

def heston_cf_log_return(
    tau,
    r,
    q,
    kappa,
    theta,
    vol_of_vol,
    v0,
    rho,
):
    """
    Heston characteristic function of log(S_T / S_0).
    """

    i = 1j

    def cf(u):
        u = np.asarray(u, dtype=complex)

        d = np.sqrt( (kappa - rho * vol_of_vol * i * u) ** 2  + vol_of_vol ** 2 * (u ** 2 + i * u) )

        g = ( (kappa - rho * vol_of_vol * i * u - d) / (kappa - rho * vol_of_vol * i * u + d) )

        exp_dt = np.exp(-d * tau)

        C = ( (1.0 - exp_dt) / (vol_of_vol ** 2 * (1.0 - g * exp_dt)) ) * (kappa - rho * vol_of_vol * i * u - d)

        A = ( i * u * (r - q) * tau + (kappa * theta / vol_of_vol ** 2)
            * ((kappa - rho * vol_of_vol * i * u - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
        )

        return np.exp(A + C * v0)

    return cf


# =========================
# COS engine
# =========================

def cos_price(
    cf,
    S0,
    r,
    tau,
    strikes,
    payoff_coefficients,
    a,
    b,
):
    """
    Payoff-agnostic COS pricing engine.
    
    The payoff is represented entirely by its cosine expansion coefficients.
    """

    K = np.asarray(strikes, dtype=float).reshape(-1, 1)
    H_k = np.asarray(payoff_coefficients, dtype=float).reshape(-1, 1)

    n_terms = len(H_k)

    k = np.arange(n_terms).reshape(-1, 1)
    u = k * np.pi / (b - a)

    x0 = np.log(S0 / K)

    mat = np.exp(1j * (x0 - a) @ u.T)

    weights = cf(u) * H_k
    weights[0] *= 0.5

    prices = np.exp(-r * tau) * K * np.real(mat @ weights)

    return prices.flatten()


def call_payoff_coefficients(a, b, n_terms):
    """
    COS coefficients for max(exp(x) - 1, 0).
    """

    k = np.arange(n_terms).reshape(-1, 1)

    if b <= 0.0:
        return np.zeros((n_terms, 1))
    
    exp_int, const_int = payoff_integrals(a, b, c=0.0, d=b, k=k)

    return 2.0 / (b - a) * (exp_int - const_int)


def put_payoff_coefficients(a, b, n_terms):
    """
    COS coefficients for max(1 - exp(x), 0).
    """

    k = np.arange(n_terms).reshape(-1, 1)
    
    exp_int, const_int = payoff_integrals(a, b, c=a, d=0.0, k=k)

    return 2.0 / (b - a) * (const_int - exp_int)


def payoff_integrals(a, b, c, d, k):
    """
    Core COS payoff integrals.

    exp_int   = integral of exp(x) times cosine basis
    const_int = integral of 1      times cosine basis
    """
    
    k = np.asarray(k, dtype=float)
    omega = k * np.pi / (b - a)

    const_int = np.empty_like(k)

    const_int[0] = d - c
    const_int[1:] = (
        np.sin(omega[1:] * (d - a))
        - np.sin(omega[1:] * (c - a))
    ) / omega[1:]

    exp_c = np.exp(c)
    exp_d = np.exp(d)

    exp_int = (
        np.cos(omega * (d - a)) * exp_d
        - np.cos(omega * (c - a)) * exp_c
        + omega * np.sin(omega * (d - a)) * exp_d
        - omega * np.sin(omega * (c - a)) * exp_c
    ) / (1.0 + omega**2)

    return exp_int, const_int


def truncation_interval(
    tau,
    volatility,
    width=10.0,
):
    """
    Symmetric truncation interval for the COS method.

    volatility : annualized volatility estimate used to approximate
                 the standard deviation of log-returns.
    """

    std = volatility * np.sqrt(tau)

    a = -width * std
    b = width * std

    return a, b


# =========================
# Calibration data structure
# =========================

@dataclass
class MarketSlice:
    tau: float
    r: float
    q: float
    strikes: np.ndarray
    market_vols: np.ndarray
    weights: np.ndarray


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)

    total = np.sum(weights)

    if not np.isfinite(total) or total <= 0.0:
        return np.ones_like(weights) / len(weights)

    return weights / total


def build_market_slices(quotes):
    """
    Converts quote DataFrame into a list of tenor-level market slices.
    """

    slices = []

    for tau, group in quotes.groupby("tenor"):
        group = group.sort_values("strike").reset_index(drop=True)

        strikes = group["strike"].to_numpy(dtype=float)
        market_vols = group["volatility"].to_numpy(dtype=float)
        vegas = group["vega"].to_numpy(dtype=float)

        market_slice = MarketSlice(
            tau=float(tau),
            r=float(group["rate"].iloc[0]),
            q=float(group["dividend"].iloc[0]),
            strikes=strikes,
            market_vols=market_vols,
            weights=normalize_weights(vegas),
        )

        slices.append(market_slice)

    return slices


# =========================
# Heston calibration
# =========================

def heston_objective(
    params,
    spot,
    market_slices,
    n_terms=256,
    truncation_width=10.0,
):
    kappa, theta, vol_of_vol, rho, v0 = params

    total_error = 0.0

    for market in market_slices:

        a, b = truncation_interval(
            market.tau,
            volatility=np.sqrt(max(theta, v0)),
            width=truncation_width,
        )

        cf = heston_cf_log_return(
            tau=market.tau,
            r=market.r,
            q=market.q,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            v0=v0,
            rho=rho,
        )

        coeffs = call_payoff_coefficients(a, b, n_terms)

        model_prices = cos_price(
            cf=cf,
            S0=spot,
            r=market.r,
            tau=market.tau,
            strikes=market.strikes,
            payoff_coefficients=coeffs,
            a=a,
            b=b,
        )

        model_vols = vectorized_implied_vol(
            model_prices,
            spot,
            market.strikes,
            market.tau,
            market.r,
            market.q,
            1,
        )

        if np.any(~np.isfinite(model_vols)):
            return 1.0e10

        vol_errors = model_vols - market.market_vols
        
        #total_error += np.sum(vol_errors**2)
        total_error += np.sum(market.weights * vol_errors**2)

    return total_error


def calibrate_heston(quotes):
    """
    Calibrates Heston stochastic volatility model to generic implied vol quotes.
    """

    quotes = quotes.copy().reset_index(drop=True)

    spot = float(quotes["spot"].iloc[0])

    quotes["vega"] = black_scholes_vega(
        quotes["spot"].to_numpy(dtype=float),
        quotes["strike"].to_numpy(dtype=float),
        quotes["tenor"].to_numpy(dtype=float),
        quotes["rate"].to_numpy(dtype=float),
        quotes["dividend"].to_numpy(dtype=float),
        quotes["volatility"].to_numpy(dtype=float),
    )

    market_slices = build_market_slices(quotes)

    bounds = [
        (0.001, 10.0),   # kappa
        (0.001, 0.25),   # theta variance
        (0.005, 2.0),    # vol-of-vol
        (-0.99, 0.50),   # rho
        (0.001, 0.25),   # v0 variance
    ]

    result = differential_evolution(
        heston_objective,
        bounds=bounds,
        args=(spot, market_slices),
        maxiter=150,
        polish=True,
        workers=1,
    )

    return result
    

def add_heston_model_vols(quotes, params, n_terms=256, truncation_width=10.0):
    quotes = quotes.copy()

    kappa, theta, vol_of_vol, rho, v0 = params
    spot = float(quotes["spot"].iloc[0])

    for tau, group in quotes.groupby("tenor"):
        idx = group.index

        r = float(group["rate"].iloc[0])
        q = float(group["dividend"].iloc[0])
        strikes = group["strike"].to_numpy(dtype=float)
        
        a, b = truncation_interval(
            tau,
            volatility=np.sqrt(max(theta, v0)),
            width=truncation_width,
        )

        cf = heston_cf_log_return(
            tau=tau,
            r=r,
            q=q,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            v0=v0,
            rho=rho,
        )

        coeffs = call_payoff_coefficients(a, b, n_terms)

        prices = cos_price(
            cf=cf,
            S0=spot,
            r=r,
            tau=tau,
            strikes=strikes,
            payoff_coefficients=coeffs,
            a=a,
            b=b,
        )
        
        vols = vectorized_implied_vol(
            prices,
            spot,
            strikes,
            tau,
            r,
            q,
            1,
        )

        quotes.loc[idx, "model_vol"] = vols

    quotes["vol_error"] = quotes["model_vol"] - quotes["volatility"]

    return quotes


def plot_market_vs_model_smiles(quotes):
    for tau, group in quotes.groupby("tenor"):

        group = group.sort_values("strike")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            group["strike"],
            group["volatility"],
            marker="o",
            linestyle="",
            label="Market",
        )

        ax.plot(
            group["strike"],
            group["model_vol"],
            linestyle="-",
            linewidth=2,
            label="Heston",
        )

        ax.set_title(f"Market vs Heston Implied Volatility (T = {tau:.2f} years)")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility")

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        ax.grid(True)
        ax.legend()

        plt.show()
    

# =========================
# Run
# =========================

quotes = pd.read_excel("C:/Users/nwozo/Desktop/CVI/option_df.xlsx")

result = calibrate_heston(quotes)

print("Calibration success:", result.success)
print("Objective value     :", result.fun)

kappa, theta, vol_of_vol, rho, v0 = result.x

print("kappa       :", kappa)
print("theta       :", theta, "long-run vol:", np.sqrt(theta))
print("vol_of_vol  :", vol_of_vol)
print("rho         :", rho)
print("v0          :", v0, "initial vol:", np.sqrt(v0))

calibrated_quotes = add_heston_model_vols(
    quotes=quotes,
    params=result.x,
    n_terms=256,
    truncation_width=10.0,
)

plot_market_vs_model_smiles(calibrated_quotes)    

