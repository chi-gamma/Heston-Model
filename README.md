# Heston Model Calibration using the Fourier-COS Method

A Python implementation of the Heston stochastic volatility model calibrated to market implied volatility surfaces using the Fourier-COS pricing method introduced by Fang & Oosterlee (2008).

---

## Features
- Heston stochastic volatility model
- Fourier-COS option pricing
- Black-Scholes pricing and implied volatility inversion (Brent's method)
- Vega-weighted calibration using Differential Evolution
- Market vs calibrated implied volatility smile comparison
---

## Methodology

For each maturity:

1. Construct the Heston characteristic function.
2. Price European options using the Fourier-COS expansion.
3. Recover Black-Scholes implied volatilities from model prices.
4. Minimize the weighted squared error between model and market implied volatilities.

The calibration objective is

Objective = Σ wᵢ (σ_model,i − σ_market,i)²

where the weights are proportional to Black-Scholes vegas.

---

## Project Structure

```
heston_calibration.py
```

Main components:

- Black-Scholes
- Implied volatility inversion
- Heston characteristic function
- Fourier-COS pricing engine
- COS payoff coefficient generation
- Differential Evolution calibration
- Smile visualization

---

## Example Calibration

### Market vs Heston Implied Volatility

<img width="1026" height="674" alt="image" src="https://github.com/user-attachments/assets/06f27099-ae42-4dcf-859e-5080b0bf291e" />


The calibrated Heston model captures the overall shape of the market implied volatility smile across maturities while maintaining computational efficiency through the COS pricing framework.

---

## References

1. Fang, F., & Oosterlee, C. W. (2008). *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*. SIAM Journal on Scientific Computing, 31(2), 826–848.

2. Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. Review of Financial Studies, 6(2), 327–343.
---
