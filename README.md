# Dynamic Programming for Deterministic and Stochastic Multi-Period Inventory Control

This project implements a finite-horizon inventory control model solved using dynamic programming.

It consists of two parts:
- A deterministic single-product model
- A stochastic two-product extension  


## Part I – Deterministic Model

A seller manages inventory over a 10-day selling period.

Key features:
- Fixed demand generated using a Beta distribution (treated as known during optimization)
- Inventory capacity constraint
- Daily ordering limits
- Selling price, purchasing cost, holding cost
- Wage cost differing between weekdays and weekends

The model determines:
- Optimal daily order quantities
- Optimal starting day
- Maximum achievable total profit



## Part II – Stochastic Two-Product Extension

The model is extended to:
- Two products
- Random demand modeled using independent Binomial distributions
- Expected profit maximization
- Two inventory state variables
- Two control variables (daily order quantities)
- Capacity and ordering constraints for both products
- Terminal value for unsold inventory
