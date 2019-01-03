# Reinforcement-Learning-for-Stock-Trading
An implementation of simple reinforcement learning in the context of stock trading

Implement continuous price 

Pipeline TODOs: 
Data Gathering: Acquire 1-minute/5-minute data on a selection of stocks

Data Wrangling: Don't think there is much to do here. All numbers are numerical

Algorithm: Design the environment
  - Observations: high, low, close, current holdings, remaining cash, timestep
  - Actions: buy, sell or do nothing (for all stocks in the pool)
  - Rewarwds: change in portfolio value (take into account for transaction fees)

Thinking TODOs:
1. What kind of hyper parameters should be set?
2. Hardcoded rules from domain expertise? Can it be a transient process?
