# Reinforcement-Learning-for-Stock-Trading
An implementation of simple reinforcement learning in the context of stock trading

Implement continuous price 
TODOs: 
1. Acquire 1-minute/5-minute data on a selection of stocks
2. Design the environment
  - Observations: high, low, close, current holdings, remaining cash, timestep
  - Actions: buy, sell or do nothing (for all stocks in the pool)
  - Rewarwds: change in portfolio value (take into account for transaction fees)
