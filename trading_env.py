# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:42:29 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
import collections
import random


df = pd.read_csv('GOOGL.csv')
df.head()
trace = go.Candlestick(x=df.Date,
                       open=df.Open,
                       high=df.High,
                       low=df.Low,
                       close=df.Close)

trading_period_short =5
trading_period_long = 20
max_time = 10


# Generate the list of states
states = []
for i in range(trading_period_long-1, len(df)):

    list_high_short = []
    list_low_short = []

    for k in range(i-trading_period_short+1, i):
        list_high_short.append(df['High'][k])
        list_low_short.append(df['Low'][k])
    
    list_high_long = []
    list_low_long = []
    
    for k in range(i-trading_period_long+1, i):
        list_high_long.append(df['High'][k])
        list_low_long.append(df['Low'][k])
    
    closing_price = df['Close'][i]
    short_period_high = max(list_high_short)
    short_period_low = min(list_low_short)
    
    long_period_high = max(list_high_long)
    long_period_low = min(list_low_long)
    # Compute the indicator for relative position of closing price to trading period
    position_relative_short = 0
    if closing_price <= short_period_low:
        position_relative_short = 0
    elif closing_price <= (short_period_low + 0.2 * (short_period_high - short_period_low)):
        position_relative_short = 1
    elif closing_price <= (short_period_low + 0.8 * (short_period_high - short_period_low)):
        position_relative_short = 2
    elif closing_price <= short_period_high:
        position_relative_short = 3
    elif closing_price > short_period_high:
        position_relative_short = 4
   
    position_relative_long = 0
    if closing_price <= long_period_low:
        position_relative_long = 0
    elif closing_price <= (long_period_low + 0.2 * (long_period_high - long_period_low)):
        position_relative_long = 1
    elif closing_price <= (long_period_low + 0.8 * (long_period_high - long_period_low)):
        position_relative_long = 2
    elif closing_price <= long_period_high:
        position_relative_long = 3
    elif closing_price > long_period_high:
        position_relative_long = 4
        
    # Append these info to the list of states
    states.append(collections.OrderedDict([('Close', closing_price),
                   ('ST Relative Indicator', position_relative_short),
                   ('LT Relative Indicator', position_relative_long)]))
    
        
# Group sequeces of states into episodes:
episodes = []
for i in range (0, len(states)-max_time):
    episodes.append(states[i:i+max_time])

class Agent(object):
    
    def __init__(self, env):

        self.env = env
        self.starting_cash = 5000
        self.stock = 0
        self.cash = 0
        self.transaction_price = 0
        self.stock_position = 0
        
    def update(self):
        pass
    


class Environment(object):
    """Environment within which agents operate."""

    #  0：None,1: buy, 2: sell
    valid_actions = [0, 1, 2]
    max_time = 10
    
    
    def __init__(self, verbose=False):
        self.done = False
        self.t = 0
        self.episode = random.choice(episodes) # each episode is a list of states (of prices)
        
        # initiate agent
        self.agent = self.create_agent(Agent)
        
        # initiate state at time zero
        self.state = (self.episode[self.t]['ST Relative Indicator'], 
                      self.episode[self.t]['ST Relative Indicator'], 
                      self.agent.stock,
                      self.t)
        
        self.transaction_fee = 0.001
        self.nA = len(self.valid_actions)
    
    def step(self, action):
        """ This function is called when a time step is taken turing a trial. """

        max_time = len(self.episode) - 1 # the number of states available in the episode , -1 for indexing purpose
        state = self.episode[self.t] # state contains LT, ST indicator, closing price
        closing_price = state['Close']
        transaction_fee = self.transaction_fee
        agent = self.agent
        reward = 0
        
        if self.t < max_time-1:
            if agent.stock == 0: # if there is no stock position the agent can buy or do nothing
                if action == 1: # buy sstock at closing price
                    agent.stock_position = agent.starting_cash / closing_price * (1 - transaction_fee)
                    agent.cash = 0
                    agent.stock = 1
                    reward = 0
                elif action == 0: # no action
                    reward = 0
                else:
                    reward = -10000
            
            elif agent.stock == 1: # if there is a stock position the agent can sell or do nothing
                if action == 2: # sell stock at closing price
                    agent.cash = agent.stock_position * closing_price * (1 - transaction_fee)
                    reward = agent.cash - agent.starting_cash
                    self.done = True
                elif action == 0: #  no action
                    reward = 0
                elif action == 1:
                    reward = -10000
            
        elif self.t == max_time-1:
            if agent.stock == 1:
                # stock position is forced to liquidate
                agent.cash = agent.stock_position * closing_price * (1 - transaction_fee)
                reward = agent.cash - agent.starting_cash
                self.done = True
                
            else:
                reward = 0
                self.done = True        
        
        next_state = (self.episode[self.t+1]['ST Relative Indicator'], 
                       self.episode[self.t+1]['ST Relative Indicator'],
                       agent.stock,
                       self.t+1)
        self.t += 1
        
        return next_state, reward, self.done
        
    def reset(self):
        """This function is called at the beginning of a new trial (episode)"""
        
        self.done = False
        self.t = 0
        self.episode = random.choice(episodes)

        # initiate agent
        self.agent = self.create_agent(Agent)
        
        # initiate state at time zero
        self.state = (self.episode[self.t]['ST Relative Indicator'], 
                      self.episode[self.t]['ST Relative Indicator'], 
                      self.agent.stock,
                      self.t)
        
        return self.state
    
    def create_agent(self, agent_class, *args, **kwargs):
        
        agent = agent_class(self, *args, **kwargs)
        
        return agent
   
        
        