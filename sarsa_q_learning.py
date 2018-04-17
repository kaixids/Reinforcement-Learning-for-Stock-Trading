# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:42:49 2018

@author: Administrator
"""
%matplotlib inline 

import random
import math
import numpy as np
from trading_env import Agent, Environment
from collections import defaultdict, deque
import sys
import matplotlib.pyplot as plt 


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 
    
        
    def __init__(self, env, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.valid_actions = self.env.valid_actions  # The list of valid actions

        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor


    def get_probs(self, Q_s, epsilon, nA):
        policy_s = np.ones(nA) * epsilon / nA
        best_a  = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    def sarsa(self, env, num_episodes, alpha, gamma=1.0):
        # initialize action-value function (empty dictionary of arrays)
        Q = defaultdict(lambda: np.zeros(env.nA))
        # initialize performance monitor
        # loop over episodes
        for i_episode in range(1, num_episodes+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
                sys.stdout.flush()   
            
            nA = env.nA
            epsilon  = 1 / (i_episode+1)
            state = env.reset()
            while True:
                action = self.valid_actions[np.random.choice(np.arange(nA), p=self.get_probs(Q[state], epsilon, nA))]
                next_state, reward, done = env.step(action)
                next_action = np.random.choice(np.arange(nA), p=self.get_probs(Q[next_state], epsilon, nA))
                Q[state][action] = Q[state][action] + alpha * (reward + gamma*Q[next_state][next_action] - Q[state][action])
                state = next_state
                if done:
                    break
      
        
        for k, v in Q.items():
            print('{} {}'.format(k, v))
        return Q
    
    
    def estimate_returns(self, env, policy, num_episodes):
        returns = []
        # initialize performance monitor
        # loop over episodes
        for i_episode in range(1, num_episodes+1):

            state = env.reset()
            return_sum = 0
            while True:
                action = policy[state]
                next_state, reward, done = env.step(action)
                return_sum += reward
                state = next_state
                if done:
                    break
                
            returns.append(return_sum)
            
        return returns

        
        
def run():

    env = Environment()
    agent = env.create_agent(LearningAgent, epsilon=1, alpha=0.6)
    Q_sarsa = agent.sarsa(env, 100000, 0.7, 1)
    
    policy_sarsa = {}
    for key, value in Q_sarsa.items():
        best_action_value = max(value)
        policy_sarsa[key] = value.tolist().index(best_action_value)
    
    print(policy_sarsa)
    
    returns = agent.estimate_returns(env, policy_sarsa, 10000)
    cumulative_returns = np.cumsum(returns)
    
    plt.plot(cumulative_returns)
    plt.show()
    
    average_return = (sum(returns) / len(returns))
    print('The average return following the optimal strategy is : $', average_return)
    

if __name__ == '__main__':
    run()

























