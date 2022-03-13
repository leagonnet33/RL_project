from environments import *
from portfolio import *
from loader import TradingDataLoader
from random import *
import numpy as np
from utils import argmax

class BaseAgent:
    def agent_init(self, agent_info={}):
        raise NotImplementedError
        
    def agent_start(self, state):
        raise NotImplementedError

    def agent_step(self, reward, state):
        raise NotImplementedError

    def agent_end(self, reward):
        raise NotImplementedError

    def agent_cleanup(self):        
        raise NotImplementedError
        
    def agent_message(self, message):
        raise NotImplementedError
        
class EpsilonGreedyAgent(BaseAgent):
    
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.last_action = None
        self.num_actions = 2
        self.q_values = [0.0 for _ in range(3)]
        self.step_size = 0.1
        self.epsilon = 0.1
        self.initial_value = 0.0
        self.arm_count = [0.0 for _ in range(10)]
    
    def agent_init(self,agent_info):
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        
        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)

        self.last_action = 0
        
    def agent_start(self,state):
        action = randint(0,2)
        self.last_state = state
        self.last_action = action
        return action

    def agent_step(self,reward,state,cash_amount,coins_amount,current_price):
        
        self.arm_count[self.last_action] += 1
        self.q_values[self.last_action] += (reward - self.q_values[self.last_action]) / self.arm_count[self.last_action]
        
        test = random()
        
        if cash_amount < current_price and int(coins_amount) == 0: # si on ne peut ni acheter ni vendre, on ne fait rien
            current_action = 0 
            
        elif cash_amount < current_price and int(coins_amount) > 0: # si on ne peut pas acheter, on vend ou on ne fait rien
            if test < 0.5: # avec probabilité 1/2
                current_action = 0
            else: current_action = 2
                
        elif cash_amount >= current_price and int(coins_amount) == 0: # si on ne peut pas vendre, on achète ou on ne fait rien
            if test < 0.5: # avec probabilité 1/2
                current_action = 0
            else: current_action = 1
        else:
            if test < self.epsilon:
                n = len(self.q_values)
                test = []
                for i in range(n):
                    test += [i]
                current_action = np.random.choice(test)
            else:
                current_action = argmax(self.q_values)
        
        self.last_action = current_action
        
        return current_action
    
    def agent_end(self, reward):
        target = reward
        self.values[self.last_state] = self.values[self.last_state] + self.step_size * (target - self.values[self.last_state])

        
def run_epsilon_greedy_agent(initial_cash,epsilon):
    agent = EpsilonGreedyAgent
    env_info = {}
    all_averages = []
    METRICS = ["open", "high", "low", "close", "volume ETH", "volume USDT", "tradecount"]
    data = TradingDataLoader().data()
    env = TradingBotEnv(data,metrics = METRICS)
    portfolio = Portfolio(portfolio_cash = initial_cash)
    is_done = False 
    agent = EpsilonGreedyAgent()
    agent_info = {"num_actions": 2, "epsilon": epsilon}

    action = agent.agent_start(state=None)
    i=0

    while not is_done:
        i+=1
    
        portfolio.apply_action(env.get_current_price(),action)
        is_done, state = env.step()
        current_reward = env.get_reward(action)
        if i%1000 == 1:
            print("iteration:" + str(i))
            print("Current holdings:", portfolio.get_current_holdings(env.get_current_price()))
            print("Reward:", env.get_reward(action))
            print("Current Price:", env.get_price_at(i))
        current_price = env.get_price_at(i)
        current_cash = portfolio.portfolio_cash_
        current_coins = portfolio.portfolio_coin_
        action = agent.agent_step(current_reward,state,current_cash,current_coins,current_price)
        if i%1000 == 1:
            print("action: ", action)
    
    print("Final holdings:", portfolio.get_current_holdings(env.get_current_price()))

