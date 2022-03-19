from environments import *
from portfolio import *
from loader import TradingDataLoader
from random import *
import numpy as np
from utils import argmax

def run_agent(agent,initial_cash,epsilon,metric):
    env_info = {}
    all_averages = []
    data = TradingDataLoader().data()
    env = TradingBotEnv(data,metrics = metric)
    portfolio = Portfolio(portfolio_cash = initial_cash)
    is_done = False 
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
