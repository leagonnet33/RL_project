from environments import *
from portfolio import *
from loader import TradingDataLoader
from random import *

def run_very_basic_agent(epsilon,cash_available = 1000,num_coins_per_order=1, recent_k=0):
    METRICS = ["open", "high", "low", "close", "volume ETH", "volume USDT", "tradecount"]
    data = TradingDataLoader().data()
    env = TradingBotEnv(data,metrics = METRICS)
    portfolio = Portfolio(portfolio_cash = cash_available)
    is_done = False 
    
    i = 0
    current_reward = 1
    action = 0
    
    while not is_done:
        i+=1
        n = random()
        if portfolio.portfolio_cash_ < env.get_current_price():
            print('not enough cash!') # si pas assez de cash pour acheter, on vend ou on ne fait rien avec proba 1/2
            if n > 0.5:
                action = 0
            else: action = 2
        else:
            if current_reward >=1:
                if n > epsilon:
                    action = 1
                else: action = 0
            else: 
                if n > epsilon:
                    action = 2
                else: action = 0
        if i%1000 == 1:
            print(i)
            print("Current holdings:", portfolio.get_current_holdings(env.get_current_price()))
            print("Reward:", env.get_reward(action))
            print("Current Price:", env.get_price_at(i))
        portfolio.apply_action(env.get_current_price(),action)
        is_done, state = env.step()
        current_reward = env.get_reward(action)

    print("Final holdings:", portfolio.get_current_holdings(env.get_current_price()))
    return None
