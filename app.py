'''
This is the main file of the project, a wrapper to run our agents in their environments and
witness their performances.
'''
from loader import TradingDataLoader
from environments import TradingBotEnv
from portfolio import Portfolio
from agents import DumbAgent
from trainers import run_agent

data = TradingDataLoader().data()
environment = TradingBotEnv(data)
portfolio = Portfolio()
agent = DumbAgent()

run_agent(agent, environment, portfolio)