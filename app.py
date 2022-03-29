'''
This is the main file of the project, a wrapper to run our agents in their environments and
witness their performances.
'''
import pickle as pk

import torch

from loader import TradingDataLoader
from environments import TradingBotEnv
from portfolio import Portfolio
from models import DenseModel
from agents import DQNAgent
from runners import test_dqn_agent, train_dqn_agent
from display import plot_history_against_xchange_rates

torch.manual_seed(seed=876438985230)

data = TradingDataLoader().data()
environment = TradingBotEnv(data)
portfolio = Portfolio()

input_dimension = len(environment.metrics_) + len(portfolio.metrics_)
output_dimension = 3

model = DenseModel(input_dimension=input_dimension, output_dimension=output_dimension)
agent = DQNAgent(model)

# with open('./models/first_dqn.pkl', "rb") as f:
#     pickler = pk.Unpickler(f)
#     model = pickler.load()
#     agent = DQNAgent(model)

portfolio_history = train_dqn_agent(agent, environment, portfolio, episodes=2)
investment_return = portfolio.get_returns_percent(environment.get_current_price())
plot_history_against_xchange_rates(portfolio_history, data, investment_return)
