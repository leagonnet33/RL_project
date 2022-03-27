'''
This is the main file of the project, a wrapper to run our agents in their environments and
witness their performances.
'''
from loader import TradingDataLoader
from environments import TradingBotEnv
from portfolio import Portfolio
from models import DenseModel
from agents import DQNAgent
from runners import test_dqn_agent, train_dqn_agent
from plot_functions import plot_results
import pickle as pk

data = TradingDataLoader().data()
environment = TradingBotEnv(data)
portfolio = Portfolio()

input_dimension = len(environment.metrics_) + len(portfolio.metrics_)
output_dimension = 3

# model = DenseModel(input_dimension=input_dimension, output_dimension=output_dimension)
with open('./models/first_dqn.pkl', 'rb') as f:
    pickler = pk.Unpickler(f)
    model = pickler.load()

agent = DQNAgent(model)

# train_dqn_agent(agent, environment, portfolio, save='./models/first_dqn.pkl')
history = test_dqn_agent(agent, environment, portfolio)

print(history[-10:])
