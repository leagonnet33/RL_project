import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from runners import test_dqn_agent, train_dqn_agent
from models import DenseModel


def plot_results(agent,environment,portfolio,episodes=10):
    data_to_plot_all = {}
    for episode in range (1,episodes+1):
        data_to_plot_all[episode] = []
    data_to_plot = train_dqn_agent(agent, environment, portfolio, episodes)
    for episode in range (1,len(data_to_plot)):
        data_to_plot_all[episode].append(data_to_plot[episode])
        
    for episode in tqdm(range(1,len(data_to_plot)+1)):
        plt.plot(np.mean(data_to_plot_all[episode],axis=0),label=episode)
    plt.xlabel("Time")
    plt.ylabel("Portfolio\n current\n value\n (USD)",rotation=0, labelpad=40)
    plt.xlim(0,9000)
    plt.ylim(0,100000)
    plt.title('Portfolio current value at each episode')
    plt.legend()
    plt.show()
    

    
