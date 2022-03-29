import numpy as np
import matplotlib.pyplot as plt
from utils import generate_rgb_color

def plot_history_against_xchange_rates(
    history,
    xchange_data,
    investment_return=.0,
    color_seed=866549187,
    save_path=None):
    '''
    A function that plots the history of portfolio values for all epsiodes
    against the values of the exchange rate
    '''
    random_gen = np.random.default_rng(seed=color_seed)

    # Get xchange useful data
    dates = xchange_data.index
    exchange_rates = xchange_data['open']

    # Primary plot
    _, ax1 = plt.subplots()

    if isinstance(history, dict):
        for episode, values in history.items():
            ax1.plot(dates[::24], values[::24], color=generate_rgb_color(random_gen), linewidth=1, label=f"episode {episode + 1}")
    else:
        ax1.plot(dates[::24], history[::24], color=generate_rgb_color(random_gen), linewidth=1, label="Test episode")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Portfolio value (USD)", rotation=90, labelpad=40)
    ax1.legend()

    # Secondary plot
    ax2 = ax1.twinx()
    ax2.plot(np.arange(dates[::24].shape[0]), exchange_rates[::24], color="black", label="X-change rate", linestyle='dashed')
    ax2.set_ylabel("X-change rate", rotation=270, labelpad=40)
    ax2.legend()

    plt.title(f'Portfolio val and xchange rates - ROI: {investment_return: .2f}%')
    plt.xticks([])

    if save_path:
        plt.savefig(save_path)

    plt.show()

if __name__ =="__main__":
    import pickle as pk
    from loader import TradingDataLoader

    data = TradingDataLoader().data()
    with open('./training_history/temp.pkl', "rb") as f:
        pickler = pk.Unpickler(f)
        portfolio_history = pickler.load()

    plot_history_against_xchange_rates(portfolio_history, data)
