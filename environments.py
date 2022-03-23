import numpy as np

METRICS = ["open", "high", "low", "close", "volume ETH", "volume USDT", "tradecount"]

class BaseEnv():
    '''
    A base reinforcement learning environment to build more complex
    environments uppon
    '''
    def __init__(self) -> None:
        pass
    
    def step(self):
        '''
        Method to take a step in the environment
        '''
        raise NotImplementedError

    def reset(self):
        '''
        A public method to reset the environment
        '''
        raise NotImplementedError
        
    def get_reward(self, action):
        '''
        A method to get the reward value from a specific action
        '''
        raise NotImplementedError

class TradingBotEnv(BaseEnv):
    '''
    The main environment of our project. We are developing a trading agent in a crypto
    environment. Therefore, in this environment, we have information relevant
    to our agent in the context of crypto-currency trading, mostly market prices,
    exchange rates, ...
    '''

    def __init__(self, data, metrics=METRICS, lookback_window_size=20):
        super().__init__()
        assert "open" in metrics, "You need at least an 'open' price in your metrics"
        self.lookback_window_size_ = lookback_window_size
        self.metrics_ = metrics
        self.market_data_ = data[metrics]
        self.length_ = len(data)
        self.current_index_ = 0

        self.action_space_ = np.array([0, 1, 2])

        self.is_done_ = np.zeros(len(self.market_data_), dtype=bool)
        self.is_done_[-1] = True

        ### Derive some new features from the data
        self.rolling_mean_ = self.market_data_["open"].rolling(window=self.lookback_window_size_, center=False, min_periods=0).mean()
        self.rolling_std_ = self.market_data_["open"].rolling(window=self.lookback_window_size_, center=False, min_periods=0).std()
        self.rolling_std_ = self.rolling_std_.fillna(value=self.rolling_std_.iloc[1])
        self.upper_band_, self.lower_band_ = self.rolling_mean_ + 2 * self.rolling_std_, self.rolling_mean_ - 2 * self.rolling_std_

        ### Create the state dictionnary
        self.state_dict_ = {}
        for column_name in self.market_data_.columns:
            self.state_dict_[column_name] = self.market_data_[column_name]
        self.state_dict_["rolling_mean"] = self.rolling_mean_
        self.state_dict_["rolling_std"] = self.rolling_std_
        self.state_dict_["upper_band"] = self.upper_band_
        self.state_dict_["lower_band"] = self.lower_band_
        self.state_dict_["price_over_sma"] = self.market_data_["open"]/self.rolling_mean_

        self.metrics_ += ["rolling_mean", "rolling_std", "upper_band", "lower_band", "price_over_sma"]
        
    def step(self):
        '''
        Method to take a step in the environment
        '''
        is_done = self.is_done_[self.current_index_]
        observation = []
        for metric in self.metrics_:
            observation.append(self.state_dict_[metric][self.current_index_])
        if not is_done:
            self.current_index_ += 1
        return is_done, observation

    def get_reward(self, action):
        '''
        A method to get the reward value from a specific action
        '''
        assert action in self.action_space_, f"You cannot take an action that's not one of: {self.action_space_}"
        a = 0
        if action == 1:
            a = 1
        elif action == 2:
            a = -1
            
        price_t = self.get_current_price()
        price_t_minus_1 = self.get_price_at(self.current_index_ - 1)
        price_t_minus_n = self.get_price_at(self.current_index_ - self.lookback_window_size_)
        
        return (1 + a*(price_t - price_t_minus_1)/price_t_minus_1)*price_t_minus_1/price_t_minus_n

    def reset(self):
        '''
        A public method to reset the environment
        '''
        self.current_index_ = 0

    def get_current_price(self):
        '''
        A method to get the current opening price
        '''
        return self.state_dict_["open"][self.current_index_]
    
    def get_final_price(self):
        '''
        A method to get the final opening price
        '''
        return self.state_dict_["open"][-1]
    
    def get_price_at(self, index):
        '''
        A method to get the opening price at a given index
        '''
        if index < 0:
            return self.state_dict_["open"][0]
        if index >= self.length_:
            return self.get_final_price()
        return self.state_dict_["open"][index]

    def get_metrics(self, metrics=None):
        '''
        A method to get the metrics at the current index
        '''
        if not metrics:
            metrics = self.metrics_
        return np.array([self.state_dict_[metric][self.current_index_] for metric in metrics])

    def plot(self, metrics=None):
        '''
        A method to plot the chosen metrics with matplotlib
        '''
        import matplotlib.pyplot as plt

        if not metrics:
            metrics = self.metrics_

        plt.figure()
        for metric in metrics:
            ax = self.state_dict_[metric].plot()

        ax.legend(metrics)
        plt.show()

if __name__ == "__main__":
    from loader import TradingDataLoader
    
    ## TEST ENVIRONMENT METHODS ##

    data = TradingDataLoader().data()
    env = TradingBotEnv(data)

    # # env.step()
    # is_done = False
    # i = 0
    # while not(is_done):
    #     is_done, obs = env.step()
    #     if i < 3:
    #         print(obs)
    #     i += 1
    # print(f"\nThe operation successfully ended after {i} steps")

    # # env.get_reward()
    # r1 = env.get_reward(0)
    # r2 = env.get_reward(1)
    # r3 = env.get_reward(2)
    # print(r1, r2, r3)
    # env.get_reward(67) # Should raise an error

    # # env.reset()
    # env.reset()
    # print(env.current_index_ == 0)

    # # env.get_current_price()
    # for _ in range(10):
    #     _ = env.step()
    # print(env.get_current_price())
    
    # # env.get_final_price()
    # print(env.get_final_price())
    
    # # env.get_price_at(8000)
    # print(env.get_price_at(8000))

    # # env.get_metrics()
    # temp = env.get_metrics()
    # print(temp)
    # temp = env.get_metrics(metrics=["open", "close"])
    # print(temp)

    # env.plot()
    env.plot(metrics=["open", "close"])
