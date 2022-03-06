import pandas as pd
import numpy as np
import random
from collections import deque

class BaseEnvironment():
    '''
    A base reinforcement learning environment to build more complex
    environments uppon
    '''
    def env_init(self, data, env_info={}):
        '''
        Method to initialize the environment
        '''
        raise NotImplementedError

    def env_start(self, state):
        '''
        A method to take the first step in the environment
        after it was initialized
        '''
        raise NotImplementedError

    def env_step(self, action):
        '''
        A method to take a step in the environement (not the first one though)
        '''
        raise NotImplementedError

    def env_end(self, reward):
        '''
        A method to take the last step in the environment
        '''
        raise NotImplementedError

    def env_render(self):
        '''
        A method to render the environment to the user
        '''
        raise NotImplementedError

    def state(self, loc):
        '''
        A helper funtion to yield the current state to the user
        '''
        raise NotImplementedError



class TradingBotEnvironment(BaseEnvironment):
    '''
    The main environment of our project. We are developing a trading agent in a crypto
    environment. Therefore, in this environment, we have information relevant
    to our agent in the context of crypto-currency trading.
    '''

    def __init__(self) -> None:
        super().__init__()

    def env_init(self, data, env_info={}):
        '''
        Method to initialize the environment
        '''
        reward = None
        observation = None
        termination = None
        # see if we really need the line below
        self.reward_obs_term = (reward, observation, termination)
        # We define custom parameters below
        self.lookback_window_size = env_info.get("lookback_window_size",50) # we define the size on our back window
        self.delta = env_info.get("delta",1) # we define our step-size
        self.total_steps = env_info.get("total_steps",500) # we define the total steps we will go through
        self.initial_balance = env_info.get("initial_balance", 1000) # we define our initial balance
        # Define the different possible actions:
        # 0 we hold, 1 we buy, 2 we sell
        self.action_space = np.array([0, 1, 2])
        # the history of orders = what we have in our portfolio at each step
        self.orders_history = deque(maxlen=self.lookback_window_size)
        # the history of the market -- see if we really need this
        self.market_history = deque(maxlen=self.lookback_window_size)
        #the size of the state ??
        self.state_size = (self.lookback_window_size, 10)

        
    def env_start(self, state):
        '''
        Method to take the first step in the environment
        '''
        # Called when the episode starts, right before the agent starts. Like a reset
        reward = 0
        self.balance = self.initial_balance # what we have in liquidities in our portfolio, in $
        self.net_worth = self.initial_balance # the total amount of what we own (liquidities + cryptos valuation), in $
        self.prev_net_worth = self.initial_balance # the total amount of what we owned at previous step
        self.crypto_held = 0 # the total number of cryptos we own - must be an integer
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.start_step = self.lookback_window_size # we look at history from the beginning
        self.end_step = self.df_total_steps
        self.current_step = self.start_step # see how to define start_step ?

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            # Below we charge our data -- to modify with the naming and format of our real data
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state
        
    def next_observation(self):
        '''
        Method to return the next observation
        '''
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ]) # here we add the market data of our current step, to modifiy with our real data
        obs = np.concatenate((self.market_history,self.orders_history),axis=1)
        return obs
        
    def env_step(self,action):
        '''
        Method to take a step in the environment
        '''
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        
        current_unit_price = closing_price # MODIFY HERE to get the closing_price with our data. The price should be for one unit of crypto.
        
        if action == 0: # we choose not to do anything, nothing happens
            pass
        
        elif action == 1 and self.balance > current_unit_price:  #MODIFY HERE THE WAY WE BUY. For now: 
            #Buy one unit of crypto only
            self.crypto_bought += 1
            self.balance -= current_unit_price
            self.crypto_held += self.crypto_bought
            
        elif action == 2: # MODIFY HERE THE WAY WE SELL. For now:
            #We sell one unit of crypto only
            self.crypto_sold += 1
            self.balance += current_unit_price
            self.crypto_held -= self.crypto_sold
            
        self.prev_net_worth = self.current_net_worth
        self.current_net_worth = self.balance + self.crypto_held*current_unit_price 
        
        self.orders_history.append([self.balance,self.current_net_worth,self.crypto_bought,self.crypto_sold,self.crypto_held])
        # we define our reward as the benefit we made since previous step
        reward = self.current_net_worth - self.prev_net_worth
        
        if self.net_worth <= self.initial_balance/2 or self.current_step >= self.total_steps:
        # We decide to stop if we have lost half our initial money.
            done = True
        else:
            done = False
        
        obs = self.next_observation()
        
        return obs,reward,done
            
    # render environement
    def env_render(self):
        '''
        Method to render the environment to the user
        '''
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

    def env_end(self, reward):
        '''
        The last step to take in the environment
        '''
        # To implement
        return None

    def state(self, loc):
        '''
        A method that returns the state of the environment
        '''
        # To implement
        return None