from utils import argmax
import numpy as np


class BaseAgent:
    '''
    The base agent class that needs to be overwritten.
    Can be used as info as what methods an agent should have.
    '''
    def agent_init(self, agent_info={}):
        '''
        Agent initialization of variables
        '''
        raise NotImplementedError
        
    def agent_start(self, state):
        '''
        First action the agent ought to take
        '''
        raise NotImplementedError

    def agent_step(self, reward, state, cash_amount, coins_amount, current_price):
        '''
        The method that makes the agent take a step (choose an action)
        '''
        raise NotImplementedError

    def agent_end(self, reward):
        '''
        The last action the agent ought to take
        '''
        raise NotImplementedError

    def agent_cleanup(self):
        '''
        A method to clean every variables the agent used
        '''
        raise NotImplementedError
        
    def agent_message(self, message):
        '''
        A method when the agent has to ping the user and tell him smthg
        '''
        raise NotImplementedError
        
        
class DumbAgent(BaseAgent):
    '''
    A very basic agent that chooses stupid actions
    '''
    def __init__(self, verbose=False):
        '''
        Class object initialization
        '''
        self.last_action = None
        self.num_actions = 2
        self.q_values = [0.0 for _ in range(3)]
        self.step_size = 0.1
        self.epsilon = 0.1
        self.initial_value = 0.0
        self.arm_count = [0.0 for _ in range(10)]
        self.last_state = None
        self.rng_ = np.random.default_rng(seed=876438985230)

        self.verbose_ = verbose
    
    def agent_init(self, agent_info):
        '''
        Agent initialization
        '''
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        
        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)

        self.last_action = 0

    def agent_start(self, state):
        '''
        Choose agent's first action
        '''
        action = self.rng_.integers(2)
        self.last_state = state
        self.last_action = action
        return action
      
    def agent_step(self, reward, state, cash_amount, coins_amount, current_price):
        '''
        Agent takes a step i.e chooses an action.
        Ff the reward of the previous action was superior to 1 we buy again with proba
        1 - eps (or do nothing with proba eps) otherwise,
        we we sell with proba 1 - eps, or do nothing with proba eps
        '''
        sample = self.rng_.random()
        if reward >= 1:
            return 1 if sample > self.epsilon else 0
        else:
            return 2 if sample > self.epsilon else 0

    def agent_end(self, reward):
        '''
        The last action the agent ought to take
        '''
        return None

    def agent_cleanup(self):
        '''
        A method to clean every variables the agent used
        '''
        return None
        
    def agent_message(self, message):
        '''
        A method when the agent has to ping the user and tell him smthg
        '''
        return None
        
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
