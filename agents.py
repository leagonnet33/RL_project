import numpy as np
from utils import argmax


class BaseAgent:
    '''
    The base agent class that needs to be overwritten.
    Can be used as info as what methods an agent should have.
    '''
    def agent_init(self, agent_info: dict) -> None:
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
        self.epsilon_ = None
        self.rng_ = np.random.default_rng(seed=876438985230)
        self.verbose_ = verbose
    
    def agent_init(self, agent_info=None):
        '''
        Agent initialization
        '''
        if agent_info:
            self.epsilon_ = agent_info.get("epsilon", 0.0)
        else:
            self.epsilon_ = 0.1

    def agent_start(self, state):
        '''
        Choose agent's first action
        '''
        return self.rng_.integers(1, 3)
      
    def agent_step(self, reward, state, cash_amount, coins_amount, current_price):
        '''
        Agent takes a step i.e chooses an action.
        Ff the reward of the previous action was superior to 1 we buy again with proba
        1 - eps (or do nothing with proba eps) otherwise,
        we we sell with proba 1 - eps, or do nothing with proba eps
        '''
        sample = self.rng_.random()
        if reward >= 1:
            return (1 if sample > self.epsilon_ else 0)
        else:
            return (2 if sample > self.epsilon_ else 0)

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
        
class ArmCountAgent(BaseAgent):
    """
    Arm count agent
    """
    def __init__(self):
        '''
        Class object initialization
        '''
        self.last_state_ = None
        self.last_action_ = None
        self.num_actions = None
        self.step_size_ = None
        self.epsilon_ = None
        self.initial_value_ = None
        self.step_size_ = None

        self.rng_ = np.random.default_rng(seed=876438985230)

        self.possible_actions_ = None
        self.q_values = None
        self.arm_count = None
    
    def agent_init(self, agent_info=None):
        '''
        Initialize the agent's variables
        '''
        if agent_info:
            self.num_actions = agent_info.get("num_actions", 3)
            self.step_size_ = agent_info.get("step_size", 0.1)
            self.epsilon_ = agent_info.get("epsilon", 0.01)
            self.initial_value_ = agent_info.get("initial_value", 0.0)
            self.step_size_ = agent_info.get("step_size", .1)
        else:
            self.num_actions = 3
            self.step_size_ = 0.1
            self.epsilon_ = 0.01
            self.initial_value_ = 0.0
            self.step_size_ = .1

        self.last_action_ = 0
        self.possible_actions_ = [i for i in range(self.num_actions)]
        self.q_values = [0.0 for _ in range(self.num_actions)]
        self.arm_count = [0.0 for _ in range(self.num_actions)]
        
    def agent_start(self, state):
        '''
        First agent action
        '''
        action = self.rng_.integers(1, self.num_actions)
        self.last_state_ = state
        self.last_action_ = action
        return action

    def agent_step(self, reward, state, cash_amount, coins_amount, current_price):
        '''
        Method for agent to choose an action to take epsilon-greedily
        '''
        self.arm_count[self.last_action_] += 1
        self.q_values[self.last_action_] += (reward - self.q_values[self.last_action_]) / self.arm_count[self.last_action_]
        
        sample = self.rng_.random()
        if sample < self.epsilon_:
            current_action = self.rng_.choice(self.possible_actions_)
        else:
            current_action = argmax(self.rng_, self.q_values)
        
        self.last_action_ = current_action
        return current_action
    
    def agent_end(self, reward):
        target = reward
        self.q_values[self.last_state_] = self.q_values[self.last_state_] + self.step_size_ * (target - self.q_values[self.last_state_])
    
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
