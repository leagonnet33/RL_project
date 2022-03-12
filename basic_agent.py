#!/usr/bin/env python

from RLGlue.agent import BaseAgent

import numpy as np

class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.last_action = None
        self.num_actions = None
        self.q_values = None
        self.step_size = None
        self.epsilon = None
        self.initial_value = 0.0
        self.arm_count = [0.0 for _ in range(10)]

    def agent_init(self, agent_info={}):
        self.num_actions = agent_info.get("num_actions", 3)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.q_values = np.ones(agent_info.get("num_actions", 3)) * self.initial_value
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)

        self.last_action = 0

    def agent_start(self, observation):
        
        self.last_action = np.random.choice(self.num_actions)  # set first action to 0

        return self.last_action

    def agent_step(self, reward, observation):

        self.last_action = np.random.choice(self.num_actions)

        return self.last_action

    def agent_end(self, reward):
        
        pass

    def agent_cleanup(self):

      pass

    def agent_message(self, message):
        
        pass
      
class EpsilonGreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
       
        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################
        
        
        self.arm_count[self.last_action] += 1
        self.q_values[self.last_action] += (reward - self.q_values[self.last_action]) / self.arm_count[self.last_action]

        test = np.random.random()
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
