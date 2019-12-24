"""
Base class describing multi armed bandit problems
"""
        
import numpy as np

from abc import ABC
from abc import abstractclassmethod

class MultiArmedBanditProblem(ABC):
    
    def __init__(self,number_of_arms, number_of_trials):
        self.number_of_arms = number_of_arms
        self.number_of_trials = number_of_trials
        
        self.number_of_selections = np.zeros(self.number_of_arms)
        
        self.selected_arms = np.zeros(number_of_trials)
        
    @abstractclassmethod
    def warm_up(self):
        pass
    
    @abstractclassmethod
    def run_single_step(self):
        pass
    
    @abstractclassmethod
    def run_experiment(self):
        pass
    
    def get_max_index(self,array):
        return np.argwhere(array==array.max()).ravel()[0]
    
    def compute_step_cumulative_reward(self, data):
        return np.cumsum([data[step,int(self.selected_arms[step])] for step in range(self.number_of_trials)])