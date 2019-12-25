"""
Implementation of different flavours of the epsilon-greedy algorithm
"""

import numpy as np

from mab.base.base_mab import MultiArmedBanditProblem

class BernoulliEGreedy(MultiArmedBanditProblem):
    def __init__(self,number_of_arms, number_of_trials,epsilon,init_probas):
        super(BernoulliEGreedy,self).__init__(number_of_arms, number_of_trials)
        
        if np.abs(epsilon) > 1:
            raise ValueError('Epsilon parameter must be bound within 0 and 1')
        elif len(init_probas) != number_of_arms:
            raise ValueError('Number of arms ({}) does not match the number of given priors ({})'.format(number_of_arms,
                                                                                                         len(init_probas)))
        elif (np.array(init_probas) > 1).any():
            raise ValueError('Prior success rates for each arm have to be bound within 0 and 1')
        
        self.epsilon = epsilon
        self.init_probas = init_probas.copy()
        
    def warm_up(self,seed):
        np.random.seed(seed)
        self.random_states = np.random.random(size=self.number_of_trials)
        
        self.arms_average_rewards = self.init_probas
        
    def run_single_step(self,step,data):
        if self.random_states[step] < self.epsilon:
            selected_arm = np.random.randint(0,self.number_of_arms)
        else:
            selected_arm = self.get_max_index(self.arms_average_rewards)
            
        self.selected_arms[step] = selected_arm
        self.number_of_selections[selected_arm] += 1
        self.arms_average_rewards[selected_arm] += 1./self.number_of_selections[selected_arm]*(data[step,selected_arm]-self.arms_average_rewards[selected_arm])
        
    def run_experiment(self,seed,data):
        
        self.warm_up(seed)
        
        for step in range(self.number_of_trials):
            self.run_single_step(step,data)
