import numpy as np

from mab.base.base_mab import MultiArmedBanditProblem
from mab.boltzmann.core_utils import compute_boltzmann_weights
from mab.boltzmann.core_utils import select_arm_by_prob


class BernoulliSoftmaxAnnealing(MultiArmedBanditProblem):
    def __init__(self,number_of_arms, number_of_trials,kappa,init_probas):
        super(BernoulliSoftmaxAnnealing,self).__init__(number_of_arms, number_of_trials)
        
        if kappa < 0:
            raise ValueError('Kb parameter must be positive')
        elif len(init_probas) != number_of_arms:
            raise ValueError('Number of arms ({}) does not match the number of given priors ({})'.format(number_of_arms,
                                                                                                         len(init_probas)))
        elif (np.array(init_probas) > 1).any():
            raise ValueError('Prior success rates for each arm have to be bound within 0 and 1')
        
        self.kappa = kappa
        self.init_probas = init_probas.copy()
        
    def warm_up(self,seed):
        np.random.seed(seed)
        self.random_states = np.random.random(size=self.number_of_trials)
        
        self.arms_average_rewards = self.init_probas
        
    def run_single_step(self,step,data):
        
        weights = compute_boltzmann_weights(self.kappa, step+1,self.arms_average_rewards)
        selected_arm = select_arm_by_prob(weights,self.random_states[step])
            
        self.selected_arms[step] = selected_arm
        self.number_of_selections[selected_arm] += 1
        self.arms_average_rewards[selected_arm] += 1./self.number_of_selections[selected_arm]*(data[step,selected_arm]-self.arms_average_rewards[selected_arm])
        
    def run_experiment(self,seed,data):
        
        self.warm_up(seed)
        
        for step in range(self.number_of_trials):
            self.run_single_step(step,data)
            
