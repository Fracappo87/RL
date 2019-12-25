import numpy as np

from mab.base.base_mab import MultiArmedBanditProblem

class BernoulliTS(MultiArmedBanditProblem):
    def __init__(self,number_of_arms, number_of_trials,prior_alphas,prior_betas):
        super(BernoulliTS,self).__init__(number_of_arms, number_of_trials)
        
        if (prior_alphas < 0).any():
            raise ValueError('Prior value of alpha parameters has to be positive or zero ')
        elif (prior_betas < 0).any():
            raise ValueError('Prior value of beta parameters has to be positive or zero ') 
        elif len(prior_alphas)!=len(prior_betas)!=number_of_arms:
            raise ValueError('Prior alpha and beta vectors must have the same length')
            
        self.alphas = prior_alphas.copy()
        self.betas = prior_betas.copy()
        
    def warm_up(self,seed):
        np.random.seed(seed)
        
        self.step_alphas = np.zeros((self.number_of_trials,self.number_of_arms))
        self.step_betas = np.zeros((self.number_of_trials,self.number_of_arms))
        
    def run_single_step(self, step, data):
        arms_average_rewards = np.random.beta(self.alphas,self.betas)
        selected_arm = self.get_max_index(arms_average_rewards)
        self.selected_arms[step] = selected_arm
        self.number_of_selections[selected_arm] += 1

        self.alphas[selected_arm] += data[step,selected_arm]
        self.betas[selected_arm] += (1 - data[step,selected_arm])
        
        self.step_alphas[step] = self.alphas
        self.step_betas[step] = self.betas
         
    def run_experiment(self,seed,data):
        
        self.warm_up(seed)
        
        for step in range(self.number_of_trials):
            self.run_single_step(step,data)
            
            