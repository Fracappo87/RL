import numpy as np

from mab.base.base_mab import MultiArmedBanditProblem


class BernoulliUCB(MultiArmedBanditProblem):

    def warm_up(self,data):
        self.arms_cumulative_rewards = np.zeros(self.number_of_arms)
        
        for step in range(self.number_of_arms):
            self.number_of_selections[step] += 1
            self.arms_cumulative_rewards[step] += data[step, step]
            self.selected_arms[step]=step
        
    def run_single_step(self, step, data):
        average_rewards = self.arms_cumulative_rewards/self.number_of_selections
        delta = np.sqrt(2*np.log(step+1)/self.number_of_selections)
        self.arms_average_rewards[step-self.number_of_arms,:] = average_rewards
        self.deltas[step-self.number_of_arms,:] = delta

        ucb = average_rewards + delta  
          
        selected_arm = self.get_max_index(ucb)
        self.number_of_selections[selected_arm] += 1
        self.selected_arms[step] = selected_arm
        self.arms_cumulative_rewards[selected_arm] += data[step,selected_arm]
        
    def run_experiment(self,data):
        
        self.warm_up(data)
        
        self.deltas = np.zeros([self.number_of_trials-self.number_of_arms,self.number_of_arms])
        self.arms_average_rewards = np.zeros([self.number_of_trials-self.number_of_arms,self.number_of_arms])
        
        for step in range(self.number_of_arms, self.number_of_trials):
            self.run_single_step(step, data)

    
