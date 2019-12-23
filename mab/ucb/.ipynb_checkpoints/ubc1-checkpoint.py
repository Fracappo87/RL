import numpy as np

from mab.base.base_mab import MultiArmedBanditProblem



import matplotlib.pyplot as plt

class BernoulliArmedBanditUCB(MultiArmedBanditProblem):

    
    
    def fit(self,data):
        
        self.warm_up(data)
        
        self.deltas = np.zeros([self.history_length-self.number_of_arms,self.number_of_arms])
        self.arms_average_rewards = np.zeros([self.history_length-self.number_of_arms,self.number_of_arms])
        
        for t in range(self.number_of_arms, self.history_length):
            average_rewards = self.arms_cumulative_rewards/self.number_of_selections
            delta = np.sqrt(2*np.log(t+1)/self.number_of_selections)
            self.arms_average_rewards[t-self.number_of_arms,:] = average_rewards
            self.deltas[t-self.number_of_arms,:] = delta
            
            ucb = average_rewards + delta            
            selected_arm = self.max_index(ucb)
            self.number_of_selections[selected_arm] += 1
            self.selected_arms[t] = selected_arm
            
            self.arms_cumulative_rewards[selected_arm] += data[t,selected_arm]
            
    def warm_up(self,data):
        self.number_of_selections = np.zeros(self.number_of_arms)
        self.arms_cumulative_rewards = np.zeros(self.number_of_arms)
        
        for t in range(self.number_of_arms):
            self.number_of_selections[t] += 1
            self.arms_cumulative_rewards[t] += data[t, t]
            self.selected_arms[t]=t
            
    def max_index(self, array):
        return np.argwhere(array==array.max()).ravel()[0]
    
