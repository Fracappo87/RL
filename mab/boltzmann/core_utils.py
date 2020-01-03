import numpy as np

def compute_boltzmann_weights(kappa,time,rewards):
    expo=rewards*kappa*time
    return np.exp(expo)/np.exp(expo).sum()

def select_arm_by_prob(weights,random_state):
    
    arm_sampling = np.random.multinomial(1, weights,size=1).ravel()
    return np.argwhere(arm_sampling==1).ravel()[0]
#    weights_cum_sum = np.cumsum(weights)
#    if (weights_cum_sum>random_state).any():
        
 #   else:
 #       return len(weights)-1
    
