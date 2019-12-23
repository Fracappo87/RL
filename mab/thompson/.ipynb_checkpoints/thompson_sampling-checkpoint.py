import numpy as np

from mab.base.base_mab import MultiArmedBanditProblem

class BernoulliTS(MultiArmedBanditProblem):
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
        self.init_probas = init_probas
        
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
            
            
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bandit)

        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1 - r)

        return i