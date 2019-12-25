import numpy as np

import inspect
import sys

from utils.data_gen import generate_fake_bernoulli_bandits

def compute_cumulative_regret(learner,rewards):
    max_rew = np.max(rewards)
    return np.cumsum([max_rew-rewards[int(arm)] for arm in learner.selected_arms])

def regret_analysis(n_experiments, number_of_trials, success_rates, learner_class, **kwargs):
    
    kwargs['number_of_arms'] = len(success_rates)
    kwargs['number_of_trials'] = number_of_trials
    
    method_input = {}        
    samples=np.zeros((n_experiments,number_of_trials))
    samples_cum_rew = np.zeros((n_experiments,number_of_trials))
    for exp in range(n_experiments):
        toy_dataset = generate_fake_bernoulli_bandits(exp,success_rates,number_of_trials)
        learner=learner_class(**kwargs)

        method_input['data']=toy_dataset.values
        if 'seed' in inspect.signature(learner.run_experiment).parameters.keys():
            method_input['seed']=exp
        learner.run_experiment(**method_input)
        
        step_cumulative_regret = compute_cumulative_regret(learner,success_rates)
        samples[exp]=step_cumulative_regret
        samples_cum_rew[exp]=learner.compute_step_cumulative_reward(toy_dataset.values)
        
        if  exp%10:
          sys.stdout.write('\r')
          sys.stdout.write("[%-20s] %d%%" % ('='*(exp//(n_experiments//20)), exp*100//n_experiments))
          sys.stdout.flush()
    
    expected_vals = samples.mean(axis=0)
    expected_std_err = samples.std(axis=0)/np.sqrt(n_experiments)
    
    expected_cum_rew = samples_cum_rew.mean(axis=0)
    expected_cum_rew_std_err = samples_cum_rew.std(axis=0)/np.sqrt(n_experiments)
    return expected_vals,expected_std_err,expected_cum_rew,expected_cum_rew_std_err