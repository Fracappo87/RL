# -*- coding: utf-8 -*-
"""
Collection of methods used to generate fake data
"""

import numpy as np
import pandas as pd

def generate_fake_bernoulli_bandits(seed, success_rates, number_of_trials):
    
    np.random.seed(seed)
    data = np.zeros([number_of_trials, len(success_rates)])

    for index, success_rates in enumerate(success_rates):
        data[:, index] = np.random.binomial(1, success_rates, number_of_trials)
    
    columns = ['amr_{}'.format(index+1) for index in range(len(success_rates))]
    return pd.DataFrame(data=data,columns=columns)