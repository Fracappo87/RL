import numpy as np
import pandas as pd

import warnings

def sigmoid(w,x,noise=0.):
    return 1./(1+np.exp(-w.dot(x.T)-noise))

def generate_synthetic_data(ndata_per_action: int, n_actions: int, actions_rewards: np.array,
                            continuos_prms:dict, ordinal_prms: dict, categorical_prms:dict,
                            rstate: int, noise_scale:float,
                            return_inputs=False,output_info=False):
    """
    Generate synthetic data apt for contextual multi armed bandit analysis.
    Given a number of actions, and the global expected reward for each action, it builds a collection of datasets with a certain number of input features representing the context,
    and a binary target representing the reward.
    For each action, the reward is generated in the following way
        * a random vector of weights is multiplied by the input features, the result then fed into a sigmoid to produce probabilities of success
        * a reward (target=1) is then assigned according to the values given by the user (e.g. if the action reward is 10%, then all the data with probabilitie belonging to the last decile are labeled as 1)
    Once labeling has been done for each class, the data are concatenated together. An integer tag is added to specify which action has been played for each row.
    The method can produce a dataset where the inputs are
        * continuos (normally distributed)
        * ordinal
        * categorical
    A certain level of noise can be introduced by the user to make the data more or less easy to fit with machine learning methods.
    
    Parameters
    ==========
    ndata_per_action: int
        Number of times we each action is played
    n_actions: int
        Number of actions
    actions_rewards: np.array
        Expected global reward for each action
    continuos_prms:dict
        Dictionary containing details to generate continuos features. It has the form
        {'feature_name_1':{'loc':mu_1, 'scale':sigma_1},
         'feature_name_2':{'loc':mu_2, 'scale':sigma_2},
         ...}
        where 'loc' is the mean of the normal distribution, and 'scale' its standard deviation.
    ordinal_prms: dict
        Dictionary containing details to generate ordinal features. It has the form
        {'feature_name_1':{'start':start_1,'stop':stop_1,'weights':[x1,x2,...]},
         'feature_name_2':{'start':start_2,'stop':stop_2,'weights':[y1,y2,...]},
         ...}
        where 'start' and 'stop' are the end point of the range of values and 'weights' the frequency of each value belonging to the interval (the sum of then has to be 1).
    categorical_prms:dict
        Dictionary containing details to generate categorical features. It has the form
        {'feature_name_1':{'levels':[l_1,l_2...,l_k],'weights':[x1,x2,...,x_k]},
         'feature_name_2':{'levels':[d_1,d_2...,d_k],'weights':[y1,y2,...,y_k]},
         ...}
         where 'levels' is the list of categorical levels and 'weights' the frequency of each level (the sum of then has to be 1).
    rstate: int
        random seed
    noise_scale:float
        Standard deviation of the noise term (which is normally distribute around 0)
    return_inputs=False
        When True, only the context is returned
    output_info=False
        When True, generci information about the output dataset are provided to the user
    
    Returns
    =======
    input_data: pd.DataFrame
        Contextual dataset    
    """

    n_continuos = len(continuos_prms)
    n_ordinals = len(ordinal_prms)
    n_categoricals = len(categorical_prms)
    
    tot_data = ndata_per_action*n_actions   
    input_data = pd.DataFrame()
    
    np.random.seed(rstate)
    # continuos data
    for feature_name, sub_dict in continuos_prms.items():
        input_data[feature_name] = np.random.normal(loc=sub_dict['loc'], scale=sub_dict['scale'], size=tot_data)
        
    # ordinal
    for feature_name, sub_dict in ordinal_prms.items():
        input_data[feature_name] = np.random.choice(np.arange(sub_dict['start'], sub_dict['stop']),p=sub_dict['weights'],size=tot_data)
        
    # categorical
    for feature_name, sub_dict in categorical_prms.items():
        input_data[feature_name] = np.random.choice(sub_dict['levels'],p=sub_dict['weights'],size=tot_data)

    if return_inputs:
        return input_data
    
    #building the target features

    rewards = np.array([])
    action_codes = []
    for action_id in range(n_actions):
        X = input_data[action_id*ndata_per_action:(action_id+1)*ndata_per_action]
        X=pd.get_dummies(X,drop_first=True).values
        X = np.hstack((np.ones((ndata_per_action,1)),X))
        
        weights = np.random.normal(size=X.shape[1])
        noise = np.random.normal(scale=noise_scale,size=ndata_per_action)
        probabilities = sigmoid(weights,X,noise)
        cut_point = np.quantile(probabilities,1-actions_rewards[action_id])
        mask = probabilities > cut_point
        
        tmp = np.zeros(len(probabilities))
        tmp[mask] = 1
        
        rewards = np.hstack((rewards,tmp))
        action_codes += [action_id+1]*ndata_per_action
    input_data['reward'] = rewards
    input_data['reward'] =  input_data['reward'].astype(int)
    input_data['action_code'] = action_codes
    
    if output_info:
        print('Total number of data: {}'.format(tot_data))
        
        actions = list(range(1,n_actions+1))
        print('Actions played: {}'.format(actions))
                       
        avg_rew = input_data.groupby(by='action_code')[['reward']].mean()

        if (avg_rew.values==0.).any():
            warnings.warn('No reward has been produced for some of the actions: reduce noise or change the imbalance values')
        
        print('Global expected rewards per action')
        display(avg_rew)
    
    return input_data

def generate_synthetic_trial_data(ndata_per_action: int, n_actions: int, actions_rewards: np.array,
                                  continuos_prms:dict, ordinal_prms: dict, categorical_prms:dict,
                                  rstate: int, noise_scale:float, weights: list,
                                  return_inputs=False,output_info=False):
    """
    Generate synthetic data apt for simulate a contextual multi armed bandit experiment.
    Given a number of actions, and the global expected reward for each action, it builds a dataset with a certain number of input features representing the context,
    and a collection of binary columns representing the reward obtained by playing each action.
    The reward is distributed among the availabkle actions in the following way
        * a random vector of weights is multiplied by the input features, the result then fed into a sigmoid to produce probabilities of success
        * a reward (target=1) is then assigned according to the action with the highest probabilities.
    Once labeling has been done, the parameters in 'actions_rewards' are used to reduce the overall average reward of each action.
    The method can produce a dataset where the inputs are
        * continuos (normally distributed)
        * ordinal
        * categorical
    A certain level of noise can be introduced by the user to make the data more or less easy to fit with machine learning methods.
    
    Parameters
    ==========
    ndata_per_action: int
        Number of times we each action is played
    n_actions: int
        Number of actions
    actions_rewards: np.array
        Expected global reward for each action
    continuos_prms:dict
        Dictionary containing details to generate continuos features. It has the form
        {'feature_name_1':{'loc':mu_1, 'scale':sigma_1},
         'feature_name_2':{'loc':mu_2, 'scale':sigma_2},
         ...}
        where 'loc' is the mean of the normal distribution, and 'scale' its standard deviation.
    ordinal_prms: dict
        Dictionary containing details to generate ordinal features. It has the form
        {'feature_name_1':{'start':start_1,'stop':stop_1,'weights':[x1,x2,...]},
         'feature_name_2':{'start':start_2,'stop':stop_2,'weights':[y1,y2,...]},
         ...}
        where 'start' and 'stop' are the end point of the range of values and 'weights' the frequency of each value belonging to the interval (the sum of then has to be 1).
    categorical_prms:dict
        Dictionary containing details to generate categorical features. It has the form
        {'feature_name_1':{'levels':[l_1,l_2...,l_k],'weights':[x1,x2,...,x_k]},
         'feature_name_2':{'levels':[d_1,d_2...,d_k],'weights':[y1,y2,...,y_k]},
         ...}
         where 'levels' is the list of categorical levels and 'weights' the frequency of each level (the sum of then has to be 1).
    rstate: int
        random seed
    noise_scale:float
        Standard deviation of the noise term (which is normally distribute around 0)
    return_inputs=False
        When True, only the context is returned
    output_info=False
        When True, generci information about the output dataset are provided to the user
    
    Returns
    =======
    input_data: pd.DataFrame
        Contextual dataset    
    """

    n_continuos = len(continuos_prms)
    n_ordinals = len(ordinal_prms)
    n_categoricals = len(categorical_prms)
    
    tot_data = ndata_per_action
    input_data = pd.DataFrame()
    
    np.random.seed(rstate)
    # continuos data
    for feature_name, sub_dict in continuos_prms.items():
        input_data[feature_name] = np.random.normal(loc=sub_dict['loc'], scale=sub_dict['scale'], size=tot_data)
        
    # ordinal
    for feature_name, sub_dict in ordinal_prms.items():
        input_data[feature_name] = np.random.choice(np.arange(sub_dict['start'], sub_dict['stop']),p=sub_dict['weights'],size=tot_data)
        
    # categorical
    for feature_name, sub_dict in categorical_prms.items():
        input_data[feature_name] = np.random.choice(sub_dict['levels'],p=sub_dict['weights'],size=tot_data)

    if return_inputs:
        return input_data
    
    #building the target features

    action_codes = []
    features = list(continuos_prms.keys())+list(ordinal_prms.keys())+list(categorical_prms.keys())
    for action_id in range(n_actions):
        X=pd.get_dummies(input_data[features],drop_first=True).values
        X = np.hstack((np.ones((ndata_per_action,1)),X))
        
        weights_vec = np.array(weights[action_id])
        noise = np.random.normal(scale=noise_scale,size=ndata_per_action)
        probabilities = sigmoid(weights_vec,X,noise)
        input_data['action_prob_{}'.format(action_id+1)] = probabilities
        input_data['action_{}_reward'.format(action_id+1)] = 0.
        
    action_probs = ['action_prob_{}'.format(action_id) for action_id in range(1,n_actions+1)]
    loc_reward = np.argmax(input_data[action_probs].values,axis=1)
    for idx,loc in enumerate(loc_reward):
        input_data.loc[idx,'action_{}_reward'.format(loc+1)] = 1.
        
    for action_id in range(n_actions):  
        probabilities = input_data['action_prob_{}'.format(action_id+1)].values
        cut_point = np.quantile(probabilities,1-actions_rewards[action_id])
        mask = probabilities <= cut_point
        input_data.loc[mask,'action_{}_reward'.format(action_id+1)] = 0.
        
    input_data.drop(columns=action_probs,inplace=True)
        
    return input_data

def generate_experimental_dataset(sizes,
                                  list_of_class_weights,
                                  list_of_continuos_dicts,
                                  list_of_ordinal_dicts,
                                  list_of_categorical_dicts,
                                  list_of_noise_scales,
                                  list_of_model_weights,
                                  seed,
                                  output_info=False):
    
    list_of_frames = []
    
    g=0
    for n,cw,ct_prm,ord_prm,catg_prm,noise_scale,weights in zip(sizes,
                                                                list_of_class_weights,
                                                                list_of_continuos_dicts,
                                                                list_of_ordinal_dicts,
                                                                list_of_categorical_dicts,
                                                                list_of_noise_scales,
                                                                list_of_model_weights):
        g+=1
        dataset = generate_synthetic_trial_data(n,len(cw),cw,
                                                ct_prm,ord_prm,catg_prm,seed,noise_scale,weights=weights)
        if output_info:
            print('Group {}'.format(g))
            action_cols = ['action_{}_reward'.format(idx) for idx in range(1,len(cw)+1)]
            display(dataset[action_cols].mean())
        list_of_frames.append(dataset)
        
    final_frame = pd.concat(list_of_frames)
    final_frame.reset_index(drop=True,inplace=True)
    
    if output_info:
        print('Overall')
        display(final_frame[action_cols].mean())
        
    final_frame = final_frame.sample(frac=1.)
    
    return final_frame

