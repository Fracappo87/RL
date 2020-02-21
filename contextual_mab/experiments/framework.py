import numpy as np
import pandas as pd

from scipy.stats import f_oneway

from tqdm import tqdm

class MABFramework(object):
    available_strategies = ['static-one-fits-all', 'dynamic-one-fits-all','contextual-one-fits-one']
                            
    def __init__(self,strategy,n_actions,
                 rstate=42,
                 static_min_steps = 1,
                 alphas=[],betas=[],
                 modelling_approach=None,
                 modelling_approach_pms=None
                ):
        if n_actions <= 0:
            raise ValueError('Invalid number of actions, it should be a positive number')
        elif strategy not in MABFramework.available_strategies:
            raise ValueError('Unrecognised MAB strategy, available strategies are {}'.format(MABFramework.available_strategies))
        elif (strategy=='dynamic-one-fits-all') and ((n_actions != len(alphas)) or (n_actions != len(betas))):
            raise ValueError('Cannot run a dynamic strategy without specified priors')
        elif (strategy=='contextual-one-fits-all') and ((modelling_approach is None) or (modelling_approach_pms is None)):
            raise ValueError('Cannot run a contextual strategy if a modelling approach and parameters are not provided')        

        
        self.strategy = strategy
        self.n_actions = n_actions
        
        if self.strategy == 'static-one-fits-all':
            self.static_status = None
            self.static_min_steps = static_min_steps
        elif self.strategy == 'dynamic-one-fits-all':
            self.alphas = alphas
            self.betas = betas
            self.thompson_pms = pd.DataFrame(columns=['alphas','betas'])
        else:        
            self.predictive_units = [modelling_approach(**modelling_approach_pms)]*self.n_actions
        
        self.current_data = pd.DataFrame()
        np.random.seed(rstate)
            
    def append_data(self, new_data_batch):
        
        if not len(self.current_data):
            self.current_data = pd.concat([self.current_data,new_data_batch])
        else:
            column_check = self.current_data.columns.intersection(new_data_batch.columns)
            if not len(column_check):
                raise ValueError('The new data batch has not the same column names as current data, stopping experiment')
            else:
                self.current_data = pd.concat([self.current_data,new_data_batch])
                
    def observe_rewards(self, new_data_batch, reward_columns):
        
        nrows=len(new_data_batch)
        new_data_batch['action_code'] = self.best_actions
        self.append_data(new_data_batch.drop(columns = reward_columns))
        
    def warm_up(self,incoming_data_batch):
        if self.strategy == 'static-one-fits-all':
            self.best_actions = np.random.choice(range(self.n_actions),size=len(incoming_data_batch))
        elif self.strategy == 'dynamic-one-fits-all':
            arms_average_rewards = np.random.beta(self.alphas,self.betas,[len(incoming_data_batch),self.n_actions])
            self.best_actions = np.argmax(arms_average_rewards,axis=1).tolist()
        elif self.strategy == 'contextual-one-fits-one':
            self.best_actions = np.random.choice(range(self.n_actions),size=len(incoming_data_batch))
        
    def apply_decision_policy(self, incoming_data_batch,step):
        
        if not(len(self.current_data)):
            self.warm_up(incoming_data_batch)
        else:
            if self.strategy == 'static-one-fits-all':
                if  self.static_status != 'converged':
                    self.static_one_fits_all('action_code','reward',incoming_data_batch,step)
            elif self.strategy == 'dynamic-one-fits-all':
                self.dynamic_one_fits_all('action_code','reward',incoming_data_batch,step)
            elif self.strategy == 'contextual-one-fits-one':
                self.contextual_one_fits_one('action_code','reward',incoming_data_batch,step)
            
    def static_one_fits_all(self, actions_column, reward_column, incoming_data_batch, step):
        
        n_choices = len(incoming_data_batch)
        
        grouped_dataset = self.current_data.groupby(by=[actions_column])[reward_column].agg([('n_trials','count'),('p','mean')])
        grouped_dataset['std_err'] = np.sqrt(grouped_dataset['p']*(1-grouped_dataset['p']))/np.sqrt(grouped_dataset['n_trials'])
        
        list_of_samples = []
        for idx in grouped_dataset.index:
            list_of_samples.append(np.random.normal(loc=grouped_dataset.loc[idx,'p'],scale=grouped_dataset.loc[idx,'std_err'],size=grouped_dataset.loc[idx,'n_trials']))
            
        pvalue = f_oneway(*list_of_samples)[1]
        if pvalue <= .05 and step>self.static_min_steps:
            self.static_status = 'converged'
            self.best_actions = [np.argmax(grouped_dataset['p'].values)]*n_choices
            
    def dynamic_one_fits_all(self, actions_column, reward_column, incoming_data_batch, step):
        
        n_choices = len(incoming_data_batch)
        
        grouped = self.current_data.groupby(by=[actions_column])[reward_column]
        self.alphas = grouped.sum().values.ravel()
        mask = self.alphas == 0.
        self.alphas[mask] = 1.        
        self.betas = grouped.count().values.ravel()-self.alphas
        arms_average_rewards = np.random.beta(self.alphas,self.betas,[n_choices,self.n_actions])
    
        self.best_actions = np.argmax(arms_average_rewards,axis=1).tolist()
        self.thompson_pms.loc[step,'alphas'] = self.alphas
        self.thompson_pms.loc[step,'betas'] = self.betas
        
    def contextual_one_fits_one(self, actions_column, reward_column, incoming_data_batch, step):
        
        predictors = self.current_data.drop(columns=['reward','action_code']).columns
        sampled_probs = np.zeros([len(incoming_data_batch),self.n_actions])
        
        for action_id in range(self.n_actions):
            subset = self.current_data[self.current_data['action_code']==action_id]
            X = pd.get_dummies(subset[predictors],drop_first=True).values
            y = subset['reward'].values
            self.predictive_units[action_id].fit(X,y)
        
            X_in = pd.get_dummies(incoming_data_batch[predictors],drop_first=True).values
            predicted_probs = self.predictive_units[action_id].predict(X_in)
            sampling_indices = np.random.choice(range(predicted_probs.shape[1]),size=len(predicted_probs))
            sampled_probs[:,action_id] = np.array([predicted_probs[row,sampling_indices[row]] for row in range(predicted_probs.shape[0])])

        self.best_actions = np.argmax(sampled_probs,axis=1).tolist()
        
def run_experiment(experiment_data, batch_size, exp_class, exp_pms, return_exp_obj=False):
    
    n_steps = experiment_data.shape[0]//batch_size
    uneven = experiment_data.shape[0]%batch_size
    
    exp_obj = exp_class(**exp_pms)
    action_cols = [column for column in experiment_data.columns if 'action' in column]
    for step in tqdm(range(n_steps)):
        incoming_data = experiment_data[step*batch_size:(step+1)*batch_size].copy()
        exp_obj.apply_decision_policy(incoming_data,step)
        rewards = incoming_data[action_cols].values
        incoming_data['reward'] = [rewards[idx,exp_obj.best_actions[idx]] for idx in range(len(incoming_data))]
        incoming_data['action_code'] = exp_obj.best_actions
        exp_obj.append_data(incoming_data.drop(columns = action_cols))
    
    if uneven:
        incoming_data = experiment_data[(step+1)*batch_size:].copy()
        exp_obj.apply_decision_policy(incoming_data,step)
        rewards = incoming_data[action_cols].values
        incoming_data['reward'] = [rewards[idx,exp_obj.best_actions[idx]] for idx in range(len(incoming_data))]
        incoming_data['action_code'] = exp_obj.best_actions[:len(incoming_data)]
        exp_obj.append_data(incoming_data.drop(columns = action_cols))
    
    if return_exp_obj:
        return exp_obj
    else:
        return exp_obj.current_data['reward'].mean()