import numpy as np
import pandas as pd

import pymc3 as pm

class BayesianLogisticRegression(object):
    def __init__(self,n_samples, n_chains, predictors,tune=2000,check_prog=True):
        
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.predictors = predictors
        self.tune = tune
        self.check_prog = check_prog
        
    def plot_traces(self, retain=0):
        '''
        Convenience function:
        Plot traces with overlaid means and values
        '''

        ax = pm.traceplot(self.trace[-retain:])

        for i, mn in enumerate(pm.summary(self.trace[-retain:])['mean']):
            ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                        ,xytext=(5,10), textcoords='offset points', rotation=90
                        ,va='bottom', fontsize='large', color='#AA0022')
            
    def fit(self,X,y):
        with pm.Model() as logistic_model:
            pm.glm.GLM(X,y,
                       labels=self.predictors,
                       family=pm.glm.families.Binomial())
            self.trace = pm.sample(draws = self.n_samples,
                                   chains = self.n_chains,
                                   tune=self.tune,
                                   init='adapt_diag',
                                   progressbar=self.check_prog)
          
    def predict(self,X):
        data = pd.DataFrame(data=X,columns=self.predictors)
        posterior_probs_series = self.posterior_predictions(data)
        posterior_probs_array = np.zeros([len(X),self.n_samples*self.n_chains])
        for idx in posterior_probs_series.index:
            posterior_probs_array[idx,:] = posterior_probs_series.loc[idx]
            
        return posterior_probs_array
            
    def posterior_predictions(self, data):
        data['Intercept'] = 1.
        data['posterior'] = data.apply(lambda row: 1./(1+np.exp(-sum([row[predictor]*self.trace[predictor] for predictor in self.predictors+['Intercept']]))),axis=1)

        return data['posterior']