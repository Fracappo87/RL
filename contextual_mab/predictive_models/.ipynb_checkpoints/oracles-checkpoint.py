import numpy as np
from tqdm import tqdm

from sklearn.utils import resample

class BootstrapOracle(object):
    def __init__(self,
                 n_bootstrap,
                 learner_class, learner_class_params, tuning_grid=None,
                 check_prog=True):
        self.n_oracles = n_bootstrap
        self.learner_class = learner_class
        self.learner_class_params = learner_class_params
        
        if check_prog:
            self.progress = trange(self.n_oracles)
        else:
            self.progress = range(self.n_oracles)

        if tuning_grid is None:
            self.run_tuning = False
        else:
            self.run_tuning = True
        
        
    def fit(self,X,y,rstate=0):
        
        self.oracles =[]
        if self.run_tuning:
            tuning(self)
        
        for idx in self.progress:
            sampled_X,sampled_y  = resample(X,y,random_state=idx)
                        
            learner = self.learner_class(**self.learner_class_params)
            learner.fit(sampled_X,sampled_y)
            self.oracles.append(learner)
    
    def predict(self,X):
        bootstrap_prediction = np.zeros((X.shape[0],self.n_oracles))
        
        for idx in self.progress:
            bootstrap_prediction[:,idx] = self.oracles[idx].predict_proba(X)[:,1]
            
        return bootstrap_prediction
        
    def tuning(self):
        pass