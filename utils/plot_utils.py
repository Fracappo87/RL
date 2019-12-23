import numpy as np

from scipy.stats import beta

import matplotlib.pyplot as plt

def plot_ucb_evolution(ucb_learner,interval=20,ncols=5):
    
    nplots = ucb_learner.number_of_trials//interval
    nrows=nplots//ncols
    step=0
    for row in range(nrows):
        fig, list_of_axes = plt.subplots(1,ncols,figsize=(30,6))
        
        xpoints=list(range(1,ucb_learner.number_of_arms+1))
        xlabels = ['bandit_{}'.format(index) for index in xpoints]

        for col in range(ncols):
            
            rewards = ucb_learner.arms_average_rewards[step*interval,:]
            deltas = ucb_learner.deltas[step*interval,:]
            
            ax=list_of_axes[col]
            y_min = (rewards-deltas).min()*(1+.2)
            y_max = (rewards+deltas).max()*(1+.2)
            xticks = np.array(xpoints)+.4
            for i,point in enumerate(xpoints):
                seg = [point,point+0.8]
                rew = [rewards[i]]*2
                up = [rewards[i]+deltas[i]]*2
                down = [rewards[i]-deltas[i]]*2
                ax.plot(seg,rew)
                ax.set_ylabel('reward')
                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                ax.fill_between(seg, down, up, color='gray', alpha=0.2)
                ax.set_ylim(y_min,y_max)
            ax.plot(xpoints[np.argmax(rewards+deltas)]+0.4,(rewards+deltas).max(),marker='o',markersize=6,color='k',label='Max UCB '+str(round((rewards+deltas).max(),3)))
            ax.legend()
            ax.set_title('{} iterations'.format(step*interval),)
            step+=1
        plt.tight_layout()
        plt.show()
        
def plot_ts_evolution(ts_learner,interval=20,ncols=5):
    
    nplots = ts_learner.number_of_trials//interval
    nrows=nplots//ncols
    step=0
    npoints=100
    for row in range(nrows):
        fig, list_of_axes = plt.subplots(1,ncols,figsize=(30,6))
        xlabels = ['bandit_{}'.format(index) for index in range(1,ts_learner.number_of_arms+1)]

        for col in range(ncols):
            
            alphas = ts_learner.step_alphas[step*interval,:]
            betas = ts_learner.step_betas[step*interval,:]
            
            ax=list_of_axes[col]
            for a,b,label in zip(alphas,betas,xlabels):
                x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), npoints)
                curve = beta.pdf(x, a, b)
                ax.plot(x, curve, lw=2, label=label)
                ax.fill_between(x, [curve.min()]*npoints, beta.pdf(x, a, b), alpha=0.2)
                ax.set_ylabel('P(reward)',fontsize=14)
                ax.set_xlabel('reward',fontsize=14)

            ax.legend(fontsize=12)
            ax.set_title('{} iterations'.format(step*interval),fontsize=20)
            step+=1
        plt.tight_layout()
        plt.show()