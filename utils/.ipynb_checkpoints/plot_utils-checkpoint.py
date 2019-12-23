

    def plot_ucb_evolution(self,interval=20,ncols=5):
        
        nplots = self.history_length//interval
        nrows=nplots//ncols
        step=0
        for row in range(nrows):
            fig, list_of_axes = plt.subplots(1,ncols,figsize=(30,6))
            
            xpoints=list(range(1,self.number_of_arms+1))
            xlabels = ['bandit_{}'.format(index) for index in xpoints]
            colors = (1.-np.array(xpoints)/ncols)*0.8
            for col in range(ncols):
                
                rewards = self.arms_average_rewards[step*interval,:]
                deltas = self.deltas[step*interval,:]
                
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