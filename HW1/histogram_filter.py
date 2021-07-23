import numpy as np 

class HistogramFilter(object): 
            
    def sense(self, cmap, belief, observation):
        nrows, ncols = belief.shape 
        posterior = np.zeros((nrows, ncols)) 
        for i in range(nrows): 
            for j in range(ncols): 
                posterior[i, j] = .9 if cmap[i,j] == observation else .1   
        q = posterior * belief 
        q = q/np.sum(q) 

        return q 
     
    def take_action(self, belief, action): 
        nrows, ncols = belief.shape 
        prior = np.zeros((nrows, ncols))
        for i in range(nrows): 
            for j in range(ncols):
                #boundary conditions
                row_mv = i - action[1]
                col_mv = j + action[0]
                if row_mv < nrows and row_mv >= 0 and col_mv < ncols and col_mv >= 0:  
                    prior[i - action[1], j + action[0]] += .9*belief[i, j] 
                    prior[i, j] += .1*belief[i, j]
                else: 
                    prior[i, j] += belief[i, j]
    
        return prior 

 
    def histogram_filter(self, cmap, belief, action, observation): 
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3

        ### Your Algorithm goes Below.
        '''  
        belief = self.take_action(belief, action)
        belief = self.sense(cmap, belief, observation)  

        return belief 