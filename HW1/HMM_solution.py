import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.num_obs = Observations
        self.T = Transition
        self.M = Emission
        self.pi_1 = Initial_distribution 
    
    def forward(self): 

        a_0 = self.pi_1[0] * self.M[0][self.num_obs[0]], self.pi_1[1] * self.M[1][self.num_obs[0]] 
        alphas = [a_0]
        alpha = a_0 
        for i in range(1, 20):  
            obs = self.num_obs[i]    
            alpha = self.M[:, obs] * np.matmul(alpha, self.T)   
            alphas.append(alpha)
        alphas = np.matrix(alphas)  
        return alphas
        
    def backward(self):        
        betas = [np.array((1, 1))]   
        beta = betas[0]
        for i in range(18, -1, -1): 
            obs = self.num_obs[i+1]
            beta = np.matmul(self.T, (beta * self.M[:, obs])) 
            betas.append(beta) 
        betas = np.flip(np.matrix(betas))  
        return betas 
    
    def gamma_comp(self, alpha, beta):
        
        etas_t = np.sum(alpha[-1])  
        prod = np.multiply(beta, alpha) 
        gammas = np.divide(prod, etas_t) 

        return gammas 
    
    def xi_comp(self, alpha, beta, gamma):
    
        xis = [] 
        for k in range(19): 
            obs_k = self.num_obs[k+1] 
            mat_k = np.zeros((2, 2)) 
            for s in range(2): 
                for s_ in range(2):   
                    alpha_val = alpha[k][:, s] 
                    beta_val = beta[k+1][:, s_] 
                    val = alpha_val*self.T[s, s_]*self.M[s_, obs_k]*beta_val 
                    mat_k[s, s_] = val   
            xis.append(np.divide(mat_k, np.sum(mat_k)))  

        return np.array(xis) 

    def update(self, alpha, beta, gamma, xi):
        
        num = np.sum(xi, axis=0)   
        denom = np.sum(gamma[0:len(gamma)-1], axis=0).reshape(-1, 1)
        T_prime = np.array((num / denom)).reshape(2, 2) 
        
        gamma_sum = np.sum(gamma, axis=0)  
        observation_new = np.zeros((2, 3)) 
        observation_new[:, 0] = np.sum(gamma[np.where(self.num_obs == 0)], axis=0)  
        observation_new[:, 1] = np.sum(gamma[np.where(self.num_obs == 1)], axis=0) 
        observation_new[:, 2] = np.sum(gamma[np.where(self.num_obs == 2)], axis=0) 
        M_prime = np.array((observation_new / gamma_sum.reshape(2, 1))).reshape(2,3) 

        new_init_state = np.zeros((2,)) 
        new_init_state[0] = gamma[0][:, 0] 
        new_init_state[1] = gamma[0][:, 1] 
        return T_prime, M_prime, new_init_state
    
    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state): 
        self.M = M_prime
        self.T = T_prime 
        
        self.pi_1 = new_init_state 
        alphas_new = self.forward() 
        P_prime = np.sum(alphas_new[-1])
        P_original = np.sum(alpha[-1]) 

        return P_original, P_prime
  
if __name__ == "__main__":  

    transition = np.array(((0.5, 0.5), (0.5, 0.5)))
    obs = np.array(((0.4, 0.1, 0.5), (0.1, 0.5, 0.4))) 
    pi = np.array((.5, .5)) 
    print("x", pi.shape)
    nums_obs = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])
    
    hmm = HMM(nums_obs, transition, obs, pi) 
    alphas = hmm.forward() 
    betas = hmm.backward()  
    gammas = hmm.gamma_comp(alphas, betas)  
    xis = hmm.xi_comp(alphas, betas, gammas)
    T_new, M_new, pi_new = hmm.update(alphas, betas, gammas, xis)
    print("Transition Matrix New: {}".format(T_new))
    print("Observation Matrix New: {}".format(M_new))
    print("New Initial State: {}".format(pi_new))
    P_orig, P_prime = hmm.trajectory_probability(alphas, betas, T_new, M_new, pi_new)
    print(P_orig, P_prime) 