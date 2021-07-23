import sys 
import numpy as np 
import math 
import matplotlib.pyplot as plt  

def sample_state_noise(): 
    epsilon_k = np.random.normal(0, 1)
    return epsilon_k

def sample_obs_noise():  
    nu_k = np.random.normal(0, math.sqrt(1/2))
    return nu_k 
    
def propagate_system_state(x):
    return -1*x + sample_state_noise()  

def propagate_system_obs(x): 
    return math.sqrt(pow(x,2) + 1) + sample_obs_noise() 

def generate_ground_truth_data(iters, x_curr):
    obs_dataset = []
    state_dataset = [x_curr] 
    for _ in range(iters):  
        x_next = propagate_system_state(x_curr) 
        obs = propagate_system_obs(x_next)  
        obs_dataset.append(obs) 
        state_dataset.append(x_next)
        x_curr = x_next 
    return np.array(obs_dataset), np.array(state_dataset)


class EKF():
    def __init__(self, state_0, var_0, actual_obs):    
        self.mean_k_k = state_0   
        self.cov_k_k = np.diag([.1, .1]) 
        self.actual_obs = actual_obs
        self.A = np.zeros((2, 2))

    def propagate_mean(self): 
        #print(self.mean_k_k)
        #print(self.A)
        self.A[0] = np.array([self.mean_k_k[1].item(), self.mean_k_k[0].item()])
        self.A[1] = np.array([0, 1]) 
        mean_knext_k = self.A@self.mean_k_k
        return mean_knext_k.astype('float64') #(2, 1)

    def propagate_covariance(self): 
        R = np.diag([2, 0.]) 
        cov_knext_k = self.A@self.cov_k_k@self.A.T 
        return cov_knext_k + R #(2, 2)

    def compute_jac_obs(self, mean_k_next_k): 
        C = np.zeros((1, 2))   
        elem = mean_k_next_k[0] / math.sqrt(pow(mean_k_next_k[0], 2) + 1)
        C[0, :] = np.array([elem, 0.], dtype='float64') 
        return C

    def compute_obs_estimate(self, C, mean_k_next_k): 
        return np.matmul(C.reshape(1, 2), mean_k_next_k.reshape(2, 1))
    
    def compute_fake_obs(self, obs_actual, obs_estimate, C, mean_k_next_k): 
        y_knext_prime = obs_actual - obs_estimate + C@mean_k_next_k
        return y_knext_prime

    def compute_kalman_gain(self, cov_knext_k, C):  
        information = 1. / (C@cov_knext_k@C.T + 1/2)
        K = cov_knext_k@C.T@information
        return K 

    def update(self, y_knext_prime, mean_knext_k, kalman_gain, C, cov_knext_k):
        
        innovation = y_knext_prime - np.matmul(C,mean_knext_k)
        mean_updated = mean_knext_k + np.matmul(kalman_gain, innovation)  
        self.mean_k_k = mean_updated 
        cov_updated = (np.identity(2) - kalman_gain@C)@cov_knext_k
        self.cov_k_k = cov_updated  
    
        return mean_updated, cov_updated

    def one_iteration(self, i): 
        mean_knext_k = self.propagate_mean()
        #print(mean_knext_k.shape)
        #sys.exit() 
        cov_knext_k = self.propagate_covariance() 
        #print(cov_knext_k.shape) 
        C = self.compute_jac_obs(mean_knext_k)
        #print(C.shape) 
        obs_estimate = self.compute_obs_estimate(C, mean_knext_k)
        #print(obs_estimate)
        y_knext_prime = self.compute_fake_obs(self.actual_obs[i], obs_estimate, C, mean_knext_k)
        K = self.compute_kalman_gain(cov_knext_k, C)  
        state_estimate_i, covariance_estimate_i = self.update(y_knext_prime, mean_knext_k, K, C, cov_knext_k)
        print(covariance_estimate_i) 
        #sys.exit() 
        return state_estimate_i, covariance_estimate_i  
        
def main(ekf, iters):  
    x_estimates = []
    a_estimates = [] 
    a_estimate_cov = [] 
    for i in range(iters): 
        x_estimate_i, covariance_estimate_i = ekf.one_iteration(i)
        x_estimates.append(x_estimate_i[0])
        a_estimates.append(x_estimate_i[1])
        a_estimate_cov.append(covariance_estimate_i[0, 0])
    return np.array(x_estimates), np.array(a_estimates), np.array(a_estimate_cov)

def plot_results(ground_truth_data, state_estimates, a_estimates, a_estimates_cov):         
    fig, axs = plt.subplots(2)
    fig.suptitle("Estimated States vs. Ground Truth Values")
    axs[0].scatter(np.arange(ground_truth_data.shape[0]), ground_truth_data, color="blue", label="Ground Truth State")
    axs[0].scatter(np.arange(100), state_estimates, color="green", label="Estimated State") 
    axs[1].plot(np.array([1.] * 100), color="blue", label="Expected a value")
    axs[1].scatter(np.arange(100), a_estimates + a_estimates_cov.reshape(100, 1), color="green", label="Estimated a value")
    for a in axs: 
        a.legend() 
        a.grid() 
    plt.show() 
    
    
if __name__=="__main__":  

    iters = 100
    x_0 = np.random.normal(1., math.sqrt(2))   
    x_a = x_0
    actual_obs, actual_state = generate_ground_truth_data(iters, x_0) 

    x_0 = 2
    a_0 = 1. 
    state_0 = np.array([x_0, a_0]).reshape(2, 1)
    initial_cov = np.diag([2, 2])
    ekf = EKF(state_0, initial_cov, actual_obs)
    x_estimates, a_estimates, a_estimates_cov = main(ekf, iters)
    #plot_results(actual_state, x_estimates, a_estimates, a_estimates_cov)
