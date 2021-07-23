import sys 
#print(sys.version)
import numpy as np 
import math 
import matplotlib.pyplot as plt  

"""
represent the state as: [x, a] 
trivial dynamics for a: a_{k+1} = a_k + epsilon_k 
[x_{k+1}, a_{k+1}] = [ax_k, a] = A*[x_k,a_k]
A = [
    [a_k , x_k], 
    [0, 1]
]

y_k = [x, a] / math.sqrt()
dynamics: 
    states: 
    obs: 
"""

def sample_state_noise(): 
    epsilon_k = np.random.normal(0, 1)
    return epsilon_k

def sample_obs_noise():  
    nu_k = np.random.normal(0, math.sqrt(1/2))
    return nu_k 
    
def propagate_system_state(x):
    return -1.*x + sample_state_noise()  

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
        self.cov_k_k = var_0                
        self.A = np.zeros((2, 2))
        self.A[0] = np.array([self.mean_k_k[1].item(), self.mean_k_k[0].item()])
        self.A[1] = np.array([0, 1]) 
        self.actual_obs = actual_obs

    def propagate_mean(self): 
        #print(self.mean_k_k)
        #print(self.A)
        mean_knext_k = self.A@self.mean_k_k
        print(mean_knext_k)
        return mean_knext_k.astype('float64') #(2, 1)

    def propagate_covariance(self): 
        R = np.diag([1, 1]) 
        cov_knext_k = self.A@self.cov_k_k@self.A.transpose() 
        return cov_knext_k + R #(2, 2)

    def compute_jac_obs(self, mean_k_next_k): 
        elem = mean_k_next_k[0] / math.sqrt(pow(mean_k_next_k[0], 2) + 1)
        C = np.diag([elem, 1.]).reshape(2,2)
        return C

    def compute_obs_estimate(self, C, mean_k_next_k): 
        return np.matmul(C.reshape(2, 2), mean_k_next_k.reshape(2, 1))
    
    def compute_fake_obs(self, obs_actual, obs_estimate, C, mean_k_next_k): 
        y_knext_prime = np.array([obs_actual, obs_actual/mean_k_next_k[0]]).reshape(2,1) - obs_estimate 
        return y_knext_prime
    
    def compute_kalman_gain(self, cov_knext_k, C):  
        print((C@cov_knext_k@C.T).shape)
        information = np.linalg.inv(C@cov_knext_k@C.T + np.diag([1/2, 1/2])) 
        K = cov_knext_k@C.T@information
        return K 

    def update(self, y_knext_prime, mean_knext_k, kalman_gain, C):
        
        innovation = y_knext_prime - np.matmul(C,mean_knext_k)
        mean_updated = mean_knext_k + np.matmul(kalman_gain, innovation)  
        self.mean_k_k = mean_updated 
        cov_updated = (np.identity(2) - kalman_gain@C)@self.cov_k_k
        self.cov_k_k = cov_updated 
        print(mean_updated)
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
        state_estimate_i, covariance_estimate_i = self.update(y_knext_prime, mean_knext_k, K, C)
        return state_estimate_i, covariance_estimate_i  
        
def main(ekf, iters): 
    state_estimates = np.zeros((iters, 2))
    for i in range(iters): 
        state_estimate_i, covariance_estimate_i = ekf.one_iteration(i)
        state_estimates[i] = state_estimate_i.reshape(2)
    return np.array(state_estimates) 

def plot_results(ground_truth_data, estimated_data):         
    _, axs = plt.subplots(2)
    axs[0].scatter(np.arange(ground_truth_data.shape[0]), ground_truth_data, color="blue", label="Ground Truth State")
    axs[0].scatter(np.arange(100), estimated_data[:, 0], color="green", label="Estimated State") 
    #axs[1].plot(np.array([-1.] * estimated_data.shape[0]), color="blue", label="Expected a value")
    #axs[1].plot(np.array([1.] * estimated_data.shape[1]), color="blue", label="Expected a value")
    axs[1].plot(estimated_data[:, 1], color="green", label="Estimated a value")
    for a in axs: 
        a.legend() 
    plt.show() 

if __name__=="__main__":  

    iters = 100
    x_0 = np.random.normal(1, math.sqrt(2))   
    x_a = x_0
    actual_obs, actual_state = generate_ground_truth_data(iters, x_0)
    #print(actual_obs)

    x_0 = x_a 
    a_0 = 2
    state_0 = np.array([x_0, a_0]).reshape(2, 1)
    initial_cov = np.diag([2, 2])
    ekf = EKF(state_0, initial_cov, actual_obs)
    state_estimates = main(ekf, iters)
    plot_results(actual_state, state_estimates)
