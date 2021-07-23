import numpy as np
import torch as torch 
import torch.nn as nn
from torch.distributions import Normal
#from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable 



class Critic(nn.Module): 
    def __init__(self, input_dim, h_dim, out_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim 
        self.out_dim = out_dim

        #define the network  
        self.model = nn.Sequential( 
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(),  
            nn.Linear(self.h_dim, self.out_dim), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, state): 
        return self.model(state)  

class u_t(nn.Module):
    def __init__(s, xdim=2, udim=1):
        super().__init__()
        """
        Build two layer neural network
        We will assume that the variance of the stochastic
        controller is a constant, to the network simply
        outputs the mean. We will do a hack to code up the constraint
        on the magnitude of the control input. We use a tanh nonlinearity
        to ensure that the output of the network is always between [-1,1]
        and then add a noise to it. While the final sampled control may be larger
        than the constraint we desire [-1,1], this is a quick cheap way to enforce the constraint.
        """
        s.m = nn.Sequential(
                nn.Linear(xdim, 8),
                nn.ReLU(True),
                nn.Linear(8, udim),
                nn.Tanh(),
                )
        
        s.std = 1 

    def forward(s, x, u=None):
        """
        This is a PyTorch function that runs the network
        on a state x to output a control u. We will also output
        the log probability log u_theta(u | x) because we want to take
        the gradient of this quantity with respect to the parameters
        """
        # mean control
        mu = s.m(x)
        # Build u_theta(cdot | x)
        n = Normal(mu, s.std)
        # sample a u if we are simulating the system, use the argument
        # we are calculating the policy gradient
        if u is None:
            u = n.rsample()
        logp = n.log_prob(u)
        return u, logp

def rollout(policy, critic):
    """
    We will use the control u_theta(x_t) to take the control action at each
    timestep. You can use simple Euler integration to simulate the ODE forward
    for T = 200 timesteps with discretization dt=0.05.
    At each time-step, you should record the state x,
    control u, and the reward r
    """
    m = 2; l=1; b=0.1; g=9.8;
    gamma=0.99;
    get_rev = lambda z, zdot, u: -0.5*((np.pi-z)**2 + zdot**2 + 0.01*u**2)

    xs = [np.zeros(2)]; us = []; rs= []; 
    values = [] 
    dt = 0.05
    for t in np.arange(0, 10, dt):
        # The interface between PyTorch and numpy becomes a bit funny
        # but all that this line is doing is that it is running u(x) to get
        # a control for one state x
        u = policy(torch.from_numpy(xs[-1]).view(1,-1).float())[0].detach().numpy().squeeze().item()   
        value = critic(torch.from_numpy(xs[-1]).view(1, -1).float())[0].detach().numpy().squeeze().item()

        z, zdot = xs[-1][0], xs[-1][1]
        zp = z + zdot*dt
        zdotp = zdot + dt*(u - m*g*l*np.sin(z) - b*zdot)/m/l**2

        rs.append(get_rev(z, zdot, u))
        us.append(u)
        xs.append(np.array([zp, zdotp])) 
        values.append(value)

    R = sum([rr*gamma**k for k,rr in enumerate(rs)])
    return {'x': torch.tensor(xs[:-1]).float(),
            'u': torch.tensor(us).float(),
            'r': torch.tensor(rs).float(), 
            'R': R}

def example_train():
    """
    The following code shows how to compute the policy gradient and update
    the weights of the neural network using one trajectory.
    """
    policy = u_t(xdim=2, udim=1) 
    critic = Critic(input_dim=2, h_dim = 8, output_dim=1)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # 1. get a trajectory
    t = rollout(policy, critic)
    """"
    2. We now want to calculate grad log u_theta(u | x), so
    we will feed all the states from the trajectory again into the network
    and this time we are interested in the log-probabilities. The following
    code shows how to update the weights of the model using one trajectory
    """
    logp = policy(t['x'].view(-1,2), t['u'].view(-1,1))[1]
    f = -(t['R']*logp).mean()

    # zero_grad is a PyTorch peculiarity that clears the backpropagation
    # gradient buffer before calling the next .backward()
    policy.zero_grad()
    # .backward computes the gradient of the policy gradient objective with respect
    # to the parameters of the policy and stores it in the gradient buffer
    f.backward()
    # .step() updates the weights of the policy using the computed gradient
    optim.step()


def train(episodes, baseline):
    """
    This is very similar to example_tran() above. You should sample
    multiple trajectory at each iteration and run the training for about 1000
    iterations. You should track the average value of the return across multiple
    trajectories and plot it as a function of the number of iterations.
    """ 
    policy = u_t(xdim=2, udim=1) 
    critic = Critic(input_dim=2, h_dim = 8, out_dim=1)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    num_trajectories = 50 
    reward_means = []
    for i in range(episodes):   
        mini_batch_losses = []
        traj_logps = []   
        
        #minibatch of trajectories
        for _ in range(num_trajectories):  
            trajectory = rollout(policy, critic)   
            logp = policy(trajectory['x'].view(-1,2), trajectory['u'].view(-1,1))[1]        
            traj_logps.append(logp.sum())  
            loss = -trajectory['R'] 
            mini_batch_losses.append(loss)    
            #f = -((trajectory['R']) *logp).mean()     
        mini_batch_losses = np.array(mini_batch_losses)
        mini_batch_loss_mean = np.mean(mini_batch_losses)          
        mini_batch_loss_mean = torch.tensor([mini_batch_loss_mean]).float()  
        mb_losses = torch.from_numpy(mini_batch_losses)   
        
        #compute advantage: test with diffeent baslines for variance reduction 
        if baseline == 'avg': 
            advantage = mb_losses - mini_batch_loss_mean  
        elif baseline == 'wavg': 
            #TODO: compute weighted average 
            advantage = np.array(mini_batch_losses) - mini_batch_loss_mean  
        elif baseline == 'ac':
            #TODO: use the critic network to compute value function 
            value = None
            advantage = np.array(mini_batch_losses) - value 
        
        policy_loss = []  
        for idx, log_p in enumerate(traj_logps):    
            policy_loss.append(advantage[idx].view(-1, 1) * log_p) 
            
        policy_loss = torch.cat(policy_loss).sum().view(-1, 1)
        
        optim.zero_grad()     
        policy_loss.backward()
        optim.step()   
          
        reward_means.append(mini_batch_loss_mean)   
        if i % 100 == 0: 
            print("Average Loss: {:.2f} at Iteration {}".format(mini_batch_loss_mean.item(), i))
    
    return reward_means

if __name__ == "__main__": 

    episodes = 1000 #Tunable 
    baseline = 'avg' #'wavg', 'ac' 
    rewards = train(episodes, baseline)
     
    #plotting a
    plt.figure(figsize=(10, 10)) 
    plt.plot(range(len(rewards)), rewards) 
    plt.title("Average Losses vs. Number of Episodes / Weight Udpates")
    plt.grid() 
    plt.show()