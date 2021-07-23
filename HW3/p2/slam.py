# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np 

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger() 
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations. 
        """
        
        grid_cells = np.zeros((2, len(x)))
        
        x = np.floor((x - s.xmin) / (s.resolution)) 
        y = np.floor((y - s.ymin) / (s.resolution)) 
         
        grid_cells[0] = np.clip(x, 0, s.szx-1)
        grid_cells[1] = np.clip(y, 0, s.szy-1)
        
        return grid_cells.astype(int)

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))
        
        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t)) 
        

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles 
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        n = p.shape[1]
        r = np.random.uniform(0., 1./n) 
        c = w[0]  
        p_prime = np.zeros(p.shape)
        
        i = 0  
       
        for m in range(n):      
            u = r + (m-1) / n     
            while u > c:
                i += 1 
                c += w[i]
            
            p_prime[:, m] = deepcopy(p[:, i])
        
        w_prime = (1./n) * np.ones(w.shape)  
        
        return p_prime, w_prime 

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """

        # TODO: change this implementation 
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading the data 
        if angles is None: 
            angles = s.lidar_angles
            
        valid = (d < s.lidar_dmin) & (d > s.lidar_dmax) 
        if np.sum(valid) < len(d): 
            d = d[valid]
            angles = angles[valid]
        
        # 1. from lidar distances to points in the LiDAR frame 
        x_pos = d * np.cos(angles) 
        y_pos = d * np.sin(angles)
        z_pos = np.zeros(x_pos.shape)
        cartesian_coords = np.vstack((x_pos, y_pos, z_pos))         
        # 2. from LiDAR frame to the body frame
        # homogeneous transformation in 3D (neck: pitch, head: yaw, s.lidar_height: displacement) 
        lidar_frame_coords = make_homogeneous_coords_3d(cartesian_coords) #4 x n 
        bf = euler_to_se3(0., head_angle, neck_angle, np.array([0., 0., s.lidar_height])) 
        bf_coords = bf @ lidar_frame_coords
    
        # 3. from body frame to world frame 
        wf = euler_to_se3(0., 0., p[2], np.array([p[0], p[1], s.head_height]))
        wf_coords = wf @ bf_coords
        
        floor_thresh = .05 
        not_floor_hit = wf_coords[2] > floor_thresh
        #wf_coords = wf_coords[0:2, :]
        wf_coords_filtered = wf_coords[:, not_floor_hit] 
        
        return wf_coords, wf_coords_filtered  

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)
        
        p_curr = s.lidar[t]['xyth'] 
        p_curr[2] = s.lidar[t]['rpy'][2]
        
        p_prev = s.lidar[t-1]['xyth']
        p_prev[2] = s.lidar[t-1]['rpy'][2] 
        
        p_delta = smart_minus_2d(p_curr, p_prev)
        
        return p_delta 
    

    def dynamics_step(s, t):    
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        control_input = s.get_control(t)   
        for i in range(s.p.shape[1]):   
            out = smart_plus_2d(s.p[:, i], control_input)  
            control_noise = np.random.multivariate_normal(np.zeros((3,)), s.Q)
            s.p[:, i] = smart_plus_2d(out, control_noise)
 
    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """  
        log_w_new = np.log(w) + obs_logp
        w_new_normalized = np.exp(log_w_new - slam_t.log_sum_exp(log_w_new)) 
        return w_new_normalized     
 
    def compute_log_odds(s, grid_cells):   
        grid_cells_copy = np.zeros((s.map.szx, s.map.szy), dtype=np.int8) 
        grid_cells_copy[(grid_cells[0], grid_cells[1])] = 1 
        both_one = grid_cells_copy & s.map.cells
        obs_logp = np.sum(both_one)

        return obs_logp
    
    
    def observation_step(s, t):
        """
        This function does the following things
            1. updates the weights of the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle) 
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """ 
        
        #First find the head, neck angle at t (this is the same for every particle)  
        t_corr = s.find_joint_t_idx_from_lidar(t) 
        neck_angle = s.joint["head_angles"][0][t_corr]  
        head_angle = s.joint["head_angles"][1][t_corr] #same for all particles          
       
        all_particles_obs_logp = [] 
        for i in range(s.p.shape[1]):
            p = s.p[:, i]
            world_frame, world_frame_filtered = s.rays2world(p, s.lidar[t]['scan'], head_angle, neck_angle, angles=s.lidar_angles)
            obstacles = s.map.grid_cell_from_xy(world_frame_filtered[0], world_frame_filtered[1]) 
            obs_logp = s.compute_log_odds(obstacles)
            all_particles_obs_logp.append(obs_logp) 

        all_particles_obs_logp = np.array(all_particles_obs_logp)
        s.w = s.update_weights(s.w, all_particles_obs_logp) #update weights  
 
        #get the particle w/ the largest weight
        best_pose_idx = np.argmax(s.w)  
        best_pose = s.p[:, best_pose_idx]
        world_frame_best, world_frame_filtered_best = s.rays2world(best_pose, s.lidar[t]['scan'], head_angle=head_angle, neck_angle=neck_angle, angles=s.lidar_angles) 
        occupied_cells_best_filtered = s.map.grid_cell_from_xy(world_frame_filtered_best[0], world_frame_filtered_best[1])
        
        #cells in between
        best_pose_grid = s.map.grid_cell_from_xy(np.array(best_pose[0]).reshape(1, 1), np.array(best_pose[1]).reshape(1, 1)) 
        best_pose_copied_x = np.full((occupied_cells_best_filtered.shape[1]), best_pose_grid[0]) 
        best_pose_copied_y = np.full((occupied_cells_best_filtered.shape[1]), best_pose_grid[1])  
        best_pose_copied = np.vstack([best_pose_copied_x, best_pose_copied_y]).reshape(-1, occupied_cells_best_filtered.shape[1]) 
        free_cells = np.linspace(best_pose_copied, occupied_cells_best_filtered, endpoint=False)   
        free_grid_cells_x = free_cells[:, 0, :].reshape(free_cells.shape[0] * free_cells.shape[2]).astype(int)
        free_grid_cells_y = free_cells[:, 1, :].reshape(free_cells.shape[0] * free_cells.shape[2]).astype(int) 
        
        s.map.log_odds[(free_grid_cells_x, free_grid_cells_y)] += s.lidar_log_odds_free / 10.
        s.map.log_odds[(occupied_cells_best_filtered[0], occupied_cells_best_filtered[1])] += s.lidar_log_odds_occ 
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max) #prevent increasingly large values 
         
        #record what the LiDAR has already seen 
        s.map.num_obs_per_cell[(free_grid_cells_x, free_grid_cells_y)] += 1 

        #log_odds and map.cell update
        above_threshold = s.map.log_odds > s.map.log_odds_thresh
        below_threshold = s.map.log_odds <= s.map.log_odds_thresh
        s.map.cells[above_threshold] = 1 
        s.map.cells[below_threshold] = 0  
        
        
        s.resample_particles() 

        return s.p[:, best_pose_idx]
    
    
    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')