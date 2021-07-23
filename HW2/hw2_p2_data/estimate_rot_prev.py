from scipy import io
import numpy as np 
import math 
import matplotlib.pyplot as plt 
import sys 
from ukf import UKF   
from quaternion import Quaternion
import os 
#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter
 
def get_euler_angles(rotation_mats):  
	num_mats = np.shape(rotation_mats)[2] 
	alphas, betas, gammas = [], [], [] 
	for i in range(num_mats):  
		r_i = rotation_mats[:, :, i]
		alphas.append(math.atan2(r_i[1, 0], r_i[0, 0])) 
		betas.append(math.atan2(-r_i[2, 0], np.sqrt(pow(r_i[2, 1], 2) + pow(r_i[2, 2], 2)))) 
		gammas.append(math.atan2(r_i[2, 1] , r_i[2, 2]))  
	
	alphas = np.array(alphas)
	betas = np.array(betas)
	gammas = np.array(gammas)
	
	return gammas, betas, alphas 
 
def get_scale_factor(sensitivity):
	vref = 3300 
	max_ad = 1023 
	return float(vref / (max_ad*sensitivity)) 

def get_sensitivity(raw_data, bias, scale_factor): 
	return (np.subtract(raw_data, bias)) * scale_factor 
		
def calibration(imu, accel, gyro, T):	

	gravity = 9.81  
	biases = np.average(accel[0:500], axis=1).reshape(3,1) 
	biases[2] = biases[1]
	accel_sensitivity = gravity / (accel[2, 0] - biases[2]) #7.18   
	accel_scaled = (accel - biases)* accel_sensitivity 
	accel_scaled = accel_scaled * np.array([-1., -1., 1.]).reshape(3, 1)
	a_x, a_y, a_z =  accel_scaled[0].astype('float64'), accel_scaled[1].astype('float64'), accel_scaled[2].astype('float64')  	   

	"""
	fig, axs = plt.subplots(3)
	fig.suptitle('Acceleration with: ' + str(accel_sensitivity))  
	axs[0].plot(a_x, color='blue', label="accel x")      
	axs[1].plot(a_y, color='red', label="accel y")  
	axs[2].plot(a_z, color='blue', label="accel z") 
	for ax in axs:
		ax.legend() 
	plt.show() 
	"""
	
	#gyro = gyro * (180./math.pi) 
	w_z, w_x, w_y = gyro[0].astype('float64'), gyro[1].astype('float64'), gyro[2].astype('float64')   
	bias_rotz, bias_rotx, bias_roty = np.average(w_z[0]).astype('float64'), np.average(w_x[0]).astype('float64'), np.average(w_y[0]).astype('float64')
	w_z -= bias_rotz  
	w_x -= bias_rotx 
	w_y -= bias_roty   
	
	rot_z, rot_x, rot_y = np.cumsum(w_z), np.cumsum(w_x), np.cumsum(w_y)  
	
	angular_sensitivity = .0015 
	rot_z *= angular_sensitivity
	rot_x *= angular_sensitivity
	rot_y *= angular_sensitivity 
	
	""""
	vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')    
	vicon_rots_data = vicon["rots"] 
	alpha, beta, gamma = get_euler_angles(vicon_rots_data)  
	fig1, axs = plt.subplots(3) 
	fig1.suptitle('Gyro and Vicon data ')
	axs[0].plot(rot_z, color='blue', label="gyro z") 
	axs[0].plot(alpha, color='green', label="vicon z") 
	axs[1].plot(rot_x, color='blue', label="gyro x")
	axs[1].plot(gamma, color='green', label="vicon x")
	axs[2].plot(rot_y, color='blue', label="gyro y")
	axs[2].plot(beta, color='green', label="vicon y")
	for ax in axs:
		ax.legend() 
	plt.show() 
	""" 
	w_z *= angular_sensitivity
	w_x *= angular_sensitivity 
	w_y *= angular_sensitivity   
	 	
	return w_z, w_x, w_y, a_x, a_y, a_z

def estimate_rot(data_num=1):
	#your code goes here 
	imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat"))  
	#imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat') 
	accel = imu['vals'][0:3,:]
	gyro = imu['vals'][3:6,:]
	T = np.shape(imu['ts'])[1]
	time = imu['ts'] 
	 
	w_z, w_x, w_y, a_x, a_y, a_z = calibration(imu, accel, gyro, T)  
	angular_velocities = np.array([w_x, w_y, w_z]) 
	accelerations = np.array([a_x, a_y, a_z])    
	dt = np.diff(time) 
	iters = accelerations.shape[1]-1 
	roll, pitch, yaw = [], [], []   
	#print(angular_velocities[:, 0])
	ukf = UKF(angular_velocities, accelerations, dt)
	roll, pitch, yaw = ukf.main(ukf, iters) 
	"""
	for i in range(iters):  
		print("At Iteration: {}".format(i ))
		omega = angular_velocities[:, i].reshape(3, 1)
		accel = accelerations[:, i].reshape(3, 1) 	
		ukf.forward(i)   
		mu_q = Quaternion()
		mu_q.q = ukf.mu
		euler_angles = mu_q.euler_angles() 
		roll.append(euler_angles[0])
		pitch.append(euler_angles[1])
		yaw.append(euler_angles[2])
	"""
	roll, pitch, yaw = np.array(roll), np.array(pitch), np.array(yaw) 
	return roll, pitch, yaw 


if __name__=="__main__": 
	data_num = 3
	roll_est, pitch_est, yaw_est = estimate_rot(data_num)    
	vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')    
	rotation_mats = vicon["rots"] 
	roll_v, pitch_v, yaw_v = get_euler_angles(rotation_mats) 



	fig2, axs = plt.subplots(3) 
	axs[0].plot(roll_v, color="blue", label="Roll Vicon") 
	axs[0].plot(roll_est, color="green", label="Roll Estimate") 	
	axs[0].legend()
	axs[1].plot(pitch_v, color="blue", label="Pitch Vicon") 
	axs[1].plot(pitch_est ,color="green", label="Pitch Estimate") 
	axs[1].legend() 
	axs[2].plot(yaw_v, color="blue", label="Yaw Vicon") 
	axs[2].plot(yaw_est ,color="green", label="Yaw Estimate") 
	axs[2].legend()
	plt.show()
