import numpy as np
import math
from quaternion import Quaternion
import sys

class UKF():
    def __init__(self, angular_velocities, accelerations, dt):
        self.quaternion = Quaternion(1, np.array([0, 0, 0]))
        self.mu = self.quaternion.q.reshape(4, 1)
        self.dt = dt
        self.Q = np.diag([1e-3, 1e-3, 1e-3])
        self.P = np.identity(3)
        np.fill_diagonal(self.P, 1)
        self.R = np.identity(3)
        np.fill_diagonal(self.R, 1)
        self.angular_velocities = angular_velocities
        self.accelerations = accelerations

    def compute_sigma_points_3(self):
        n = self.P.shape[0]
        p_plus_q = self.P + self.Q
        S = np.linalg.cholesky(p_plus_q)
        Xi = np.zeros((4, 2*n))  # 4x6
        left = np.sqrt(2*n)*S
        right = -np.sqrt(2*n)*S
        #S = np.linalg.cholesky(2*n*p_plus_q)
        S = np.hstack([left, right])

        for i in range(2*n):
            Wi_rot = S[:, i]
            Wi_q = Quaternion()
            Wi_q.from_axis_angle(Wi_rot)
            mu_q = Quaternion()
            mu_q.q = self.mu.reshape(4)
            q_w = Wi_q*mu_q
            #q_w_prime = Wi_q.inv()*mu_q
            Xi[:, i] = q_w.q.reshape(4)
            #Xi[:, n+i] =  q_w_prime.q.reshape(4)

        return Xi

    def process_model_3(self, delta_t, omega, Xi):

        Yi = Xi.copy()  # 4x6
        alpha_delta = omega*delta_t
        for i in range(Xi.shape[1]):
            qk_vec = Xi[:, i]
            qk = Quaternion()
            qk.q = qk_vec 
            q_delta = Quaternion()
            q_delta.from_axis_angle(alpha_delta.reshape(3))
            Yi[:, i] = (q_delta*qk).q

        return Yi

    def quaternion_mean(self, sigma_points):
        """ 
        @params: sigma_points 
        removes the mean from the quaternion comp of sigma points 
        """

        error_lim = 1e-2
        q_curr = Quaternion()
        q_curr.q = self.mu.reshape(4).astype('float64')
        while True:
            errors_axis_angle = []
            for i in range(sigma_points.shape[1]):
                qi = Quaternion()
                qi.q = sigma_points[:, i]
                ei = qi*q_curr.inv()
                ei.normalize()
                errors_axis_angle.append(ei.vec())  # of axis angle
            e_bary = np.average(np.array(errors_axis_angle), axis=0)
            e_bary_norm = np.linalg.norm(e_bary)
            if e_bary_norm < error_lim:
                return q_curr, np.array(errors_axis_angle)

            # e_bary - axis angle
            e_bary_quat = Quaternion()
            e_bary_quat.from_axis_angle(e_bary)
            q_curr.normalize()
            q_curr = e_bary_quat * q_curr

    def compute_w_prime(self, sigma_points):
        """
        @params: self,sigma points, quaterion_mean
        # map from 4 x 6 to 3 x 6
        """
        q_curr_mean, errors = self.quaternion_mean(sigma_points)
        self.mu = q_curr_mean.q.reshape(4, 1)
        w_points = np.zeros((sigma_points.shape[0]-1, sigma_points.shape[1]))
        for i in range(sigma_points.shape[1]):
            w_points[:, i] = errors[i].reshape(3)

        return w_points

    def compute_covariance(self, sigma_points, sigma_points_2):
        """ 
        @params: self, sigma_points
        computes covariance matrix given a set of sigma points 
        """
        sigma_points_cov = np.zeros(
            (sigma_points.shape[0], sigma_points_2.shape[0]))
        for i in range(sigma_points.shape[0]):
            sigma_points_cov += sigma_points[:, i].reshape(
                sigma_points.shape[0], 1)@sigma_points_2[:, i].reshape(sigma_points_2.shape[0], 1).T

        sigma_points_cov /= sigma_points.shape[1]

        return sigma_points_cov.reshape(sigma_points.shape[0], sigma_points_2.shape[0])

    def measurement_model(self, sigma_points):
        """ 
        @params: self, Y, model  
        Uses on of 3 measurement models to project Yi into 3-d measurement space  
        """
        Z = np.zeros((3, 6))
        g = Quaternion(0, np.array([0, 0, 9.81]))
        for i in range(sigma_points.shape[1]):
            sigma_point_q = Quaternion()
            sigma_point_q.q = sigma_points[:, i]
            g_prime = sigma_point_q.inv() * g * sigma_point_q
            z_h2 = g_prime.vec()
            Z[:, i] = z_h2

        return Z

    def compute_innovation(self, measured_z, estimated_z):
        """ 
        @params: self, measured_z, estimated_z
        """
        innovation = np.subtract(measured_z, estimated_z)
        return innovation

    def compute_kalman_gain(self, W_prime, Z_shifted, Pvv):
        """ 
        @params: self, Z_shifted, Y_shifted, Pvv 
        computes cross correlation matrix 
        """
        Pxz = self.compute_covariance(W_prime, Z_shifted)
        K = Pxz@np.linalg.inv(Pvv)
        return K

    def update_state_est(self, kalman_gain, innovation):
        update_vec = np.matmul(kalman_gain, innovation)
        update_vec_q = Quaternion()
        update_vec_q.from_axis_angle(update_vec)
        x_mu = Quaternion()
        x_mu.q = self.mu
        x_new_mu = x_mu*update_vec_q
        self.mu = x_new_mu.q.reshape(4, 1)

    def estimate_error_cov(self, kalman_gain, Pvv, Pk_prior):
        """ 
        @params: self, K, Pvv, Pk^-  
        """
        self.P = Pk_prior - kalman_gain@Pvv@kalman_gain.T

    def forward(self, delta_t, omega, accel, i):
        # prediction
        Xi = self.compute_sigma_points_3() 
        Yi = self.process_model_3(delta_t, omega, Xi)
        W_prime = self.compute_w_prime(Yi) 
        pk_prior = self.compute_covariance(W_prime, W_prime)
        Z = self.measurement_model(Yi)
        estimated_z = np.average(Z, axis=1) 
        Z_shifted = Z - estimated_z.reshape(3, 1)
        Pzz = self.compute_covariance(Z_shifted, Z_shifted)
        Pvv = Pzz + self.R
        K = self.compute_kalman_gain(W_prime, Z_shifted, Pvv) 
        vk = self.compute_innovation(accel.reshape(3), estimated_z.reshape(3))

        self.update_state_est(K, vk)
        self.estimate_error_cov(K, Pvv, pk_prior) 

    def main(self, ukf, iters):
        roll, pitch, yaw = [], [], []
        for i in range(iters):
            #print("Iteration: {}".format(i))
            omega = self.angular_velocities[:, i]
            accel = self.accelerations[:, i] 
            delta_t = self.dt[:, i]
            self.forward(delta_t, omega, accel, i)
            mu_q = Quaternion()
            mu_q.q = ukf.mu
            euler_angles = mu_q.euler_angles()
            roll.append(euler_angles[0])
            pitch.append(euler_angles[1])
            yaw.append(euler_angles[2])

        return roll, pitch, yaw
