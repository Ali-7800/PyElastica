import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from numpy.linalg import inv



class Filter:
    
    def __init__(self,n_particle,omega_0,sigma_W,delta,h_fun,learning,learning_rate,recorder):
        
        self.N = n_particle
        self.h_fun = h_fun
        self.M = len(self.h_fun.c)
        self.theta = 2.0*np.pi * np.stack([npr.rand(self.N) for i in range(self.M)])
        # self.omega = omega_0 + delta*(2*npr.rand(self.N) - 1)
        self.omega = omega_0 * np.stack([(1.0 + delta*(2.0*npr.rand(self.N) - 1.0)) for i in range(self.M)])
        self.sigma_W = sigma_W
        self.learning = learning
        self.learning_rate = learning_rate
        self.recorder = recorder
        self.h = self.h_fun.eval_h(self.theta)
        self.set_h_bar(self.h)

    def set_h_bar(self,h):
        self.h_bar = h.mean(axis=0)

    def time_update(self,dt):
        self.theta += self.omega*dt
        self.theta %= (2.0*np.pi)


        # self.theta = (self.theta) % (2*np.pi)
        
    def info_update(self,y,dt):
        cos = np.cos(self.theta)
        sin = np.sin(self.theta)
        cos2 = cos*cos - sin*sin
        sin2 = 2.0*sin*cos
        # cos3 = 4*cos*cos*cos - 3*cos
        # sin3 = 3*sin - 4*sin*sin*sin
        if self.learning:
            # cos_bar = cos.mean(axis=1)
            sin_bar = sin.mean(axis=1)
            cos2_bar = cos2.mean(axis=1)
            sin2_bar = sin2.mean(axis=1)
            # cos3_bar = cos3.mean(axis=1)
            # sin3_bar = sin3.mean(axis=1)
            gradient = np.stack([sin_bar, sin2_bar, cos2_bar]).T
            # gradient = np.vstack([np.ones(len(sin_bar)),sin_bar,sin2_bar,cos2_bar]).T
            dI = -(y - self.h_bar) * dt * self.learning_rate # 0.15 is learning rate
            self.h_fun.g_descent(dI,gradient)

        self.h = self.h_fun.eval_h(self.theta)
        self.set_h_bar(self.h)
        ss_bar = (sin*sin).mean(axis=1)
        sc_bar = (sin*cos).mean(axis=1)
        cc_bar = (cos*cos).mean(axis=1)
        K = np.zeros([self.M,self.N])
        K_prime = np.zeros([self.M,self.N])
        for i in range(self.M):
            A11 = ss_bar[i]
            A12 = -sc_bar[i]
            A22 = cc_bar[i]
            # A = np.array([[A11, A12], [A12, A22]])
            A_inv = np.array([[A22,-A12],[-A12,A11]]) / (A11*A22-A12*A12)
            B1 = cos[i] @ (self.h[:,i] - self.h_bar[i]) / self.N
            B2 = sin[i] @ (self.h[:,i] - self.h_bar[i]) / self.N
            B = np.array([B1,B2])
            C = A_inv @ B
            K[i] = C @ np.array([-sin[i],cos[i]])
            K_prime[i] = C @ np.array([-cos[i],-sin[i]])
        dI = K * (y - 0.5 * (self.h + self.h_bar)).T * dt
        correction = 0.5 * K * K_prime * dt

        d_theta = (dI + correction) / (self.sigma_W*self.sigma_W)
        self.theta += d_theta
        self.theta %= (2.0*np.pi)
        
        self.recorder["h_bar"].append(self.h_bar.copy())
        self.recorder["theta_phase"].append(self.theta.mean(axis=1).copy())
        self.recorder["correction"].append(self.theta.copy())




class H_fun:

    '''
        observation model:
        s = [[s_1, s_2, s_3, ...], [s_1, s_2, s_3, ...]]
        c = [[c_1, c_2, c_3, ...], [c_1, c_2, c_3, ...]]
    '''

    # def __init__(self,c=np.array([[0.0,0.0,0.0]]),s=np.array([[0.0,0.0,0.0]])):
    def __init__(self,c,s):
        self.s = s
        self.c = c
        self.M = len(c)     #number of sensors
        self.ms = np.array([len(si) for si in s])
        self.mc = np.array([len(ci) for ci in c])

    def eval_h(self,theta):
        h = []
        for i in range(self.M):
            # cos = np.array([np.cos((k+2)*theta[i]) for k in range(self.mc[i])])
            # sin = np.array([np.sin((k)*theta[i]) for k in range(self.ms[i])])
            # cos = np.array([np.cos(2*theta[i])])
            sin = np.array([np.sin((k+1.0)*theta[i]) for k in range(self.ms[i])])
            cos = np.array([np.cos((k+2.0)*theta[i]) for k in range(self.mc[i])])
            h_i = self.c[i] @ cos + self.s[i] @ sin
            h.append(h_i)
        h = np.array(h).T
        return h

    def eval_dh(self,theta):
        dh = []
        for i in range(self.M):
            # cos = np.array([(k) * np.cos((k)*theta[i]) for k in range(self.ms[i])])
            # sin = np.array([(k+2) * np.sin((k+2)*theta[i]) for k in range(self.mc[i])])
            # sin = np.array([2 * np.sin(2*theta[i])])
            cos = np.array([(k+1.0) * np.cos((k+1.0)*theta[i]) for k in range(self.ms[i])])
            sin = np.array([(k+2.0) * np.sin((k+2.0)*theta[i]) for k in range(self.mc[i])])
            dh_i = -self.c[i] @ sin + self.s[i] @ cos
            dh.append(dh_i)
        dh = np.array(dh).T
        return dh

    def g_descent(self,dI,gradient):
        self.s[:,0] -= dI * gradient[:,0]
        self.s[:,1] -= dI * gradient[:,1]
        self.c[:,0] -= dI * gradient[:,2]
        # self.c[0,0] = 0
        # self.c[:,1] -= dI * gradient[:,1]
        # for i in range(self.M):
        #   self.s[i,0] -= dI[i] * gradient[i,0]
        #   self.s[i,1] -= dI[i] * gradient[i,1]
        #   self.s[i,2] -= dI[i] * gradient[i,2]
        #   # self.s[i,3] -= dI[i] * gradient[i,3]
        #   self.c[i,0] -= dI[i] * gradient[i,3]
        #   # self.c[i,1] -= dI[i] * gradient[i,5]


if __name__ == "__main__":
    main()
