# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:31:45 2018

@author: Jun Wei
"""
import numpy as np
import cProfile

#t = [0,1,2,3,4,5,6,7,8,9]
#T = [50.4375,
#48.5,
#46.625,
#44.8125,
#43.125,
#41.5,
#39.9375,
#38.5,
#37.125,
#35.8125
#]

#def tempfit8(t,T):
#    #Fit: (Tpi-Ts)*exp(-t/tau)+Ts
#    #Loss Function: (T - ((Tpi-Ts)*exp(-t/tau)+Ts) )^2
#    #Loss Gradient [(Tpi), Ts, tau]: [ -2*exp(-t/tau)*((Tpi-Ts)*(-exp(-t/tau))-Ts+T),
#    #                                 2*(exp(-t/tau)-1)*((Tpi-Ts)*(-exp(-t/tau))-Ts+T),
#    #                                -(2*t/tau**2)*(Tpi-Ts)*exp(-t/tau)*((Tpi-Ts)*(-exp(-t/tau))-Ts+T) ]
#    
#    Tpi = T[0]; tau = 19.5; Ts = Tpi + tau*(T[1]-T[0])/(t[1]-t[0])
#    
#    grad_mag = 1
#    while grad_mag > 1e-4:
#        n = 0; loss_grad = [0,0]
#        while n < 10:
#            loss_grad[0] += 2*(np.exp(-t[n]/tau)-1)*((Tpi-Ts)*(-np.exp(-t[n]/tau))-Ts+T[n])
#            loss_grad[1] += -(2*t[n]/tau**2)*(Tpi-Ts)*np.exp(-t[n]/tau)*((Tpi-Ts)*(-np.exp(-t[n]/tau))-Ts+T[n])
#            n += 1      
#        grad_mag = (loss_grad[0]**2+loss_grad[1]**2)**0.5
#        Ts = Ts - 0.3*loss_grad[0]
#        tau = tau - 0.3*loss_grad[1]
#    
#    return Ts, tau

def tempfit(t,T):
    Tpi = T[0]; tau = 18.; Ts = float(Tpi + tau*(T[1]-T[0])/(t[1]-t[0]))
    W = np.asarray([Ts,tau]).reshape(-1,1) #W = [[Ts],[tau]]
    
    grad_mag = 1; step = np.asarray([0,0]).reshape(-1,1); wdc = 0
    while grad_mag > 1e-6:
        if wdc > 25000:
            return W, loss_sum
        X = np.exp(-t/W[1][0]); tX = -((Tpi-W[0][0])/W[1][0]**2)*np.multiply(t,X); L = T - (Tpi-W[0][0])*X - W[0][0]; loss_sum = L.sum()
        XtX = np.concatenate((X.T,tX.T)); d_loss = 2*(np.matmul(XtX,L)+np.asarray([[-loss_sum],[0]]))
        step = 0.52*d_loss + 0.69*step
        W -= step; wdc += 1
        grad_mag = np.linalg.norm(d_loss)
    
    return W, loss_sum

#cProfile.run('''
#test2 = tempfit8(t,T)
#''')

t = np.asarray([0.00,
0.87,
1.74,
2.61,
3.48,
4.35,
5.22,
6.09,
6.96,
7.83,
8.70,
9.57,
10.44,
11.31,
12.18
]).reshape(-1,1)
T = np.asarray([25.625,
26.562,
27.687,
28.75,
29.75,
30.687,
31.625,
32.437,
33.25,
34,
34.75,
35.437,
36.062,
36.687,
37.312
]).reshape(-1,1)
    
cProfile.run('''
t2, loss = tempfit(t-t[0],T)
''')

    
        
        