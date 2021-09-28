#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:30:42 2021

@author: rgn
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters # 

n = 100
t = 1500

int_mean = -2.
int_sd = 1.

delta = 1.
theta = .5


a = .3
b = 1.
c = .8


analytic_x = delta * (1/2) + (1/delta) * np.log(((b-(1-a))/(c-a))*((1-theta)/theta))

# Simulation #

def sim(n, t, int_mean, int_sd, delta, theta, a, b, c):
    x_t = np.sort(np.random.normal(loc = int_mean, scale = int_sd, size = n))
    
    p_t = np.full(n, fill_value = 1/n) 


    q_sn = norm.cdf(x_t, delta) # CDF for signal + noise distribution
    
    q_n = norm.cdf(x_t) # CDF for only noise distribution
    
    D_p = []
    D_x = []
    
    for i in range(t):
        
        W_t1 = theta * ( (1 - a) * (1 - q_sn) + (1 - c) * q_sn ) + (1 - theta) * ( (1 - a) * (1 - q_n) + b * q_n ) 
            
        mean_W = sum(W_t1) / len(W_t1)
        
        p_t1 = p_t * (W_t1 / mean_W) # replicator dynamics

        p_t1 = p_t1 / sum(p_t1) # normalise proportions in t+1

        x_t = x_t # No updating of genomes (no mutation)
        
        p_t = p_t1 # Updating probabilities of each genome
        
        D_p = np.append(D_p, p_t)
        D_x = np.append(D_x, x_t)
        
    D_x = np.reshape(D_x, (t, n))
    D_p = np.reshape(D_p, (t, n))
    
    return(D_x, D_p)

    
# Plotting #

output = sim(n, t, int_mean, int_sd, delta, theta, a, b, c)
products = output[0] * output[1]
means_x = np.sum(products, axis = 1)

"""
xax = (1, t)

yax = (-6, 5)

plt.subplot(1, 2, 1)

plt.plot(output[0][-1], output[1][-1], '-ok')
plt.axvline(analytic_x, color = 'red')
plt.xlim(-5, 5)
plt.xlabel("Thresholds X")
plt.ylabel("Proportion")

plt.subplot(1, 2, 2)

final_pop = [output[0][-1], output[1][-1]]

plt.axhline(analytic_x, color = "blue")
plt.plot(means_x, color = 'red')
plt.plot(final_pop)
plt.xlim(xax)
plt.ylim(yax)

plt.ylabel("Mean Population Threshold E[x]")
plt.xlabel("Generations")

plt.subplots_adjust(wspace = 0.4)
plt.show()

"""
        
        
        
        
    
    
    
    
    
    
    
    