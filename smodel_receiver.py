#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 13:00:54 2021

@author: rgn
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import signaldetectionmodel as smodel

n = 100
t = 350

int_mean = -3.
int_sd = 1.

delta_r = .8
theta = smodel.theta


a = .8
b = 1.
c = .9

r = .9 # relevance index 0 < r < 1

senders_x = smodel.output[0][-1] # Takes the genomes of senders in the last generation
senders_p = smodel.output[1][-1] # Takes the proportions of genomes in the last generation

x = np.sum(senders_x * senders_p) # Calculates final mean threshold for sender calling


def mean(values):
    mean = sum(values) / len(values)
    return mean

def call_reception(n, t, int_mean, int_sd, delta_r, theta, a, b, c, r, x):
    
    y_t = np.random.normal(loc = int_mean, scale = int_sd, size = n)  # genotypes for threshold y in the initial population
    u_t = np.random.binomial(size = n, n = 1, p = .5) # genotypes for eavesdropping trait in the initial population

    Pe_t = len([u for u in u_t if u == 1]) / n # Proportion of eavesdroppers in the initial receiver population 

    Pne_t = (1 - Pe_t)
    
    Pcall = 1 - norm.cdf(x) # Probability of a senders call. CDF of the average calling threshold of the last generation from the smodel
    
    q_sn = norm.cdf(y_t, delta_r) # CDF for the signal + noise distribution for the level of evidence that a call has been emitted
    q_n = norm.cdf(y_t) # CDF for the noise only distribution for the level of evidence of call

    D_pe =[] # Stores proportions of eavesdroppers after each generation
    D_pne = [] # Stores proportions of non eavesdroppers after each generation

    
    for i in range(t):
        
        # Probability of Correct detection:
            
        Pcd = r * ( theta * ( Pcall * (1 - q_sn) + (1 - Pcall) * (1 - q_n)) 
                   + (1 - theta) * ( Pcall * (1 - q_sn) + (1 - Pcall) * (1 - q_n)) )
        Pcd = mean(Pcd)
        
        # Probability of False alarm:
        
        Pfa = (1 - r) * ( theta * ( Pcall * (1 - q_sn) + (1 - Pcall) * (1 - q_n))
                         + (1 - theta) * ( Pcall * (1 - q_sn) + (1 - Pcall) * (1 - q_n)))

        Pfa = mean(Pfa)

        # Probability of correct rejection:
        
        Pcr = (1 - r) * ( theta * ( Pcall * q_sn + (1 - Pcall) * q_n)
                         + (1 - theta) * ( Pcall * q_sn + (1 - Pcall) * q_n))
        Pcr = mean(Pcr)
        
        # Probability of missed detection:
        
        Pmd = r * ( theta * ( Pcall * q_sn + (1 - Pcall) * q_n)
                         + (1 - theta) * ( Pcall * q_sn + (1 - Pcall) * q_n))
        Pmd = mean(Pmd)
       
       # Fitness outcome of eavesdroppers:
            
        We_t1 = (1 - a) * (Pcd + Pfa) + (1 - c) * Pmd + b * Pcr

        Wcd = (1 - a) * Pcd
        Wfa = (1 - a) * Pfa
        Wcr = b * Pcr
        Wmd = (1 - c) * Pmd
        
        
        # Fitness outcome of non eavesdroppers 
        
        Wne_t1 = (1 - c) * (Pcd + Pmd) + b * (Pfa + Pcr) 

        # Average fitness in the population
        
        avg_W = We_t1 * Pe_t + Wne_t1 * Pne_t
        
        # Proportion of each genome in the next generation 
        
        Pe_t1 = Pe_t * (We_t1 / avg_W)
        Pne_t1 = Pne_t * (Wne_t1 / avg_W) 
       
        
        # Updating proportions of each genome 
        
        D_e = Pe_t - Pe_t1 # Growth rate of eavesdroppers 
        D_ne = Pne_t - Pne_t1 # Growth rate of non eavesdroppers
        
        Slowed_De = D_e/10**100 # Slowed growth rate
        Slowed_Dne = D_ne/10**100 # Slowed growth rate
        
        
        Pe_t = Pe_t1 # Proportion updating
        Pne_t = Pne_t1 # Proportion updating
        
        Pe_t1 = Slowed_De + Pe_t
        Pne_t1 = Slowed_Dne + Pne_t
        
        Pe_t = Pe_t1 
        Pne_t = Pne_t1 
        
        D_pe = np.append(D_pe, Pe_t)
        D_pne = np.append(D_pne, Pne_t)
    
    return D_pe, D_pne, Wcd, Wfa


r = call_reception(n, t, int_mean, int_sd, delta_r, theta, a, b, c, r, x)

eavesdroppers = r[0] # Propotion of eavesdroppers after each generation
noneavesdroppers = r[1] # Proportion of non eavesdroppers after each gen


# Plotting #

xax = (0, t)
yax = (0, 1)


r_low = .2
r_high = .8

a_high = .7

m = call_reception(n, t, int_mean, int_sd, delta_r, theta, a, b, c, r_low, x)
ea = m[0]
nonea = m[1]
lolo = call_reception(n, t, int_mean, int_sd, delta_r, theta, a_high, b, c, r_high, x)
loloe = lolo[0]
lolonoe = lolo[1]
 

plt.plot(eavesdroppers, color = "blue", label = "p(u) for R = 0.5")
plt.plot(noneavesdroppers, color = "red", label = "1 - p(u) for R = 0.5")
plt.plot(ea, color = "lightblue", label = "p(u) for R = 0.2")
plt.plot(nonea, color = "indianred", label = "1 - p(u) for R = 0.2")
plt.plot(loloe, color = "navy", label = "p(u) for R = 0.8, α = 0.7")
plt.plot(lolonoe, color = "darkred", label = "1 - p(u) for R = 0.8, α = 0.8")
plt.ylim(yax)
plt.xlim(xax)
plt.ylabel("Proportion of receiver type")
plt.xlabel("Generations")
plt.legend()
plt.show()











