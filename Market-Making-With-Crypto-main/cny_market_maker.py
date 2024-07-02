#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 05:24:35 2022

@author: larrychen
"""

# For calculations
import numpy as np
import pandas as pd
import random
from datetime import datetime
import powerlaw
import scipy.optimize

# For Visualization
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib.dates as mdates

# Save To/Navigate Through System Directory
import sys
import os
import pickle

# For Parallel Computation
from multiprocessing import Pool

# Other/APIs
import krakenex
from pykrakenapi import KrakenAPI
import binance

from bcalc import Indicator as ind

# Create a brownian motion generator
def brownian_motion(x0, n, dt, sigma):

    # Generate normal random variables
    r = np.random.normal(size = n, scale = sigma*np.sqrt(dt))

    # Calculate cumulative sum of random variables
    cumsum = np.cumsum(r)

    # Add in initial conditions
    cumsum += np.expand_dims(x0, axis=-1)
    cumsum[0] = x0
    
    return cumsum


# Set seed for reproducible results
np.random.seed(101)

# Initialize parameters
# 开始时点市场价格
# s0 = 100
# s=pd.read_csv("../cny2019.csv",header=None,dtype=np.float16).head(1000)
s=np.genfromtxt("../cny2019.csv")

# 年化时间
T = 1
# 波动率，为什么波动率这么大？需要再研究200%？
sigma = 0.01
# 价格步长
dt = 0.005
# 用来调整 ，gamma=0时，买卖点差变为对称方式（Symmetric)
# gamma越小，风险越小，如：gamma=0.01；gamma越大，风险越大，收益增加，如：gamma=1
gamma = 0.1
# 市场订单的密度（the intensity of the arrival of orders）
k = 400 #1.5
# 初始资金量（敞口）
A = 140
# 模拟次数
sim_length = 1000

# Use lists to keep track of results
spreadlist = []
reslist = []
qlist = []
sharpeList = []
drawDownList = []

# Duration of each simulation
N = 200
temp = s[1:N+2]

for sim in range(sim_length):
    # s = brownian_motion(s0, N+1, dt, sigma)
    s = temp
    # Initialize empty array for pnl
    pnl = np.zeros(N+2)
  
    # Inventory
    q = np.zeros(N+2)
    
    # Capital/Cash
    x = np.zeros(N+2)
    
    # Ask limit order
    s_a = np.zeros(N+1)

    # Reserve price
    r = np.zeros(N+1)
    
    # Bid limit order
    s_b = np.zeros(N+1)

    for i in range(len(s)):
        
        # Calculate reservation price
        r[i] = s[i] - q[i] * gamma * sigma**2 * (T-i*dt)

        # Calculate spread
        spread = gamma * sigma**2 * (T - i * dt) + (2 / gamma) * np.log(1 + (gamma / k))
        
        spread = spread / 2

        # Adjust spreads for gap between reserve price
        # and asset mid-price
        gap = np.abs(r[i] - s[i])

        if r[i] >= s[i]:
            delta_a = spread + gap
            delta_b = spread - gap    
        else:
            delta_a = spread - gap
            delta_b = spread + gap

        s_a[i] = s[i] + delta_a
        s_b[i] = s[i] - delta_b
        
        # Calculate our lambdas, (12)
        lambda_a = A*np.exp(-k*delta_a) * dt
        lambda_b = A*np.exp(-k*delta_b) * dt
        lambda_a
        lambda_b
        # Restrict to domain of feasible probabilities
        # i.e. [0,1]
        prob_ask = max(0, min(lambda_a,1))
        prob_bid = max(0, min(lambda_b,1))

        # Determine whether or not we buy/sell according
        # to the above probabilities
        sell = np.random.choice([1,0], p=[prob_ask, 1-prob_ask])
        buy  = np.random.choice([1,0], p=[prob_bid, 1-prob_bid])

        # Adjust inventory to reflect transactions
        q[i+1] = q[i] + buy - sell
        
        # Calculate new capital
        x[i+1] = x[i] + sell*(s[i]+delta_a) - buy*(s[i]-delta_b)

        # Calculate pnl of assets
        pnl[i+1] = x[i+1] + q[i+1]*s[i]
        
        # Append results to lists
        spreadlist.append(spread * 2)
        
        
    drawDownList.append(ind.MaxDrawdown(pnl))
    sharpeList.append(ind.sharpe_ratio(pd.DataFrame(pnl)).astype(float))
    reslist.append(pnl[-1])
    qlist.append(q[-1])



pd.DataFrame([np.mean(spreadlist), np.mean(reslist), np.std(reslist), np.mean(qlist), np.std(qlist)],
        index=['Avg. Spread','Profit','Std (Profit)', 'Final $q$', 'Std (Final $q$)'],
        columns=['Inventory']).T



plt.figure(figsize=(15,8))

plt.title('Histogram and Kernel Density Estimate of Final PnL Values')

# Set plot style
sns.set()

# Plot kernel density estimate with normalized histogram
sns.distplot(reslist,bins=55)

plt.axvline(x=np.mean(reslist),
            color='red', label='mean $\mu$ = '+str(np.round(np.mean(reslist),2)))

plt.legend()
plt.ylabel('Density')
plt.xlabel('Final PnL Value')

plt.show()



plt.figure(figsize=(15,8))

plt.title('Histogram and Kernel Density Estimate of Final Inventory Values')

# Set plot style
sns.set()

# Plot kernel density estimate with normalized histogram
sns.distplot(qlist,bins=19)

plt.axvline(x=np.mean(qlist),
            color='red', label='mean $\mu$ = '+str(np.round(np.mean(qlist),2)))

plt.legend()
plt.ylabel('Density')
plt.xlabel('Final Inventory Value')

plt.show()




# Combine all our results into a single DataFrane
res = pd.DataFrame([s_a, s, s_b], index=['sell limit orders', 'mid price', 'buy limit orders']).T

res['sells']  = ((pd.Series(q).diff() < 0)[:-1] * s_a).replace(0,np.nan)
res['buys']   = ((pd.Series(q).diff() > 0)[:-1] * s_b).replace(0,np.nan)

fig = plt.figure(figsize=(15,10))

spec = gridspec.GridSpec(ncols=1, nrows=2, wspace=0.5,
                         hspace=0.5, height_ratios=[3, 1])

fig.add_subplot(spec[0])

# Plot limit orders and mid price
plt.plot(res.iloc[:,0], color='grey',  label='Ask Limit Orders', ls=':')
plt.plot(res.iloc[:,1], color='black', label='Mid Price')
plt.plot(res.iloc[:,2], color='grey',  label='Bid Limit Orders', ls=':')

# Add buys and sells
plt.scatter(range(201), res.iloc[:,3], color='red', label='Sell Limit Order Filled')
plt.scatter(range(201), res.iloc[:,4], color='green', label='Buy Limit Order Filled')

plt.legend()
plt.title('Trade History of a Single Simulation')

fig.add_subplot(spec[1])

plt.subplot(3,1,3)
plt.bar(range(len(q)), q*(q>0), color='g', edgecolor='g')
plt.bar(range(len(q)), q*(q<0), color='r', edgecolor='r')
plt.title('Inventory')

plt.show()