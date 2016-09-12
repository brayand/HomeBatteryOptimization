# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:59:39 2015
https://www.chrisstucchio.com/blog/2014/work_hours_with_python.html
@author: abray
"""
#Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = pd.DataFrame.from_csv("C:/Users/abray/Documents/GitHub/Blog/HomeBatteryOptimization/Load_Price_Nov.csv", parse_dates=True, index_col=0)
data = data[:50]
data["Load"] = data["Load"]/20
n = len(data)
plt.plot(data.index,data["Load"])

#Parameters
BatterySize = 7000 #7 kWh
MinCharge = 0
BegCharge = 1

days = np.unique(data.index.date)

chargeState = BatterySize

#def f(x):
#    for day in days:
#        dayData = data[data.index.date==day]
#        hours = np.unique(dayData.index.hour)
#        for hour in hours:
#            hourData = dayData[dayData.index.hour==hour]
            
            
#Not For Loop

power = np.ones([n,1])#Energy moved in hour
def obj(power):
    
    price = np.array(data["Settlement Point Price"])#Price of Power
    cashFlow = np.transpose(price * np.transpose(power)) #Cash flow in each hour
    cumSumCash = np.cumsum(cashFlow) #Cumulative sum of cash flow in each hour
    
    return -1*cumSumCash[len(cumSumCash)-1]
    
 #State of battery
batteryState = np.cumsum(power)
powerRange = np.tile([-200,200],[n,1])
    
cons = ({'type':'ineq','fun': lambda x: np.cumsum(x)},
        {'type':'ineq','fun': lambda x: BatterySize - np.cumsum(x)},
        )

res = minimize(obj, power, bounds = powerRange, constraints=cons)
print res
-1*res.fun
moves = res.x

