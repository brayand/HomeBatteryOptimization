# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:20:22 2015
References:
http://stackoverflow.com/questions/26470570/seasonal-decomposition-of-time-series-by-loess-with-python
http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.filters.filtertools.convolution_filter.html
http://www.cs.cornell.edu/courses/cs1114/2013sp/sections/s06_convolution.pdf
http://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
@author: abray
"""

import pandas as pd
import numpy as np
import sklearn.cross_validation
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from scipy import stats
import math

data = pd.read_csv("C:/Users/abray/Documents/GitHub/EnergyDataSimulationChallenge/challenge1/data/dataset_500.csv", index_col=0)
#Set timestamp
data["Ones"] = 1
format = "%Y/%m/%d"
times = pd.to_datetime(data.Year.astype(int).astype(str) + '/' + data.Month.astype(int).astype(str) + '/' + data.Ones.astype(int).astype(str), format=format)
data = data.drop(["Label","Ones"], axis=1)
data.set_index(times, inplace=True)

#Only Energy Production Data
#energy = pd.DataFrame(data["EnergyProduction"] ,index=times)
data = pd.DataFrame(data[["House","EnergyProduction"]] ,index=times)

#Break up into train and test
datatrain = data[:int(.8*len(data))]
datatest = data[int(.8*len(data)):]
#datatrain, datatest = sklearn.cross_validation.train_test_split(data, train_size=.8)

#Only Energy Production Data
#energy = pd.DataFrame(data["EnergyProduction"] ,index=times)
#houseEnergy = pd.DataFrame(datatrain[["House","EnergyProduction"]] ,index=times)

#Dates on row, Houses on columns
groups = datatrain.groupby("House")
dateGroups = pd.DataFrame()
for k,g in groups:
    dateGroups[k] = g["EnergyProduction"]

#Get medians of dates    
medians = dateGroups.median(1)

#Plot    
dateGroups.plot(linewidth=1,legend=None)
medians.plot(linewidth=5, c='r')
plt.title("EnergyProduction by House")
plt.xlabel('Date')
plt.ylabel('EnergyProduction')

#Seasonal Decompose
decomposition = seasonal_decompose(medians, model='additive',freq=12)

#Plotting Seasonal
decomposition.plot()
plt.suptitle("Seasonal Decomposition of Energy Production Medians")
plt.xlabel('Date')

#Sine fit
data1 = datatrain["EnergyProduction"]
#data1 = medians
N = len(data1) # number of data points
x = data1.index
td = data1.index-pd.Timestamp(x[0])
t = td.days/30.42

guess_mean = np.mean(data1)
slope, intercept, r_value, p_value, std_err = stats.linregress(t,data1)
guess_slope = slope
guess_std = 3*np.std(data1.values)/(2**0.5)
guess_phase = math.pi/6

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_std*np.sin(t*guess_phase) + t*guess_slope + guess_mean

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(t*x[1]) + x[2]*t + x[3] - data1
est_std, est_phase, est_slope, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_slope, guess_mean])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_std*np.sin(t*est_phase) + t*guess_slope + est_mean


plt.plot(t, data1, '.')
lab1 = "{0:.2f}*sin({1:.2f}*x) + {2:.2f}*x + {3:.2f}".format(guess_std,guess_phase, guess_slope, guess_mean)
plt.plot(t[0:24:], data_first_guess[0:24:], label="First Guess: "+lab1)
lab2 = "{0:.2f}*sin({1:.2f}*x) + {2:.2f}*x + {3:.2f}".format(est_std,est_phase, est_slope, est_mean)
plt.plot(t[0:24:], data_fit[0:24:], label="After fitting: "+lab2)
plt.legend()
plt.title("Fitted Energy Production")
#plt.show()

#Predictionsdata1 = datatrain["EnergyProduction"]
#data1 = medians
N = len(data1) # number of data points
x = data1.index
td = data1.index-pd.Timestamp(x[0])
t = td.days/30.42

data2 = datatest["EnergyProduction"]
#data1 = medians
Npred = len(data2) # number of data points
xpred = data2.index
tdpred = data2.index-pd.Timestamp(xpred[0])
tpred = tdpred.days/30.42

data_pred = est_std*np.sin(tpred*est_phase) + tpred*guess_slope + est_mean

plt.plot(tpred, data2, '.')
lab2 = "{0:.2f}*sin({1:.2f}*x) + {2:.2f}*x + {3:.2f}".format(est_std,est_phase, est_slope, est_mean)
plt.plot(t[0:24:], data_pred[0:24:], label="Fitted Curve: "+lab2)
plt.legend()
plt.title("Predicted Energy Production")
plt.show()

#Check score with MAPE
from sklearn.utils import check_array
def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_array(y_true, y_pred)
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
mape = mean_absolute_percentage_error(data2,data_pred)
