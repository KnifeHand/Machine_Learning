# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:02:01 2020

@author: Jacob Watters

This program solves the least squares problem for a line by relatively straightforward means. 
"""

import numpy as np
import matplotlib.pyplot as plt


def residual(x, y, i):
    return y[i] - pred(x, y, x[i])


def mean(data):
    n = len(data)
    ave = 0
    for i in range(n):
        ave += data[i]
    return ave/n


def S(x, y):
    x_bar = mean(x)
    y_bar = mean(y)
    
    n = len(x)
    result = 0    
    for i in range(n):
        result += (x[i]-x_bar)*(y[i]-y_bar)
        
    return result


def r(x, y):
    result = S(x, y) / (np.sqrt(S(x, x))*np.sqrt(S(y, y)))
    return result


def r2(x, y):
    return r(x, y)**2


def line(x, y, significance = 10):
    b1 = S(x, y) / S(x, x)
    b0 = mean(y)-b1*mean(x)
    return "y = " + str(round(b0, significance)) + " + " + str(round(b1, significance)) + "x"


def pred(x, y, val):
    b1 = S(x, y) / S(x, x)
    b0 = mean(y)-b1*mean(x)
    return b0 + b1*val


def SSE(x, y):
    n  = len(x)
    result = 0
    for i in range(n):
        result += (y[i]-pred(x, y, x[i]))**2
    return result


def SST(x, y):
    n  = len(x)
    result = 0
    for i in range(n):
        result += (y[i]-mean(y))**2
    return result


def var(x, y):
    return SSE(x, y) / (len(x)-2)
    

def std(x, y):
    return (var(x,y))**(0.5)


def plot(x, y, significance=4):
    x1 = np.linspace(np.min(x), np.max(x), 1000)
    
    significance = 4
    
    plt.scatter(x, y, label="data")
    plt.title("Linear Regression: r^2=" + str(round(r2(x, y), significance)) + ", r=" + str(round(r(x, y), significance)))
    plt.plot(x1, pred(x, y, x1), '-r', label=line(x, y,significance))
    plt.text(np.median(x[int(len(x)/2):len(x)]), np.median(y[:int(len(y)/2)]), 'SSE=' + str(round((SSE(x, y)), significance)) + "\nSST=" + str(round(SST(x, y), significance)), horizontalalignment='center',  verticalalignment='center', fontsize=12)
    plt.legend(loc="upper left")
    
    plt.show
    



def main():
    # Enter data
    x = np.array([4,12,14,16,23,30,40,46,55,67,72,79,96,112,127], float)
    y = np.array([4,10,13,14,15,25,27,46,38,46,53,72,82,99,102], float)
    
    plot(x, y)
    
    
if __name__ == "__main__":
    main()








