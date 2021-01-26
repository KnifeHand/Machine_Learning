#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:41:29 2018

"""

#import sys

def pow2(n):
    pow = 1
    for i in range(n):
        pow = pow * 2
    return pow

N = 3
#N = int(sys.argv[1])
P = pow2(N)
print(P)

#exercise

#What happens when you execute the following program?
def abs(x):
    if x < 0:
        return -x
    else:
        return x
    

print(abs(-8))


def is_prime(N):
    if N < 2: # deal with 1, defensively
        return False
    for i in range(2, N):
        if N % i == 0:
            return False
        return True
    
print(is_prime(3))