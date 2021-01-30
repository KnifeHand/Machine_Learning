 #-*- coding: utf-8 -*-
""" 
Matt Hurt
Machine Learning - Dr. Feng Jiang
Quiz 1 PythonTest - Contours
"""
############################3rd attempt##################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111)
u = np.linspace(0,10,11)
x, y = np.meshgrid(u,u)
# z = x**2 + y**2
z = (x-5)**2 + (y-5)**2
ax.contourf(x,y,z, 20, alpha = 0.7, cmap = plt.get_cmap('jet'))
plt.plot([5], [5], 'o', ms = 17, markeredgewidth = 20, color = 'blue')
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()

############################1st attempt##################################

# X = np.array([1,2,3,4,5,6,7,8,9,10] )
# Y = np.array([1,2,3,4,5,6,7,8,9,10])
# Z = (X-5)**2 + (X-5)**2

# for i in range(len(X)):
#     for j in range(len(Y)):
#         x = X[i]
#         y = Y[j]
#         #Z[j][i] = 
        
# plt.contourf(X, Y, Z, 20, alpha = 0.5, cmap = plt.get_cmap('jet'))
# plt.plot([5], [5], 'o', ms = 12, markeredgewidth = 3, color = 'orange')
# plt.xlim(0,10)
# plt.ylim(0,10)
# plt.xlabel(r'$X$', fontsize = 16)
# plt.ylabel(r'$Y$', fontsize = 16)
# plt.show()

###########################2nd attempt###################################
# import numpy as np
# import matplotlib.pyplot as plt
# N = 8
# y = np.zeros(N)
# x1 = np.linspace(0, 10, N, endpoint=True)
# x2 = np.linspace(0, 10, N, endpoint=False)
# plt.plot(x1, y, 'o')
#[<matplotlib.lines.Line2D object at 0x...>]
# plt.plot(x2, y + 0.5, 'o')
# [<matplotlib.lines.Line2D object at 0x...>]
# plt.ylim([-0.5, 1])