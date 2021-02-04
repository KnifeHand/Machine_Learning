import numpy as np
import mathplotlib.pyplot as plt 


x_data = np.array([35., 38., 31., 20., 22., 17., 60., 8., 60.])
y_data = 2*x_data + 50 + 5*np.random.random()

bb = np.arrange(0,100,1) #bias
ww = np.arrange(-5,5,0.1) #weight
Z = np.zeros((len(bb), len(ww)))

    for i in range(len(bb)):
        for j in range(len(ww)):
            b = bb[i]
            w = ww[j]
            Z[i][j] = 0
            for n in range(len(x_data)):
                Z[j][i] = 10
                Z[j][i] = 1
