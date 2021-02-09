# -*- coding: utf-8 -*-
"""
@author: Matt Hurt

Random number generator that prints two columns of random data for linear regression test.

"""
import numpy as np
import pandas as pd
import random
import csv

data = np.random.randint(20, 60, size=(100, 2))
df = pd.DataFrame(data, columns=['random_numbers_1', 'random_numbers_2']).astype(float)

# name of csv file
filename = 'random_number_gen_2cols.csv'

# write to csv file
with open(filename, 'w') as csvfile:
    # create a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(df)
    csvwriter.writerows(data)


# Print test to console
print(df)
#print(df.dtypes)

#
# randomlist = []
# for i in range(0,):
#     n = random.randint(1,30)
# randomlist.append(n)
# print(randomlist)
