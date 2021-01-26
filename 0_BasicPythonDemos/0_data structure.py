# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("hello world!")

#for input
user_number = ''
while not user_number.isdigit(): 
    user_number = input("Please input a number to get the evens up to: ")
user_number = int(user_number)

    
#pylab inline

#DATA STRUCTURE
alist = ['I', 'love', 'Python']
blist=[1,2,3,4,5,6,7,8,9,10]
clist=list()
dlist = [alist,blist,clist]
#simple index and slice
blist[0]
blist[1]
blist[-1]
blist[-2]
blist[0:4]
blist[:4]
blist[7:]
blist[7:10]

# multiple slice and index
dlist[0]
dlist[0][1]
dlist[0][1][1]

# slice with step parameter
blist[::2]
blist[1::2]
blist[-1::-1]
blist[-1::-2]
blist[-2::-2]

a = [10, 20, 30, 40]
a.index(10)
a.reverse()
print (a)
# delete list members
a.remove(20)
print(a)
a.pop(0)
print(a)

#some common functions
# add list members
a = [10, 20, 30]
a.append(40)
print(a)
a = [10, 20, 30]
b = [40, 50, 60]
a.extend(b)
print (a)

#tuples---list that can't be changed

alist = ["I","Love","Python", "Still"]
atuple= ("I","Love","Python", "Still")

# we can do index and slice for both tuple and list
alist.count("Love")
atuple.count("Love")
alist.index("Love")
atuple.index("Love")
alist[0]
atuple[0]

# list is mutable, while tuple is immutable.
# thus, we can change a list, but cannot change a tuple.
alist.append("very much")
#atuple.append("very much")


#dictionary
adict={"Feng":100,"Robot":99,"Monkey King":59}
print (adict.keys())
print (adict.values())
adict["Chao"] =101
print (adict)

adict["Feng"]


#numpy array
import numpy
bArray = numpy.array([1, 2, 3, 4]) 
print (bArray)

# index and slice a numpy array
bArray[0] 
bArray[-1]
bArray[1:3]

# +,-,*,/ common mathematical operation for numpy array.
bArray+2
bArray- 2
bArray* 3
bArray / 2.0

# manipulate a specific element of numpy array
bArray[0] += 1
bArray[1:3] += 10
print (bArray)

RT = numpy.array([1001, 1510, 1203, 905,897]) 
numpy.mean(RT)
numpy.median(RT)
numpy.std(RT)
numpy.var(RT)
