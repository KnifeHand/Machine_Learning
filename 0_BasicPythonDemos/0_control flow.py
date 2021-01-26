#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:49:54 2018

"""
#Conditions
#egg distributor
diameter = 8

if diameter >7:
    print("large egg")
    
if diameter > 11:
    print("really large egg")
    
#one condition    
RT = 80
if RT <100:print ("outlier")

fixationDuration = 60
if fixationDuration<80:
    print ("fixation is too small, we should delete it")
    print (fixationDuration)
    
#two conditions
fixationDuration = 100
if fixationDuration <80:
    print ("fixation duration is too short")
else:
    print ("fixation duration is acceptable")
    print (fixationDuration)
    
    
#multi conditions
    
dateOfWeek=1
if dateOfWeek ==1:
    print ("Today is Monday")
elif dateOfWeek ==2:
    print ("Today is Tuesday")
elif dateOfWeek ==3:
    print ("Today is Wednesday")
elif dateOfWeek ==4:
    print ("Today is Thursday")
elif dateOfWeek ==5:
    print ("Today is Friday")
elif dateOfWeek ==6:
    print ("Today is Saturday")
elif dateOfWeek ==7:
    print ("Today is Sunday")
else:
    print ("Please enter a valid number for dateOfWeek.")
    
#for loop
import numpy
data = {}
data['subject-1'] = {'RT':[300, 256, 35], 'acc':[1, 1, 0]}
data['subject-2'] = {'RT':[400, 512, 100009], 'acc':[1, 0, 1]}
data['subject-3'] = {'RT':[732, 542, 839], 'acc':[1, 1, 1]}
print (data)

# find all subject data information using the for loop.
RTList = []
for subjectID in data.keys():
    print (data[subjectID]["RT"])
    RTList.extend(data[subjectID]["RT"])

# calculate mean and std at group level. This is only calculated once as it is out of the indention level of the for group.
print ("mean and standard deviation of RT:",numpy.mean(RTList),numpy.std(RTList))

print ("-----------section separator--------------")
# demostrate indention is part of the grammar of Python
RTListIndividual = []
for subjectID in data.keys():
    print (data[subjectID]["RT"])
    RTListIndividual.extend(data[subjectID]["RT"])
    # calculate mean and std at individual participant level.This is calculated three times as it is within the same indention level of the for group.
    print ("mean and standard deviation of RT:", numpy.mean(RTListIndividual),numpy.std(RTListIndividual))
    

#while loop
import numpy
data = {}
data['subject-1'] = {'RT':[300, 256, 35], 'acc':[1, 1, 0]}
data['subject-2'] = {'RT':[400, 512, 100009], 'acc':[1, 0, 1]}
data['subject-3'] = {'RT':[732, 542, 839], 'acc':[1, 1, 1]}
print (data)

# find all subject data information using the for loop.
subjectIndex=1
RTList = []
while subjectIndex<4:
    key = 'subject-%d'%subjectIndex
    print (data[key]["RT"])
    RTList.extend(data[key]["RT"])
    subjectIndex=subjectIndex+1

# calculate mean and std at group level. This is only calculated once as it is out of the indention level of the for group.
print ("mean and standard deviation of RT:",numpy.mean(RTList),numpy.std(RTList))


#The use of break
import numpy
data = {}
data['subject-1'] = {'RT':[300, 256, 35], 'acc':[1, 1, 0]}
data['subject-2'] = {'RT':[400, 512, 100009], 'acc':[1, 0, 1]}
data['subject-3'] = {'RT':[732, 542, 839], 'acc':[1, 1, 1]}
print (data)

# find all subject data information using the for loop.
subjectIndex=1
RTList = []
while True:
    if subjectIndex>2:break
    key = 'subject-%d'%subjectIndex
    print (data[key]["RT"])
    RTList.extend(data[key]["RT"])
    subjectIndex=subjectIndex+1

# calculate mean and std at group level. This is only calculated once as it is out of the indention level of the for group.
print ("mean and standard deviation of RT:",numpy.mean(RTList),numpy.std(RTList))


#the use of continue
import numpy
data = {}
data['subject-1'] = {'RT':[300, 256, 35], 'acc':[1, 1, 0]}
data['subject-2'] = {'RT':[400, 512, 100009], 'acc':[1, 0, 1]}
data['subject-3'] = {'RT':[732, 542, 839], 'acc':[1, 1, 1]}
print (data)

# find all subject data information using the for loop.
RTList = []
for subjectID in data.keys():
    if subjectID =='subject-2':
        print ("skip 'subject-2'")
        continue
    print (data[subjectID]["RT"])
    RTList.extend(data[subjectID]["RT"])

# calculate mean and std at group level. This is only calculated once as it is out of the indention level of the for group.
print ("mean and standard deviation of RT:",numpy.mean(RTList),numpy.std(RTList))

