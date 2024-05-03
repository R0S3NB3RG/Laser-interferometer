import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math
from scipy.fft import fft2, fftfreq
import csv
from plyer import notification
from cProfile import label
import pandas as pd
from scipy import signal
import torch

openning_file = 'red laser.csv'
num = 4000   #resolution
start_place = 400
end_place = 800
#green: 400-600; num: 10000
#red: 550-750; num: 10000
#best: 0-1000; num: 10000


def read_file(name):
    new_data = None
    
    data = np.genfromtxt(name, delimiter=',',comments='%', dtype='float64')
    new_data = np.zeros([0, len(data[0])])
    for i in data:
        if not np.isnan(i[0]) and not np.isnan(i[1]):
            if np.isfinite(i[0]) and np.isfinite(i[1]):
                new_data = np.vstack((new_data, i))
        
    return new_data


new = np.array([], dtype=float)
stack = np.zeros(num)
sum = 0


data = read_file(openning_file)

for i in range(0,len(data)):
    new_data = data[i][~np.isnan(data[i])]
    #new_data = data
    #print(new_data)
    new = np.hstack((new,new_data))
    for j in new_data:
        if j<0:
            j =-j
        order = int((j-start_place)//((end_place-start_place)/num))
        #print(order)
        stack[order] = stack[order]+1
'''


with open(openning_file, 'r', encoding='utf-8') as file:
    data = file.read()

data = data.strip('\ufeff')

data = data.replace('\n', ',')

data = filter(None, data.split(","))

data = [float(num) for num in data]

for i in range(0,1):
    new_data = data
    print(new_data)
    new = np.hstack((new,new_data))
    if new_data[i]<0:
        new_data[i] =-new_data[i]
    order = int((new_data[i]-start_place)//((end_place-start_place)/num))
    stack[order] = stack[order]+1
'''
index = np.argmax(stack)
max_index = index*((end_place-start_place)/num)+start_place
#print('wavelength:',max_index,stack[index])


#print(new)
#print(np.sum(new))
new = np.abs(new)
mean = np.sum(new)/len(new)
#print(new)
#print(np.sum(new))
for i in new:
    sum = sum +(i-max_index//0.1/10)**2
uncertainty = (sum/len(new))**(1/2)


#print(uncert)
print('wavelength: '+str(max_index//0.1/10)+'±'+str(uncertainty//0.1/10)+'nm')
print('mean: '+str(mean//0.1/10)+'nm')

plt.figure(figsize=(30, 20))
plt.plot(np.arange(start_place, end_place,((end_place-start_place)/num)),stack)
plt.xlabel('wavelength(nm)',fontsize=60)
plt.ylabel('weights',fontsize=60)
plt.title('Stack Plot(wavelength distribution of red laser)',fontsize=80)
plt.scatter(max_index, stack[index],c = 'r')
plt.ylim(-5,stack[index]*1.1)
plt.xlim(550,750)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.text(max_index-(end_place-start_place)/10, stack[index]*1.02,'wavelength:'+str(max_index//0.1/10)+'±'+str(uncertainty//0.1/10)+'nm', fontsize=35, color='red')
plt.show()
