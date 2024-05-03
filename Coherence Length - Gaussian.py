import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math
from scipy.fft import fft2, fftfreq
import csv
from plyer import notification
from scipy import signal


name = 'w'

name1 = 'BLUE LED 0.001_1.csv'
name2 = 'BLUE LED 0.001_2.csv'
line = 1
#FFT dots
start_dot = -5
end_dot = 9
#range of filting
start_order = 75000
end_order = 127000
#wn_filt:1/1000
#wn_filt:1/100

#blue 75000-127000 v=1.3059725211912945*532/500 x0=10
#'BLUE LED 0.001_1.csv'
#'BLUE LED 0.001_2.csv'

#red 0-120000 v=1.00664384940608*532/500 x0=10
#'red led 1.csv'
#'red led 2.csv'

#white 5000-22000 v=1.4168319637291018*532/500
#WHITE LED 0.0001_2.csv

#white 15000-25000 v=20.746887966804977*532/500
#'WHITE LED 0.01.csv'

wn = 2*1/100  # 截止频率1Hz,采样频率10000Hz
v = 1.3059725211912945*532/500
amplify = 1000

if name == 'w':
    name1 = 'WHITE LED 0.01.csv'
    name2 = ''
    start_dot = -6
    end_dot = 9
    start_order = 15000
    end_order = 25000
    v = 20.746887966804977*532/500
    wn = 2*1/75
    name = 'white'
    uncert = 4.8/100
elif name == 'b':
    name1 = 'BLUE LED 0.001_1.csv'
    name2 = 'BLUE LED 0.001_2.csv'
    start_dot = -6
    end_dot = 9
    start_order = 75000
    end_order = 127000
    v=1.3059725211912945*532/500
    wn = 2*1/1000
    name = 'blue'
    uncert = 6.3/100
elif name == 'r':
    name1 = 'red led 01.csv'
    name2 = 'red led 02.csv'
    start_dot = -6
    end_dot = 9
    start_order = 0
    end_order = 170000
    v = 1.0642128425685138
    wn = 2*1/1200
    name = 'red'
    uncert = 3.4/100

def read_file(name):
    new_data = None
    data = np.genfromtxt(name, delimiter=',',
                         comments='%', dtype='float64')
    new_data = np.zeros([0, len(data[0])])
    for i in data:
        if not np.isnan(i[0]) and not np.isnan(i[1]):
            if np.isfinite(i[0]) and np.isfinite(i[1]):
                new_data = np.vstack((new_data, i))
    return new_data


mins = np.zeros((0, 2))
maxs = np.zeros((0, 2))


def function(parameter):
    L,A = parameter
    sum = 0
    for i in mins:
        sum = sum + (abs(np.exp(-(i[0]-phi)**2/(2*L**2))/np.sqrt(2*np.pi)/L)/A-abs(i[1]))**2
    for i in maxs:
        sum = sum + (abs(np.exp(-(i[0]-phi)**2/(2*L**2))/np.sqrt(2*np.pi)/L)/A-abs(i[1]))**2
    return sum


data = read_file(name1)
if name2 != '':
    data = np.vstack((data, read_file(name2)))
mean = np.mean(data[:,1])
for a in data:
    a[1] = (a[1]-mean)*amplify

ddt = 0
last_place = 0
from_place = 0
DOIT = True
for i in range(1,len(data)):
    if data[i,0]<data[i-1,0] and DOIT:
        last_place = data[i,0]
        from_place = data[i-1,0]
        ddt = data[i+10,0]-data[i+9,0]
        DOIT = False
    data[i,0] = data[i,0]+from_place-last_place+ddt



b,a = signal.butter(4,wn,'low')
filtedData = signal.filtfilt(b,a,data[:,line]).copy()


data0 = data[start_order:end_order:,0]
data1 = filtedData[start_order:end_order:]
#data1 = data[::,line]

#A = (np.max(data1)-np.min(data1))/2
phi = (data0[np.argmax(data1)]+data0[np.argmin(data1)])/2



need_mean = np.mean(data1)

for i in range(1,len(data1)-1):
    if data1[i] > data1[i-1] and data1[i] > data1[i+1] and data1[i] > need_mean:
        #print([data0[i],data1[i]])
        maxs = np.vstack((maxs,[data0[i],data1[i]]))
        if i < np.argmax(data1) and maxs[len(maxs)-1,1] < maxs[len(maxs)-2,1]:
            maxs = np.delete(maxs, len(maxs)-1,axis = 0)
        if i > np.argmax(data1) and maxs[len(maxs)-1,1] > maxs[len(maxs)-2,1]:
            maxs = np.delete(maxs, len(maxs)-2,axis = 0)
        #print(maxs)
    if data1[i] < data1[i-1] and data1[i] < data1[i+1] and data1[i] < need_mean:
        mins = np.vstack((mins,[data0[i],data1[i]]))
        if i < np.argmin(data1) and mins[len(mins)-1,1] > mins[len(mins)-2,1]:
            mins = np.delete(mins, len(mins)-1,axis = 0)
        if i > np.argmin(data1) and mins[len(mins)-1,1] < mins[len(mins)-2,1]:
            mins = np.delete(mins, len(mins)-2,axis = 0)


start = [10,0]
result,A = abs(fmin(function,x0=start))
#A,L,phi = parameter
#print(A,phi,result)
#print("coherence length:",np.pi*result*v//0.01*0.01)
#print(6*result*v)


plt.figure(figsize=(20, 10))
plt.ylim(np.min(data1)*1.2,np.max(data1)*1.7)
plt.xlabel('distance(μm)',fontsize=35)
plt.ylabel('Intensity',fontsize=35)
plt.title('Coherence Length of '+name+' led',fontsize=50)
plt.plot(data[:,0]*v, data[:,line], label='Before Filtered',color = 'gray',linewidth = 3)
plt.plot(data0*v, data1, label='After Filtered',color = 'black',linewidth = 3)
plt.plot(data[:,0]*v,np.exp(-(data[:,0]-phi)**2/(2*result**2))/np.sqrt(2*np.pi)/result/A,color = 'red',linewidth = 10,label = 'fitted envolope, coherence length='+"{:.2f}".format(2*v*np.pi*result*np.sqrt(2*np.log(2)/np.pi))+'±{:.2f}'.format(2*v*np.pi*result*np.sqrt(2*np.log(2)/np.pi)*uncert)+'μm')
plt.plot(data[:,0]*v,-np.exp(-(data[:,0]-phi)**2/(2*result**2))/np.sqrt(2*np.pi)/result/A,color = 'red',linewidth = 10)#,label = 'fitted envolope, coherence length='+str(np.pi*result*v//0.01*0.01)+'μm'
plt.legend(loc='upper right',fontsize=20)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.show()

'''
plt.figure(figsize=(20, 10))
plt.ylim(np.min(data1)*1.2,np.max(data1)*1.7)
plt.xlabel('time(s)',fontsize=35)
plt.ylabel('Intensity',fontsize=35)
plt.title('Coherence Length of '+name+' led',fontsize=50)
plt.plot(data[:,0], data[:,line], label='Before Filtered',color = 'gray',linewidth = 3)
plt.plot(data0, data1, label='After Filtered',color = 'black',linewidth = 3)
plt.plot(data[:,0],np.exp(-(data[:,0]-phi)**2/(2*result**2))/np.sqrt(2*np.pi)/result/A,color = 'red',linewidth = 10,label = 'fitted envolope, coherence length='+"{:.2f}".format(2*v*np.pi*result*np.sqrt(2*np.log(2)/np.pi))+'μm')
plt.plot(data[:,0],-np.exp(-(data[:,0]-phi)**2/(2*result**2))/np.sqrt(2*np.pi)/result/A,color = 'red',linewidth = 10)#,label = 'fitted envolope, coherence length='+str(np.pi*result*v//0.01*0.01)+'μm'
plt.legend(loc='upper right',fontsize=20)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.show()
'''
notification.notify(title='Mission Complete',message = 'your Coherence Length Program has just completed the mission, please check it soon!',app_name = 'My Python App',timeout = 10)