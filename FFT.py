import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math
from scipy.fft import fft2, fftfreq
import csv
from plyer import notification
from scipy import signal

name = 'filtered wave and unfiltered wave'

name1 = 'green_1.csv'
name2 = ''
line = 1
#FFT dots
start_dot = -6
end_dot = 9
#range of filting
start_order = 0
end_order = 100000000
wn = 2*1/10000  # 截止频率1Hz,采样频率10000Hz

v = 1

if name == 'w':
    name1 = 'WHITE LED 0.01.csv'
    name2 = ''
    start_dot = -6
    end_dot = 9
    start_order = 15000
    end_order = 25000
    v = 20.746887966804977*532/500
    wn = 2*1/75
    name = 'white led'
elif name == 'b':
    name1 = 'blue 01.csv'
    name2 = ''
    start_dot = -6
    end_dot = 9
    start_order = 55000
    end_order = 1000000
    v = 1.596159615961596
    wn = 2*1/1000
    name = 'blue led'
elif name == 'r':
    name1 = 'red led 01.csv'
    name2 = 'red led 02.csv'
    start_dot = -6
    end_dot = 9
    start_order = 0
    end_order = 170000
    v = 1.0642128425685138
    wn = 2*1/1200
    name = 'red led'

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


data = read_file(name1)
if name2 != '':
    data = np.vstack((data, read_file(name2)))


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

ed = int(end_order*1.2//1)
if ed > len(data):
    ed = len(data)

plt.figure(figsize=(30, 20))
plt.plot(data[int(start_order*0.8//1):ed,0]*v, data[int(start_order*0.8//1):ed,line], label='Before Filtered',linewidth = 10)
plt.plot(data0*v, data1, label='After Filtered',linewidth = 10)
plt.xlabel('distance',fontsize=60)
plt.ylabel('Intensity',fontsize=60)
plt.xticks([])
plt.yticks([])
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.title('Waveform of '+name+'',fontsize=80)
plt.legend(loc='upper left', fontsize=40, borderpad=1.5)
plt.show()


dt = data0[1]-data0[0]
fft = np.fft.fft(data1)
#fft = np.fft.fft(filtedData)
frequency = np.fft.fftfreq(len(data0))/dt

#print(fft)
#print(frequency)
#print(dt)
# 找到傅立叶变换绝对值中的最大值的索引
start_index = np.argmax(np.abs(fft))+1 # 使用该索引从频率数组中找到相应的频率
max_index = np.argmax(np.abs(fft[start_index:]))


start_place = max_index+start_index+start_dot
end_place = max_index+start_index+end_dot
#print(max_index+start_index+start_dot,max_index+start_index+end_dot)
if start_place < 0:
    start_place = 0
if end_place > len(frequency):
    end_place = len(frequency)

new_frequency = np.zeros_like(frequency[start_place:end_place])
new_fft = np.zeros_like(frequency[start_place:end_place])


for i in range(0,len(new_frequency)):
    new_frequency[i] = abs(v*1000/frequency[start_place:end_place][i])
    new_fft[i] = abs(fft[start_place:end_place][i])

#np6 = new_frequency[-start_dot-6]
#np5 = new_frequency[-start_dot-5]
#np4 = new_frequency[-start_dot-4]
#np3 = new_frequency[-start_dot-3]
np2 = new_frequency[-start_dot-2]
np1 = new_frequency[-start_dot-1]
n = new_frequency[-start_dot]
nm1 = new_frequency[-start_dot+1]
nm2 = new_frequency[-start_dot+2]
#nm3 = new_frequency[-start_dot+3]
#nm4 = new_frequency[-start_dot+4]
#nm5 = new_frequency[-start_dot+5]

#print(peak_frequency,next_frequency)
print('wavelength: '+str(n//0.1/10)+'±'+str(abs(np1-nm1)//0.2/10)+'nm')


plt.figure(figsize=(30, 20))
plt.plot(new_frequency,new_fft)
plt.xlabel('wavelength(nm)',fontsize=60)
plt.ylabel('weights',fontsize=60)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.ylim(0,abs(fft[max_index+start_index]*1.1))
plt.title('Stack Plot(wavelength distribution of '+name+')',fontsize=80)

plt.scatter(n, abs(fft[max_index+start_index]),c = 'r')
plt.text(n*0.98, abs(fft[max_index+start_index])*1.02,str(n//0.1/10)+'±'+str(abs(np1-nm1)//0.2/10)+'nm', fontsize=40, color='red')


'''
plt.scatter(np2, abs(fft[max_index+start_index]),c = 'r')
plt.text(np2*0.98, abs(fft[max_index+start_index])*1.02,str(np2//0.1/10)+'±'+str(abs(np3-np1)//0.2/10)+'nm', fontsize=40, color='red')
'''


#plt.scatter(np1, abs(fft[max_index+start_index-1]),c = 'r')
#plt.text(np1*0.98, abs(fft[max_index+start_index-1])*1.02,str(np1//0.1/10)+'±'+str(abs(np2-n)//0.2/10)+'nm', fontsize=40, color='red')
#plt.scatter(np2, abs(fft[max_index+start_index-2]),c = 'r')
#plt.text(np2*0.98, abs(fft[max_index+start_index-2])*1.02,str(np2//0.1/10)+'±'+str(abs(np3-np1)//0.2/10)+'nm', fontsize=40, color='red')
#plt.scatter(np3, abs(fft[max_index+start_index-3]),c = 'r')
#plt.text(np3*0.98, abs(fft[max_index+start_index-3])*1.02,str(np3//0.1/10)+'±'+str(abs(np4-np2)//0.2/10)+'nm', fontsize=40, color='red')
#plt.scatter(np4, abs(fft[max_index+start_index-4]),c = 'r')
#lt.text(np4*0.98, abs(fft[max_index+start_index-4])*1.02,str(np4//0.1/10)+'±'+str(abs(np5-np3)//0.2/10)+'nm', fontsize=40, color='red')
#plt.scatter(np5, abs(fft[max_index+start_index-5]),c = 'r')
#plt.text(np5*0.98, abs(fft[max_index+start_index-5])*1.02,str(np5//0.1/10)+'±'+str(abs(np6-np4)//0.2/10)+'nm', fontsize=40, color='red')


#plt.scatter(nm1, abs(fft[max_index+start_index+1]),c = 'r')
#plt.text(nm1*0.98, abs(fft[max_index+start_index+1])*1.02,str(nm1//0.1/10)+'±'+str(abs(nm2-n)//0.2/10)+'nm', fontsize=40, color='red')
#plt.scatter(nm2, abs(fft[max_index+start_index+2]),c = 'r')
#plt.text(nm2*0.98, abs(fft[max_index+start_index+2])*1.02,str(nm2//0.1/10)+'±'+str(abs(nm3-nm1)//0.2/10)+'nm', fontsize=40, color='red')
#plt.scatter(nm3, abs(fft[max_index+start_index+3]),c = 'r')
#plt.text(nm3*0.98, abs(fft[max_index+start_index+3])*1.02,str(nm3//0.1/10)+'±'+str(abs(nm4-nm2)//0.2/10)+'nm', fontsize=40, color='red')

plt.show()
if line ==2:
    print('velocity should be: ',532.0*v/(n//0.1/10))

notification.notify(title='Mission Complete',message = 'your FFT has just completed the mission, please check it soon!',app_name = 'My Python App',timeout = 10)