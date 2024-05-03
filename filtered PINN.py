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

openning_file = 'day3 polarised green 350_05.csv'
writting_file = 'green laser.csv'
save = 'Intensity(temp).pth'
load = 'Intensity(temp).pth'#Intensity(n) green min = 0 new.pth
seed = 8888
v = 1
amplify = 100
loops = 300000
gap = 50
load_yes_or_not = 'y'
up_side_down = 'n'
order_i = 3




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

data = read_file(openning_file)
wn = 2*1/5000  # 截止频率1Hz,采样频率10000Hz
b,a = signal.butter(4,wn,'low')
plt.figure('current')
filtedData = signal.filtfilt(b,a,data[:,1]).copy()
plt.plot(data[:,0], data[:,1], label='Filtered Before')
plt.plot(data[:,0], filtedData, label='Filtered After')
plt.show()
#--------------------------------------------------------------------------------------------------


max_order_list = []
min_order_list = []
max_list = []
min_list = []

max = -100000000
min = 100000000

for i in range(0,len(filtedData)):
    if filtedData[i] > max:
        max = filtedData[i]
    if filtedData[i] < min:
        min = filtedData[i]
        
        
amplitude = (max-min)/2*amplify
mean = (max+min)/2
for i in range(0,len(filtedData)):
    if up_side_down == 'n':
        data[i,1] = (filtedData[i]-mean)*amplify
    elif up_side_down == 'y':
        data[i,1] = (-filtedData[i]+mean)*amplify
    data[i,0] = data[i,0]/v



for i in range(1,len(data[:,1])-1):
    if data[i,1] > data[i+1,1] and data[i,1] > data[i-1,1]:
        max_order_list.append(i)
        max_list.append(data[i,1])
    if data[i,1] < data[i+1,1] and data[i,1] < data[i-1,1]:
        min_order_list.append(i)
        min_list.append(data[i,1])

print('max order:',max_order_list)
print('max list:',max_list)
print()
print('min order:',min_order_list)
print('min list:',min_list)
print()
print()


amplitude = (sum(max_list)/len(max_list)-sum(min_list)/len(min_list))/2
print('amplitude:',amplitude)
print()
print()

datai = []
dataj = []

place_1 = min_order_list[order_i]
place_2 = min_order_list[order_i+1]


if place_1<place_2:
    start_point = place_1
    end_point = place_2
    datai = torch.tensor(data[start_point:end_point:gap,0], dtype=torch.float32).unsqueeze(1)
    dataj = torch.tensor(data[start_point:end_point:gap,1], dtype=torch.float32).unsqueeze(1)
if place_1>place_2:
    start_point = place_2
    end_point = place_1
    datai = torch.tensor(data[start_point:end_point:gap,0], dtype=torch.float32).unsqueeze(1)
    dataj = torch.tensor(data[start_point:end_point:gap,1], dtype=torch.float32).unsqueeze(1)


def intensity():
    x = datai
    return x.requires_grad_(True)

def order_2(u):
    x = intensity()
    A = u(x)[:,0]
    k = u(x)[:,1]
    K = (sum(k)/len(k))*torch.ones_like(k)
    #Mean = torch.ones_like(A)*mean
    return loss(gradients(A, x, 2)/(K**2), -dataj)

def order_1(u):
    x = intensity()
    A = u(x)[:,0]
    k = u(x)[:,1]
    #K = (sum(k)/len(k))*torch.ones_like(k)
    Amplitude = torch.ones_like(A)*amplitude
    #Mean = torch.ones_like(A)*mean
    return loss(dataj**2 + (gradients(A, x, 1)/k)**2,Amplitude**2)

def order_0(u):
    x = intensity()
    A = u(x)[:,0]
    Amplitude = torch.ones_like(A)*amplitude
    return loss(torch.log10(A+10*Amplitude),torch.log10(dataj+10*Amplitude))


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,50),
            torch.nn.Tanh(),
            torch.nn.Linear(50,2))
    
    def forward(self,x):
        return self.net(x)


loss = torch.nn.MSELoss()


def gradients(u,x,order):
    if order == 1:
        return torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph = True, only_inputs = True, )[0]
    else:
        if order > 1:
            du_dx = gradients(u, x, 1)
            return gradients(du_dx, x, order - 1)
        else:
            return False
    


if load_yes_or_not == 'y':
    u = torch.load(load)
elif load_yes_or_not == 'n':
    u = MLP()

u.train()
opt = torch.optim.Adam(params=u.parameters())
loss_values = np.array([])
min_loss_values = np.array([])
min_loss_values_order = np.array([])

l_min = 10**8


for i in range(loops):
    opt.zero_grad()
    #order_00 = torch.log10(order_0(u))
    l = torch.log10(order_0(u))+torch.log10(order_1(u))+torch.log10(order_2(u))#+torch.log10(order_2_add(u))+torch.log10(order_1_add(u))#+torch.log10(not_zero(u))
    l.backward()
    loss_values = np.append(loss_values,l.item())
    
    opt.step()
    if i % (loops/100) == 0:
        #print((100*i)/loops,'%  completed','     ','loss:',l.item(),'      current min loss:',l_min)
        print((100*i)/loops,'%  completed','      current min loss:',l_min)
        #print(order_0(u).item(), order_1(u).item(), order_2(u).item())
        print()
    if l_min > l.item():
        l_min = l.item()
        min_loss_values = np.append(min_loss_values,l_min)
        min_loss_values_order = np.append(min_loss_values_order,i)
        torch.save(u, save)
    if np.isnan(l.item()):
        u = torch.load(save)
        break

print()
print()
print()
u = torch.load(save)
#--------------------------------------------------------------------------------------------------

A_pred = u(datai)[:,0]
k_pred = u(datai)[:,1]

mean_wave_number = (sum(k_pred.detach().numpy())/len(k_pred.detach().numpy()))
print('best fit wave loss(l):',l_min)
print('best fit wave number(k):',mean_wave_number)
print('best fit wave length(λ):',2000*np.pi*v/mean_wave_number)

plt.figure(figsize=(30, 10))
plt.subplot(1, 3, 3)
plt.scatter(range(loops), loss_values,color ='green',s = 5)
plt.plot(min_loss_values_order,min_loss_values,color = 'red',linewidth=2)
plt.xlabel('iterations',fontsize=15)
plt.ylabel('loss values',fontsize=15)
plt.title('Loss',fontsize=20)
plt.subplot(1, 3, 1)
plt.scatter(datai.detach().numpy(), A_pred.detach().numpy(),color ='green',s = 20)
plt.scatter(datai.detach().numpy(), dataj.detach().numpy(),color ='red',s = 20)
plt.xlabel('Δ distance',fontsize=15)
plt.ylabel('intensity',fontsize=15)
plt.title('Intensity',fontsize=20)
plt.subplot(1, 3, 2)
plt.scatter(datai.detach().numpy(), k_pred.detach().numpy(),color ='green',s = 20)
plt.xlabel('Δ distance',fontsize=15)
plt.ylabel('k',fontsize=15)
plt.title('Wave number',fontsize=20)
plt.show()
#--------------------------------------------------------------------------------------------------


with open(writting_file, 'r', newline='', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    line = list(reader)

k_wave = 2000*np.pi*v/k_pred.detach().numpy()
line.append(k_wave)

with open(writting_file, 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerows(line)


    

#--------------------------------------------------------------------------------------------------

notification.notify(title='Mission Complete',message = 'your PINN has just completed the mission, please check it soon!',app_name = 'My Python App',timeout = 10)