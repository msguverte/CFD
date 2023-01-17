# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:43:05 2022

@author: Mustafa
"""

import numpy as np 
import matplotlib.pyplot as plt 

#import plotly.graph_objects as go
N=192
nu=0.0008

import scipy.io

mat1=scipy.io.loadmat('uvw_physical.mat')
mat2=scipy.io.loadmat('uvw_fourier.mat')    
u=mat1['u']
v=mat1['v']
w=mat1['w']   
uk=mat2['uk']
vk=mat2['vk']
wk=mat2['wk']

#Pre-emptive function to scale the inverse 
def inv(mk):
    return np.fft.ifftn(mk)*(192**3)


x=np.linspace(0,2*np.pi*(1-1/N),N)   
y=np.linspace(0,2*np.pi*(1-1/N),N)  
z=np.linspace(0,2*np.pi*(1-1/N),N)
[X,Y,Z]=np.meshgrid(x,y,z)

xi_i=np.concatenate((np.linspace(0,96-1,96), np.linspace(-96,-1,96)), axis=None)
xix = xi_i
xiy = xi_i
xiz = xi_i
[xix,xiy,xez]=np.meshgrid(xi_i,xi_i,xi_i)
Xi = np.meshgrid(xi_i,xi_i,xi_i)

'~~~~~~~~~~question2~~~~~~~~~~'

exprms = np.exp(1j*xix*x)*np.exp(1j*xiy*y)*np.exp(1j*xiz*z)
nax = np.newaxis 
a = (1j * xix * exprms)[:,nax,nax] * uk
b = (1j * xiy * exprms)[nax,:,nax] * vk
c = (1j * xiz * exprms)[nax,nax,:] * wk

all_sums = a + b + c
rmspre = np.sum(all_sums * all_sums)
rms = np.real(rmspre)/(192**3)

#rmsphys = np.sum(np.gradient(u)[0]**2 + np.gradient(u)[1]**2 + np.gradient(u)[2]**2) +np.sum(np.gradient(v)[0]**2 + np.gradient(v)[1]**2 + np.gradient(v)[2]**2) +np.sum(np.gradient(w)[0]**2 + np.gradient(w)[1]**2 + np.gradient(w)[2]**2)
#rmsphys = rmsphys/(192**3)

'~~~~~~~~~~Question 4~~~~~~~~~~'

boxaveragespectral=np.mean(0.5*u**2 + 0.5*v**2 + 0.5*w**2)
boxaveragephysical=np.sum(0.5*uk*np.conj(uk) + 0.5*vk*np.conj(vk) + 0.5*wk*np.conj(wk))

'~~~~~~~~~~question 5~~~~~~~~~~'

dissipationrate=np.real(nu*np.sum((xix**2+xiy**2+xiz**2)*uk*np.conj(uk))+nu*np.sum((xix**2+xiy**2+xiz**2)*vk*np.conj(vk))+nu*np.sum((xix**2+xiy**2+xiz**2)*wk*np.conj(wk)))

'~~~~~~~~~~question 6~~~~~~~~~~'

ximod =np.sqrt(xix**2+xiy**2+xiz**2)
meshxi = np.linspace(0,167,167)
Exi = np.zeros(167)
def modxi(i,j,k):
    return np.sqrt(i**2 + j**2 + k**2)

for i in range(192):
    for j in range(192):
        for k in range(192):
            usal = [uk[i,j,k],vk[i,j,k],wk[i,j,k]]
            n = round(ximod[i,j,k])
            Exi[n] += 0.5*np.dot(usal,np.conj(usal))
         
for i in range(96,167):
    Exi[i] = -Exi[i]  

'~~~~~~~~~~question 7~~~~~~~~~~'

Dissipationspectrum=2*nu*(np.real(sum(sum((ximod**2)*0.5*uk*np.conj(uk))))
              + np.real(sum(sum((ximod**2)*0.5*vk*np.conj(vk))))\
        +np.real(sum(sum((ximod**2)*0.5*wk*np.conj(wk)))))
Dissipationspectrum=2*nu*(meshxi**2)*Exi
Dissipationrate2=np.sum(Dissipationspectrum)

'~~~~~~~~~~question 8~~~~~~~~~~'

meshxi = np.linspace(0,167,167)
kolmogorovconstant=Exi[4:45]*(dissipationrate**(-2/3))*meshxi[4:45]**(5/3)
Kolmogorovconstant_mean=np.mean(kolmogorovconstant)
Energydensity=Kolmogorovconstant_mean*(dissipationrate**(2/3))*meshxi**(-5/3)

'~~~~~~~~~~question 9~~~~~~~~~~'

Reynold_large=(boxaveragespectral**2)/(Dissipationrate2*nu)
lambdag=np.sqrt(10*nu*boxaveragespectral/Dissipationrate2)
Reynold_small=lambdag*np.sqrt(2*boxaveragespectral/3)/nu

'~~~~~~~~~~queston10~~~~~~~~~~'

NLES=24
filterwidth=2*np.pi*(1-1/NLES)/(NLES-1)
xin=np.pi/filterwidth

def filter(x):
    if x>12:
        a = 0
    elif x==0:
        a= 1
    else:
        a = (((xin*2)/(np.pi * x))*np.sin((x*np.pi)/(2*xin)))
    return a

filterz = np.linspace(0,167,167)
for i in range(0,167):
    filterz[i] = filter(i)
  
ukf = np.zeros((192,192,192))
vkf = np.zeros((192,192,192))
wkf = np.zeros((192,192,192))

for i in range(192):
    for j in range(192):
        for k in range(192):
            inp = ximod[i,j,k]
            ukf[i,j,k] = uk[i,j,k]*filter(inp)
            vkf[i,j,k] = vk[i,j,k]*filter(inp)
            wkf[i,j,k] = wk[i,j,k]*filter(inp)

meshxif = np.linspace(0,167,167)
Exif = np.zeros(167)

for i in range(167):
    Exif[i] = Exi[i]*filter(i)

Disspecf=2*nu*(meshxif**2)*Exif    

'~~~~~~~~~~queston11~~~~~~~~~~'

pltval=31

uf=inv(ukf)
vf=inv(vkf)
wf=inv(wkf)

curl=inv(1j*xix*vk)- inv(1j*xiy*uk) 

curlf=inv((1j*xix*ukf))  -inv((1j*xiy*vkf)) 

magni=np.sqrt(u**2 + v**2 + w**2)
magnif=np.sqrt(uf**2+vf**2+wf**2) #(magnificent init)

'~~~~~~~~~~queston12~~~~~~~~~~'   

tSGSXX = inv(ukf*ukf) - inv(ukf)*inv(ukf)        
tSGSXY = inv(ukf*vkf) - inv(ukf)*inv(vkf)  
tSGSXZ = inv(ukf*wkf) - inv(ukf)*inv(wkf)    
tSGSYY = inv(vkf*vkf) - inv(vkf)*inv(vkf)      
tSGSYZ = inv(vkf*wkf) - inv(vkf)*inv(wkf)                                     
tSGSZZ = inv(wkf*wkf) - inv(wkf)*inv(wkf)               

'~~~~~~~~~~queston13 tensor~~~~~~~~~~' 

dukdxf =  inv(1j*xix*ukf)         
dukdzf =  inv(1j*xix*ukf)            
dukdyf =  inv(1j*xix*ukf)  
dvkdxf =  inv(1j*xix*vkf)   
dvkdyf =  inv(1j*xix*vkf)          
dvkdzf =  inv(1j*xix*vkf)      
    
dwkdxf =  inv(1j*xix*wkf)         
dwkdyf =  inv(1j*xix*wkf)    
dwkdzf =  inv(1j*xix*wkf)             
  
Cs=0.173

def sgs(ui,uj,dui,duj):
    return -2*((Cs*filterwidth)**2)*np.sqrt(0.25*(ui+uj)**2)*(duj+dui)/2

SGSXX=sgs(uf,uf,dukdxf,dukdxf)
SGSXY=sgs(uf,vf,dukdyf,dvkdxf)
SGSXZ=sgs(uf,wf,dukdzf,dwkdxf)

SGSYY=sgs(vf,vf,dvkdyf,dvkdyf)
SGSYZ=sgs(vf,wf,dvkdzf,dwkdyf)
SGSZZ=sgs(wf,wf,dwkdzf,dwkdzf)

'~~~~~~~~~~queston13 Viscosity~~~~~~~~~~'
  
def sgsvisc(dui,duj):
      return (Cs*filterwidth)*(.5*duj+.5*dui)/2

nusgsxx=sgs(dukdxf,dukdxf)
nusgsxy=sgs(dvkdxf,dukdyf)
nusgsxz=sgs(dwkdxf,dukdzf)
nusgsyy=sgs(dvkdyf,dvkdyf)
nusgsyz=sgs(dwkdyf,dvkdzf)
nusgszz=sgs(dwkdzf,dwkdzf)

'~~~~~~~~~~queston 14~~~~~~~~~~' 

ukff = ukf
vkff = vkf 
wkff = wkf 

ukfukf = ukf*uk
ukfvkf = ukf*uk
ukfwkf = ukf*uk
vkfvkf = vkf*vk
vkfwkf = vkf*wk
wkfwkf = wkf*wk

for i in range(192):
    for j in range(192):
        for k in range(192):
            inp = ximod[i,j,k]
            ukff[i,j,k] = ukf[i,j,k]*filter(inp)
            vkff[i,j,k] = vkf[i,j,k]*filter(inp)
            wkff[i,j,k] = wkf[i,j,k]*filter(inp)
            ukfukf[i,j,k] = ukfukf[i,j,k]*filter(inp)
            ukfvkf[i,j,k] = ukfvkf[i,j,k]*filter(inp)
            ukfwkf[i,j,k] = ukfwkf[i,j,k]*filter(inp)
            vkfvkf[i,j,k] = vkfvkf[i,j,k]*filter(inp)
            vkfwkf[i,j,k] = vkfwkf[i,j,k]*filter(inp)
            wkfwkf[i,j,k] = wkfwkf[i,j,k]*filter(inp)
        
uff = inv(ukff)
vff = inv(vkff)
wff = inv(wkff)

tbSGSXX = inv(ukfukf) - inv(ukff)*inv(ukff)                      
tbSGSXY = inv(ukfvkf) - inv(ukff)*inv(vkff)                   
tbSGSXZ = inv(ukfwkf) - inv(ukff)*inv(wkff)                        
tbSGSYY = inv(vkfvkf) - inv(vkff)*inv(vkff)                        
tbSGSYZ = inv(vkfwkf) - inv(vkff)*inv(wkff)                 
tbSGSZZ = inv(wkfwkf) - inv(wkff)*inv(wkff)                     
