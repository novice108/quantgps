#!pip3 install qubovert
import qubovert
import math
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import matplotlib.pyplot as plt
from time import time

#%config InlineBackend.figure_format = 'svg' # Makes the images look nice
#%config InlineBackend.figure_format = 'retina'
#%config InlineBackend.figure_format = 'pdf'

N = 6 #select the value of N
# show the location of the nodes of the problem to solve
M=N+1
puntos = np.random.rand(M,2)
for i in range(M):
    ang = 2*i*np.pi/M
    puntos[i,0],puntos[i,1] = np.cos(ang),np.sin(ang)
#print(puntos)
plt.plot(puntos[:,0],puntos[:,1],'o')

def fnorm(v):  ## v must be a np.array
    return np.sqrt(np.sum(v**2))
dist = np.zeros((N+2,N+2))
for i in range(N):
    for j in range(i+1,N+1):
        aux  =  fnorm(puntos[i,:]-puntos[j,:])
        dist[i,j],dist[j,i] = aux,aux

for j in range(0,N+1):
    i = N+1
    aux  =  fnorm(puntos[0,:]-puntos[j,:])
    dist[i,j],dist[j,i] = aux,aux
print("dist = "dist)
print("N=", N)

#transform the distances into integers
## calculate the best path for nearby neighbors

for N in range(2,10):
    M=N+1
    puntos = np.random.rand(M,2)
for i in range(M):
    ang = 2*i*np.pi/M
    puntos[i,0],puntos[i,1] = np.cos(ang),np.sin(ang)
#print(puntos)
plt.plot(puntos[:,0],puntos[:,1],'o')

def fnorm(v):  ## v must be a np.array
    return np.sqrt(np.sum(v**2))
dist = np.zeros((N+2,N+2))
for i in range(N):
    for j in range(i+1,N+1):
        aux  =  fnorm(puntos[i,:]-puntos[j,:])
        dist[i,j],dist[j,i] = aux,aux

for j in range(0,N+1):
    i = N+1
    aux  =  fnorm(puntos[0,:]-puntos[j,:])
    dist[i,j],dist[j,i] = aux,aux
#print(dist)


start_time = time()
lis_n = range(N+2)
dist_aux  = np.copy(dist)
for i in lis_n:
    dist_aux[i,i] = np.inf
dist_aux = np.copy(dist_aux[:-1,:-1])

i = 0
dist_vc = 0
ord_vc = [i]
for jj in range(N+1):
    dist_aux[jj,0] = np.inf
    
for cont in range(N):
    sig_dist = np.min(dist_aux[i,:])
    dist_vc += sig_dist
    sig = np.where(dist_aux[i,:] == sig_dist)[0][0]
    for jj in range(N+1):
        dist_aux[jj,sig] = np.inf
    i = sig
    ord_vc.append(i)

dist_vc += dist[ord_vc[-1],0]
ord_vc.append(0)
## print(dist_aux)

print("The order by nearest neighbors is ", ord_vc)
print(dist_vc)

## paint the proposed path
plt.plot(puntos[:,0],puntos[:,1],'o')
for i in range(len(ord_vc)-1):
        plt.plot(puntos[(ord_vc[i],ord_vc[i+1]),0],puntos[(ord_vc[i],ord_vc[i+1]),1])
plt.show()
elapsed_time = np.round(time()-start_time,3)
print("The time taken has been ", elapsed_time, "seconds.")

