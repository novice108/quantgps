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
print("dist = ", dist)
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

#SIMULATION
nsim = 1
SOLUCIONES = np.zeros((nsim,3))
print(SOLUCIONES)
np.savetxt("Modelo_General.txt", SOLUCIONES)
lectura = np.loadtxt("Modelo_General.txt")
#print("lectura ", lectura)

#General Model Simulation
## Simulation General modeling
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

    ##  Painting the proposed path
    plt.plot(puntos[:,0],puntos[:,1],'o')
    for i in range(len(ord_vc)-1):
            plt.plot(puntos[(ord_vc[i],ord_vc[i+1]),0],puntos[(ord_vc[i],ord_vc[i+1]),1])
    plt.show()
    elapsed_time = np.round(time()-start_time,3)
    print("The time taken has been ",elapsed_time, "seconds.")
    ######
    from neal import SimulatedAnnealingSampler
    modelo = 0 ## Put 0 for SA, 1 for simulated QA, 2 for real QA
    n_samples = 2

    stm1 = time() ## start time model 1
    pen = 2
    pen1 = 4

    Q = qubovert.QUBO()
    for u in range(N+2):
        for v in range(N+2):
            if u !=v:
                for t in range(N+1):
                    Q.create_var(f"x_{u}_{v}_{t}")

    for v in range(N+2):
        for t in range(N):
            Q.create_var(f"a_{v}_{t}")


    nqm1 = (N+2)*(N+2)*(N+1) ## Nnumber of qubits model 1

    ## Restriction 1: We leave once from each city
    for u in range(N+1):
        lambda_1 = pen1*np.max(dist[u,])
        for v1 in range(1,N+2):
            if u!=v1:
                for t1 in range(N+1):
                    for v2 in range(1,N+2):
                        if u!=v2:
                            for t2 in range(N+1):
                                Q[(f"x_{u}_{v1}_{t1}",f"x_{u}_{v2}_{t2}")] += lambda_1
                    Q[(f"x_{u}_{v1}_{t1}",)] += -2*lambda_1
    ## Minimo $-lambda_1*(N+1)$

    ## Restriction 2: We arrive once in each city
    for v in range(1,N+2):
        lambda_2 = np.max(dist[:,v])*pen1
        for u1 in range(N+1):
            if v!=u1:
                for t1 in range(N+1):
                    for u2 in range(N+1):
                        if v!=u2:
                            for t2 in range(N+1):
                                Q[(f"x_{u1}_{v}_{t1}",f"x_{u2}_{v}_{t2}")] += lambda_2
                    Q[(f"x_{u1}_{v}_{t1}",)] += -2*lambda_2
    ## Minimo -lambda_2*(N+1)

    ## Restriction 3: At each instant we can only be on one edge
    lambda_3 = dist_vc*pen
    for t in range(N+1):
        for u1 in range(N+2):
            for v1 in range(N+2):
                if u1 !=v1:
                    for u2 in range(N+2):
                        for v2 in range(N+2):
                            if u2!=v2:
                                Q[(f"x_{u1}_{v1}_{t}",f"x_{u2}_{v2}_{t}")] += lambda_3
                    Q[(f"x_{u1}_{v1}_{t}",)] += -2*lambda_3
    ## Minimo -lambda_3*(N+1)

    ## Restriction 4: We avoid cycles
    lambda_4 = dist_vc*pen

    ## variables a_{v,t} = \sum x_{v,w,t+1}
    for v in range(N+2):
        for t in range(N):
            Q[(f"a_{v}_{t}",)] += lambda_4
            for w in range(1,N+2):
                if v!=w:
                    Q[(f"a_{v}_{t}",f"x_{v}_{w}_{t+1}")] += -2*lambda_4
            for w1 in range(1,N+2):
                if w1 !=v:
                    for w2 in range(1,N+2):
                        if v != w2:
                            Q[(f"x_{v}_{w1}_{t+1}",f"x_{v}_{w2}_{t+1}")] += lambda_4

    for t in range(N):
        for u in range(N+2):
            for v in range(N+2):
                if u!=v:
                    Q[(f"x_{u}_{v}_{t}",f"a_{v}_{t}")] += -lambda_4
                    Q[(f"x_{u}_{v}_{t}",)] += +lambda_4



    # Objective Function
    lambda_obj = 1
    for u in range(N+2):
        for v in range(N+2):
            if u != v:
                for t in range(N+1):
                    Q[(f"x_{u}_{v}_{t}",)] += lambda_obj*dist[u,v]


    dwave_dic = {}
    for i in Q:
        if len(i) == 1:
            dwave_dic[(i[0],i[0])] = Q[i]
        else:
            dwave_dic[i] = Q[i]



    ### We perform the simulation
    from neal import SimulatedAnnealingSampler
    #from dwave.system import DWaveSampler, EmbeddingComposite

    for jjj in range(nsim):
        if modelo == 0:
            sampleset = qubovert.sim.anneal_qubo(dwave_dic, num_anneals=n_samples)
            solution = sampleset.best.state

        if modelo == 1:
            sampler = SimulatedAnnealingSampler()
            #sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample_qubo(dwave_dic, num_reads = n_samples)
            solution = sampleset.first.sample

        if modelo == 2:
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample_qubo(dwave_dic,num_reads = n_samples)
            solution = sampleset.first.sample

        ## We check the results
        print()
        print("We are in the simulation",jjj, "of ",nsim)
        print()
        print("The number of qubits is",nqm1)
        mat_sol = np.zeros((N+2,N+2))
        for u in range(N+2):
            for v in range(N+2):
                if u!=v:
                    for t in range(N+1):
                        if solution[f"x_{u}_{v}_{t}"] == 1:
                            mat_sol[u,v] = 1
        #print(mat_sol)
        print()

        ## We paint the proposed path
        print("Solution path drawing.")
        plt.plot(puntos[:,0],puntos[:,1],'o')
        vaux = np.array(list(range(N+2)))
        suma_ruta = 0
        for i in range(N+1):
            sig_aux = mat_sol[i,:]==1
            if np.sum(sig_aux) > 0:
                sig = (int(vaux[sig_aux][0]))%(N+1)
                plt.plot(puntos[(i,sig),0],puntos[(i,sig),1])
                suma_ruta += np.floor(1000*fnorm(puntos[i,:]-puntos[sig,:]))
        plt.show()



        # Objective Function
        val_obj = 0
        for u in range(N+2):
            for v in range(N+2):
                if u != v:
                    for t in range(N+1):
                        val_obj += solution[f"x_{u}_{v}_{t}"]*dist[u,v] * lambda_obj

        print()
        print("The length of the path is",val_obj)
        lpm1 = val_obj

        etm1 = np.round((time()-stm1)/60,3) ## Elapsed time model 1
        print()
        print("Running time has been",etm1,"minutes.")


        ## We check that the restrictions are verified
        ## Restriction 1: We leave once from each city
        val_res1 = 0
        val1_tot = 0
        for u in range(N+1):
            lambda_1 = pen1*np.max(dist[u,])
            val1_tot += lambda_1
            for v1 in range(1,N+2):
                if u!=v1:
                    for t1 in range(N+1):
                        for v2 in range(1,N+2):
                            if u!=v2:
                                for t2 in range(N+1):
                                    val_res1 += solution[f"x_{u}_{v1}_{t1}"]*solution[f"x_{u}_{v2}_{t2}"] * lambda_1
                        val_res1 += solution[f"x_{u}_{v1}_{t1}"] * (-2*lambda_1)
        print("The value of constraint 1 is ",val_res1,"and it should be ",-val1_tot)
        ## Minimo $-lambda_1*(N+1)$

        ## Restriction 2: We arrive once in each city
        val_res2 = 0
        val2_tot = 0
        for v in range(1,N+2):
            lambda_2 = pen1*np.max(dist[:,v])
            val2_tot += lambda_2
            for u1 in range(N+1):
                if v!=u1:
                    for t1 in range(N+1):
                        for u2 in range(N+1):
                            if v!=u2:
                                for t2 in range(N+1):
                                    val_res2 += solution[f"x_{u1}_{v}_{t1}"]*solution[f"x_{u2}_{v}_{t2}"] * lambda_2
                        val_res2 += solution[f"x_{u1}_{v}_{t1}"] * (-2*lambda_2)
        print("The value of constraint 2 is ",val_res2," and it should be ",-val2_tot)
        ## Minimo -lambda_2*(N+1)

        ## Restriction 3: At each instant we can only be on one edge
        val_res3 = 0
        for t in range(N+1):
            for u1 in range(N+2):
                for v1 in range(N+2):
                    if u1 !=v1:
                        for u2 in range(N+2):
                            for v2 in range(N+2):
                                if u2!=v2:
                                    val_res3 += solution[f"x_{u1}_{v1}_{t}"]*solution[f"x_{u2}_{v2}_{t}"] * lambda_3
                        val_res3 += solution[f"x_{u1}_{v1}_{t}"] * (-2*lambda_3)
        print("The value of constraint 3 is ",val_res3,"and it should be ",-lambda_3*(N+1))
        ## Minimo -lambda_3*(N+1)

        ## Restriction 4: We avoid cycles
        val_res41 = 0
        val_res42 = 0
        ## variables a_{v,t} = \sum x_{v,w,t+1}
        for v in range(N+2):
            for t in range(N):
                Q[(f"a_{v}_{t}",)] += lambda_4
                for w in range(1,N+2):
                    if v!=w:
                        val_res41 = solution[f"a_{v}_{t}"]*solution[f"x_{v}_{w}_{t+1}"] * (-2*lambda_4)
                for w1 in range(1,N+2):
                    if w1 != v:
                        for w2 in range(1,N+2):
                            if v != w2:
                                val_res41 += solution[f"x_{v}_{w1}_{t+1}"]*solution[f"x_{v}_{w2}_{t+1}"] * lambda_4
        print("The value of the 4-1 constraint is ",val_res41)


        for t in range(N):
            for u in range(N+2):
                for v in range(N+2):
                    if u!=v:
                        val_res42 += solution[f"x_{u}_{v}_{t}"]*solution[f"a_{v}_{t}"] * (-lambda_4)
                        val_res42 += solution[f"x_{u}_{v}_{t}"] * (lambda_4)
        print("The value of the 4-2 constraint is ", val_res42)


        # Objective Function
        val_obj = 0
        for u in range(N+2):
            for v in range(N+2):
                if u != v:
                    for t in range(N+1):
                        val_obj += solution[f"x_{u}_{v}_{t}"]*dist[u,v] * lambda_obj
        print("The value of the objective function is ",val_obj)

        val_res_suma = val_res1+val_res2+val_res3
        val_res_cor = -val1_tot-val2_tot-lambda_3*(N+1)
        if np.round(val_res_suma,3) == np.round(val_res_cor,3):
            SOLUCIONES[jjj,0] = 0
            SOLUCIONES[jjj,1] = etm1
            SOLUCIONES[jjj,2] = lpm1
        else:
            SOLUCIONES[jjj,0] = 1
            SOLUCIONES[jjj,1] = etm1
            SOLUCIONES[jjj,2] = 0

        #np.savetxt("Modelo_General.txt",SOLUCIONES)
print("SOLUCIONES ", SOLUCIONES)
#print(len(dwave_dic))

# Model GPS
modelo = 0 ## Put 0 for SA, 1 for simulated QA, 2 for real QA
n_samples = 300 # number of times we run the system

stm2 = time()

R=3
lis_r = range(R)
lis_n = range(0,N+2)

pen = 2
pen1 = 4

# We create the variables of our model
coef = qubovert.QUBO()

## Variables x_{i,j,r,q}
for i in lis_n:
    for j in lis_n:
        if not i==j:
            for r in lis_r:
                coef.create_var(f"x_{i}_{j}_{r}")
nqm2 = len(lis_n)*len(lis_n)*len(lis_r)
nqm2 = (N+2)*(N+2)*3

## Restriction 1
## A single case for each r
lambda_1 = dist_vc*pen
for i in lis_n:
    for j in lis_n:
        if not i==j:
            for r1 in lis_r:
                for r2 in lis_r:
                    coef[(f"x_{i}_{j}_{r1}",f"x_{i}_{j}_{r2}")] += (lambda_1)
            for r in lis_r:     
                    coef[(f"x_{i}_{j}_{r}",)] += -2*(lambda_1)
 
## Restriction 2
## Must exit once from each node
for i in range(N+1):
    lambda_2 = pen1*np.max(dist[i,])
    for j in lis_n:
        if not i == j:
            coef[(f"x_{i}_{j}_{1}",)] += -2*lambda_2 
    for j1 in  lis_n:
        if j1 != i:
            for j2 in lis_n:
                if j2!=i:
                    coef[(f"x_{i}_{j1}_{1}",f"x_{i}_{j2}_{1}")]+= lambda_2

## Restriction 3
## Each node must be reached once
for j in range(1,N+2):
    lambda_3 = np.max(dist[:,j])*pen1
    for i in range(N+1):
        if i!=j:
            coef[(f"x_{i}_{j}_{1}",)] += -2*lambda_3
    for i1 in  lis_n:
        if i1!=j:
            for i2 in lis_n:
                if i2!=j:
                    coef[(f"x_{i1}_{j}_{1}",f"x_{i2}_{j}_{1}")]+= lambda_3 

## Restriction 7
## Correct order of nodes i,j
lambda_7 = dist_vc*pen
for i in lis_n:
    for j in lis_n:
        if i!=j:
            coef[(f"x_{i}_{j}_{2}",f"x_{j}_{i}_{2}")] += lambda_7 
            coef[(f"x_{i}_{j}_{2}",)] += -lambda_7 
            coef[(f"x_{j}_{i}_{2}",)] += -lambda_7 

## Restriction 8
## Order relationship between nodes i,j,k
lambda_8 = dist_vc*pen
for i in range(1,N+1):
    for j in range(1,N+1):
        for k in range(1,N+1):
            if i!=j and j!=k and i!=k:
                        coef[(f"x_{j}_{i}_{2}",f"x_{k}_{j}_{2}")] += lambda_8
                        coef[(f"x_{j}_{i}_{2}",f"x_{k}_{i}_{2}")] -= lambda_8
                        coef[(f"x_{k}_{j}_{2}",f"x_{k}_{i}_{2}")] -= lambda_8
                        coef[(f"x_{k}_{i}_{2}",)] +=  lambda_8

## Objective Function
lambda_obj = 1
for i in lis_n:
    for j in lis_n:
        if i!=j:
            coef[(f"x_{i}_{j}_{1}",)] += lambda_obj*dist[i,j]

dwave_dic = {}
for i in coef:
    if len(i) == 1:
        dwave_dic[(i[0],i[0])] = coef[i]
    else:
        dwave_dic[i] = coef[i]

for jjj in range(nsim):

    if modelo == 0:
        sampleset = qubovert.sim.anneal_qubo(dwave_dic, num_anneals=n_samples)
        solution = sampleset.best.state
        
    if modelo == 1:
        sampler = SimulatedAnnealingSampler()
        #sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(dwave_dic, num_reads = n_samples)
        solution = sampleset.first.sample
    
    if modelo == 2:
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(dwave_dic,num_reads = n_samples)
        solution = sampleset.first.sample

    print()
    print("The number of qubits is",nqm2)
    
    ## better energy
    #print(sampleset.first.energy)
    ## Matriz solucion
    mat_sol = np.zeros((N+2,N+2))
    for i in  range(N+2):
        for j in range(N+2):
            if i!=j:
                if solution[f"x_{i}_{j}_{1}"] == 1:
                    mat_sol[i,j] = 1
    #print(mat_sol)

## We paint the proposed path
    print()
    print("Solution path drawing.")
    plt.plot(puntos[:,0],puntos[:,1],'o')
    vaux = np.array(list(range(N+2)))
    suma_ruta = 0
    for i in range(N+1):
        sig_aux = mat_sol[i,:]==1
        if np.sum(sig_aux) > 0:
            sig = (int(vaux[sig_aux][0]))%(N+1)
            plt.plot(puntos[(i,sig),0],puntos[(i,sig),1])
            suma_ruta += np.floor(1000*fnorm(puntos[i,:]-puntos[sig,:]))
    plt.show()
    
    ## We calculate the distance traveled
    val_resobj = 0
    for i in lis_n:
        for j in lis_n:
            if i!=j:
                val_resobj += solution[f"x_{i}_{j}_{1}"]*lambda_obj*dist[i,j]
    print()
    print("The length of the solution path is",val_resobj)
    lpm2 = val_resobj

    etm2 = np.round((time()-stm2)/60,3) ## Elapsed time model 1
    print()
    print("Running time has been",etm2,"minutes.")
    
    best_energy = 0
    ## Restriction 1
    val_res1 = 0
    for i in lis_n:
        for j in lis_n:
            if i != j:
                for r1 in lis_r:
                    for r2 in lis_r:
                        val_res1 += lambda_1*solution[f"x_{i}_{j}_{r1}"]*solution[f"x_{i}_{j}_{r2}"]  
                for r in lis_r:     
                        val_res1 += -2*lambda_1*solution[f"x_{i}_{j}_{r}"] 

    print("Constraint 1 is contributing ",val_res1, " and should contribute ",-((N+2)**2-(N+2))*lambda_1)                  
    best_energy += val_res1
    val_cor1 = -((N+2)**2-(N+2))*lambda_1
    
    ## Restriction 2
    val_res2 = 0
    val2_tot = 0
    for i in range(N+1):
        lambda_2 = pen1*np.max(dist[i,])
        val2_tot += lambda_2
        for j in lis_n:
            if i != j:
                val_res2 += -2*lambda_2 *solution[f"x_{i}_{j}_{1}"] 
        for j1 in  lis_n:
            if i!=j1:
                for j2 in lis_n:
                    if i!=j2:
                        val_res2 += solution[f"x_{i}_{j1}_{1}"]*solution[f"x_{i}_{j2}_{1}"]*lambda_2 

    print("Constraint 2 is contributing ",val_res2, " and should contribute ",-val2_tot)                
    best_energy += val_res2                
    val_cor2 = -val2_tot
    
    ## Restriction 3
    val_res3 = 0
    val3_tot = 0
    for j in range(1,N+2):
        lambda_3 = pen1*np.max(dist[:,j])
        val3_tot += lambda_3
        for i in range(N+1):
            if i!=j:
                val_res3 += -2*lambda_3*solution[f"x_{i}_{j}_{1}"] 
        for i1 in  lis_n:
            if i1 !=j:
                for i2 in lis_n:
                    if i2 !=j:
                        val_res3 += solution[f"x_{i1}_{j}_{1}"]*solution[f"x_{i2}_{j}_{1}"]*lambda_3 

    print("Constraint 3 is contributing ",val_res3, " and should contribute ",-val3_tot) 
    val_cor3 = -val3_tot
    best_energy += val_res3
    
    ## Restriction 7
    val_res7 = 0
    for i in lis_n:
        for j in lis_n:
            if i!=j:
                val_res7 += solution[f"x_{i}_{j}_{2}"]*solution[f"x_{j}_{i}_{2}"] * lambda_7 
                val_res7 += solution[f"x_{i}_{j}_{2}"] * (-lambda_7) 
                val_res7 += solution[f"x_{j}_{i}_{2}"] * (-lambda_7) 
    print("Constraint 7 is contributing ",val_res7, " and should contribute ",(-(N+2)**2+N+2)*lambda_7)      
    val_cor7 = (-(N+2)**2+N+2)*lambda_7
    best_energy += val_res7
    
    ## Restriction 8
    val_res8 = 0
    for i in range(1,N+1):
        for j in range(1,N+1):
            for k in range(1,N+1):
                if i!=j and j!=k and i!=k:
                            val_res8 += solution[f"x_{j}_{i}_{2}"]*solution[f"x_{k}_{j}_{2}"] * lambda_8
                            val_res8 += -solution[f"x_{j}_{i}_{2}"]*solution[f"x_{k}_{i}_{2}"] * lambda_8
                            val_res8 += -solution[f"x_{k}_{j}_{2}"]*solution[f"x_{k}_{i}_{2}"] * lambda_8
                            val_res8 += solution[f"x_{k}_{i}_{2}"]*solution[f"x_{k}_{i}_{2}"] * lambda_8
    print("Constraint 8 is contributing ",val_res8, " and should contribute ",0)      
    val_cor8  = 0
    best_energy += val_res8
    
     ## Objective Function
    val_resobj = 0
    for i in lis_n:
        for j in lis_n:
            if i != j:
                val_resobj += solution[f"x_{i}_{j}_{1}"]*lambda_obj*dist[i,j]
    print("The objective function is contributing ",val_resobj )                  
    best_energy += val_resobj              

    #print("Our energy is ",best_energy)
    #print("The energy of sampleset is ",sampleset.first.energy)
    
    val_res_suma = val_res1+val_res2+val_res3+val_res7+val_res8
    val_res_cor = val_cor1+val_cor2+val_cor3+val_cor7+val_cor8
    if np.round(val_res_suma,3) == np.round(val_res_cor,3):
        SOLUCIONES[jjj,0] = 0
        SOLUCIONES[jjj,1] = etm2
        SOLUCIONES[jjj,2] = lpm2
    else:
        SOLUCIONES[jjj,0] = 1
        SOLUCIONES[jjj,1] = etm2
        SOLUCIONES[jjj,2] = 0
        
    np.savetxt("Modelo_GPS.txt", SOLUCIONES)
 
