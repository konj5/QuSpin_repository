import TFIM_QuSpin_TimeEvolution2 as OneChain
import TFIM_QuSpin_TimeEvolution_DveVerigi as TwoChain
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library


N = 4
J = 1

t0 = 0
tend = 10
hx0 = 10
hxend = 0
hx2 = 100

JT = 1
hxstart = 1

w = 2* (hx2 - hxstart)
k = 1/5

#GET DATA FOR ONE CHAIN
hxDrive = TwoChain.hxDrive(t0=t0, tend=tend, hx0=hx0, hxend=hxend, hx2=hx2, a = 10)
JTDrive = TwoChain.JTDrive(JT = JT, w=w, hxstart=hxstart, hxdrive=hxDrive)
#JTDrive = TwoChain.JTDriveAutoW(JT = JT, hxstart=hxstart, hxdrive=hxDrive)

#OneChain.evolvePQA_st(N=N, J = J, hx=hxend, a = [""], drives=[hxDrive])
#(ts1, ds1, dost1, exactdots1) = OneChain.evolvePQA_st_data(N=N, J = J, hx=hxend, a = "", drive=hxDrive)

#GET DATA FOR TWO CHAINS
basestate = OneChain.exactDiag(N = N, hx=hxend, J = J)
ts = np.linspace(t0,tend,1000)

vs2 = TwoChain.timeEvolution(N = N, J = J, hxdrive=hxDrive, JTdrive=JTDrive, k = k)

#Demonstracija Drive
y1s = [hxDrive.drive(t,"trash") for t in ts]
y2s = [JTDrive.drive(t,k) for t in ts]



plt.plot(ts, y1s, label = "$h_x$")
plt.plot(ts, y2s, label = "$J_T$")
plt.legend()
plt.show()


#Magnetizacija primerjava
M1s = [TwoChain.magnetization(vs2[:,i])[1] for i in range(len(vs2[0,:]))]
M2s = [TwoChain.magnetization(vs2[:,i])[2] for i in range(len(vs2[0,:]))]

vs1 = OneChain.timeEvolution(N = N, J = J, drive=hxDrive.drive, hx = hxend, a = "trash", t0 = t0, tmax=tend)

MOneChain = [OneChain.magnetization(vs1[:,i]) for i in range(len(vs1[0,:]))]

plt.plot(ts, M1s, label = "$M_1$")
plt.plot(ts, M2s, label = "$M_2$")
plt.plot(ts, MOneChain, label = "$M_{one chain}$")
plt.title("Magnetizacija")
plt.legend()
plt.show()




#Energija veriga 1 in 2 primerjava
J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
J_interaction2 = [[-J,N+i,N+i+1] for i in range(N-1)]

def getHamiltonianChain1(N:int, hx1:float, J:float):
    J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
    hx1_field = [[-hx1, i] for i in range(N)] #Transverzalno polje

    spin_basis = spin_basis_1d(2*N)
    return hamiltonian([["zz", J_interaction1], ["x", hx1_field]], [], basis=spin_basis,dtype=np.float64)

def getHamiltonianChain2(N:int, hx2:float, J:float):
    J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
    hx2_field = [[-hx2, N+i] for i in range(N)] #Transverzalno polje

    spin_basis = spin_basis_1d(2*N)
    return hamiltonian([["zz", J_interaction1], ["x", hx2_field]], [], basis=spin_basis,dtype=np.float64)


E1s = [np.real(np.conjugate(vs2[:,i]).dot(getHamiltonianChain1(N=N, J=J, hx1 = hxDrive.drive(ts[i],"trash")).dot(vs2[:,i]))) for i in range(len(vs2[0,:]))]
E2s = [np.real(np.conjugate(vs2[:,i]).dot(getHamiltonianChain2(N=N, J=J, hx2 = hx2).dot(vs2[:,i]))) for i in range(len(vs2[0,:]))]

vs1 = OneChain.timeEvolution(N = N, J = J, drive=hxDrive.drive, hx = hxend, a = "trash", t0 = t0, tmax=tend)

def getHamiltonianChainOneChain(N:int, hx:float, J:float):
    J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
    hx1_field = [[-hx, i] for i in range(N)] #Transverzalno polje

    spin_basis = spin_basis_1d(N)
    return hamiltonian([["zz", J_interaction1], ["x", hx1_field]], [], basis=spin_basis,dtype=np.float64)

Eonechains = [np.real(np.conjugate(vs1[:,i]).dot(getHamiltonianChainOneChain(N=N, J=J, hx = hxDrive.drive(ts[i],"trash")).dot(vs1[:,i]))) for i in range(len(vs1[0,:]))]

plt.plot(ts, E1s, label = "1. veriga")
plt.plot(ts, E2s, label = "2. veriga")
plt.plot(ts, Eonechains, label = "Enoverižni primer")
plt.title("Energija")
plt.legend()
plt.show()


#plt.plot(ts1, dost1)
#plt.plot(ts1,exactdots1, linestyle = "dashed")

#plt.show()






#ugotovitve

#-> če je hstart preveč stran od hend, je magnetizacija čisto uničena -> ?

"""
#Zanimive nastavitve
N = 4
J = 1

t0 = 0
tend = 10
hx0 = 10
hxend = 0
hx2 = 100

JT = 1
hxstart = 1

w = 2* (hx2 - hxstart)
k = 1/5
"""