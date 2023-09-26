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

JT = 1
hxstart = 1

w = 3


#GET DATA FOR ONE CHAIN
hxDrive = TwoChain.hxDrive(t0=t0, tend=tend, hx0=hx0, hxend=hxend, a = 10)
JTDrive = TwoChain.JTDrive(JT = JT, w=w, hxstart=hxstart, hxdrive=hxDrive)

OneChain.evolvePQA_st(N=N, J = J, hx=hxend, a = [""], drives=[hxDrive])
#(ts1, ds1, dost1, exactdots1) = OneChain.evolvePQA_st_data(N=N, J = J, hx=hxend, a = "", drive=hxDrive)

#GET DATA FOR TWO CHAINS
basestate = OneChain.exactDiag(N = N, hx=hxend, J = J)
ts = np.linspace(t0,tend,100)

vs = TwoChain.timeEvolution(N = N, J = J, hxdrive=hxDrive, JTdrive=JTDrive)


M1s = [TwoChain.magnetization(vs[:,i])[1] for i in range(len(vs[0,:]))]
M2s = [TwoChain.magnetization(vs[:,i])[2] for i in range(len(vs[0,:]))]

vs = OneChain.timeEvolution(N = N, J = J, drive=hxDrive.drive, hx = hxend, a = "trash", t0 = t0, tmax=tend)

MOneChain = [OneChain.magnetization(vs[:,i]) for i in range(len(vs[0,:]))]

plt.plot(ts, M1s, label = "$M_1$")
plt.plot(ts, M2s, label = "$M_2$")
plt.plot(ts, MOneChain, label = "$M_{one chain}$")
plt.legend()
plt.show()



#plt.plot(ts1, dost1)
#plt.plot(ts1,exactdots1, linestyle = "dashed")

#plt.show()