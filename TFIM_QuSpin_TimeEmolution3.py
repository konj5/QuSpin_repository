from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
import TFIM_QuSpin_TimeEvolution as evo1 # time evolution for our specific problem
import TFIM_QuSpin_TimeEvolution2 as evo2 # time evolution for our specific problem

def sequential(N:int, times:int, h:float, J:float, JT:float):
    
    state = evo1.timeEvolution(N=N,hmax=h, J=J, a=0.0469,drive=evo1.exponentialDrive(-100,100),t0=-100,tmax=100)[-1]
    
    #Kako zdej predstavim to stanje v bazi TFIM2????
    
    for _ in range(times):
        pass