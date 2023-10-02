from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
import TFIM_QuSpin_TimeEvolution_DveVerigi as TwoChain

def bin_array(num:int, m:int) -> list:
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def convertWaveFunction(v2):
    #get state
    N = int(np.round(np.log2(len(v2))))
    assert(N % 2 == 0)

    state = np.array([0 for _ in range(N)])
     
    for i in range(len(v2)):
        state += np.abs(v2[i])**2 * (-2*bin_array(i,N)+1)

    #cut only 1. chain
    N = N//2
    state = state[0:N]


    #razvoj po bazi

    #?????????????????????????????????????


#convertWaveFunction([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

