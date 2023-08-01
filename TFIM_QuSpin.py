from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library

def bin_array(num:int, m:int) -> list:
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def exactDiag(N:int, h:float, J:float) -> tuple:

    h_field = [[-h,i] for i in range(N)] #Transverzalno polje

    #NO PERIODIC BORDER
    J_interaction = [[-J,i,i+1] for i in range(N-1)] #Isingova interakcija

    static_spin = [["zz", J_interaction], ["x", h_field]]
    dynamic_spin = []

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis,dtype=np.float64)

    E, eigvect = H.eigh()

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> float:
    
    M = 0
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i]**2 * np.sum(2 *bin_array(i,int(np.round(np.log2(len(ground_state))))) - 1))
    
    return M

Ms = []
hs = np.linspace(0,2,100)
for h in hs:
    Ms.append(magnetization(exactDiag(8,h,1)[1][:,0]))

plt.plot(hs,Ms)
plt.show()
    