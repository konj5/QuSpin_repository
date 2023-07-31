from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


#Dve verigi TFIM v QuSpin 


N = 4 #število spinov v posamezni verigi
J = 1
h = 0
JT = 1


#Veriga 1 [0 : N], veriga 2 [N:2N]
h_field = [[-h, i] for i in range(2*N)] #Transverzalno polje

J_interaction = [[-J,i,i+1] for i in range(2*N-1) if i != N-1]  #Isingova interakcija

JT_interaction =[[-JT,i,N+i] for i in range(N)] #sklopitev med verigama

static_spin = [["zz", J_interaction], ["x", h_field], ["zz", JT_interaction]]
dynamic_spin = []


#Izberi "brez zblock" ali "zblock = 1"
basis_spin = spin_basis_1d(2*N)
#basis_spin = spin_basis_1d(2*N, zblock = 1)

H = hamiltonian(static_spin, dynamic_spin, basis=basis_spin,dtype=np.float64)

E, vect = H.eigh()

#Get ground state
E0 = E[0]
ground_state = vect[:,0]

p = np.square(np.absolute(ground_state))

expected_value = np.zeros(2*N)
for i in range(len(p)):
    state = 2*bin_array(basis_spin[i], 2*N)-1
    expected_value += p[i] * state

print(expected_value[0:N])
print(expected_value[N:2*N])


