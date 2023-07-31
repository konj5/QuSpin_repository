from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)



#Ena veriga TFIM v QuSpin 

N = 8 #število spinov v verigi
J = 1 

hs = np.linspace(0,2,100)
Ms = []

for h in hs:

    h_field = [[-h,i] for i in range(N)] #Transverzalno polje

    #NO PERIODIC BORDER
    J_interaction = [[-J,i,i+1] for i in range(N-1)] #Isingova interakcija

    #PERIODIC BORDER
    #J_interaction = [[-J,i,(i+1)%N] for i in range(N)]

    static_spin = [["zz", J_interaction], ["x", h_field]]
    dynamic_spin = []


    #Izberi "brez zblock" ali "zblock = 1"
    basis_spin = spin_basis_1d(N)
    #basis_spin = spin_basis_1d(N, zblock = 1)
                            
    H = hamiltonian(static_spin, dynamic_spin, basis=basis_spin,dtype=np.float64)

    E, vect = H.eigh()

    state = vect[:,1]

    M = 0
    for i in range(2**N):
        bstate_M = 0
        for spin in 2*bin_array(i,N)-1:
            bstate_M += state[i]**2 * spin / N

        M += abs(bstate_M)

    Ms.append(M)

plt.plot(hs,Ms)

###########################

Ms = []

for h in hs:

    h_field = [[-h,i] for i in range(N)] #Transverzalno polje

    #NO PERIODIC BORDER
    J_interaction = [[-J,i,i+1] for i in range(N-1)] #Isingova interakcija

    #PERIODIC BORDER
    #J_interaction = [[-J,i,(i+1)%N] for i in range(N)]

    static_spin = [["zz", J_interaction], ["x", h_field]]
    dynamic_spin = []


    #Izberi "brez zblock" ali "zblock = 1"
    basis_spin = spin_basis_1d(N)
    #basis_spin = spin_basis_1d(N, zblock = 1)
                            
    H = hamiltonian(static_spin, dynamic_spin, basis=basis_spin,dtype=np.float64)

    E, vect = H.eigh()

    state = vect[:,1]

    expected_value = np.zeros(N)
    for i in range(2**N):
        expected_value +=  state[i]**2 * (2*bin_array(i,N)-1)    #########################ZAKAJ TA PRISTOP NE DELUJE?????????????????

    M = np.sum(expected_value) / N

    Ms.append(M)

plt.plot(hs,Ms)




plt.show()



