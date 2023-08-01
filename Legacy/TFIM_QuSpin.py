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

expected_values = []
hs = np.linspace(0,1,100)
M = [] #"Magnetizacija"
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

        

    #Osnovno stanje
    E0 = E[0]
    ground_state = vect[:,0]
    normalized_probability_distribution = np.square(np.abs(ground_state))

    expected_value = np.zeros(N)
    for i in range(len(normalized_probability_distribution)):

        state = (2*bin_array(basis_spin[i],N)-1)
        
        p = normalized_probability_distribution[i]

        expected_value += p * state                

    expected_values.append(expected_value)
    M.append(np.sum(expected_value) / np.size(expected_value))

with open("Pricakovane_vrednosti.txt", "w+") as f:
    for (i,x) in enumerate(expected_values):
        f.write(f"h = {hs[i]} , {x}\n")

plt.plot(hs,M)
plt.savefig(fname = "Magnetizacija")



