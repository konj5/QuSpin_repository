from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from scipy import interpolate # polynomial interpolation library


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

    E, eigvect = H.eigsh(k=1,which = "SA")

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> float:
    
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i]**2 * np.sum(2 *bin_array(i,N) - 1)) / N
    
    return M

def linear(t:float, a:float) -> float:
    if t < 100: return t/100

    return 1


def timeEvolution(N:int, hmax:float, J:float, a:float, drive, t0:float, tmax:float):
    J_interaction = [[-J,i,i+1] for i in range(N-1)]
    static_spin = [["zz", J_interaction]]

    dynamic_list = [["x",[[-hmax,i] for i in range(N)],drive,[a]]]

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis,dtype=np.float64)

    #print(spin_basis[0])
    #print(bin_array(spin_basis[0], m = N))

    return(H.evolve(v0=[0 if x != 0 else 1 for x in range(2**N)], t0 = 0, times=np.linspace(t0,tmax,100)))
    

def energyTimeEvolution(N:int, hmax:float, J:float, a:float, drive:callable, t0:float, tmax:float):
    J_interaction = [[-J,i,i+1] for i in range(N-1)]
    static_spin = [["zz", J_interaction]]

    dynamic_list = [["x",[[-hmax,i] for i in range(N)],drive,[a]]]

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis,dtype=np.float64)

    #print(spin_basis[0])
    #print(bin_array(spin_basis[0], m = N))

    ts = np.linspace(t0,tmax,100)
    states = H.evolve(v0=[0 if x != 0 else 1 for x in range(2**N)], t0 = 0, times=ts)
    Es = []

    for i in range(len(ts)):
        state = states[:,i]

        E = np.conjugate(state).dot(H(time=ts[i]).dot(state))
        assert(np.abs(np.imag(E)) < 0.0001)
        E = np.real(E)

        Es.append(E)

    #E = np.conjugate(endstate).dot(endstate)

    return Es

def energyComparison(N:int, hmax:float, J:float, a:float, drive:callable, t0:float, tmax:float):
    
    #exact diagonalization
    (E0_exact, trash) = exactDiag(N, hmax, J)
    E0_exact = E0_exact[0]


    #evolution
    Es = energyTimeEvolution(N,hmax, J, a, drive, t0, tmax)
    E0_evolved = Es[-1]

    print(f"Exact ground state: {E0_exact}, Evolved: {E0_evolved}, difference: {np.abs(E0_exact-E0_evolved)}")

energyComparison(N=12,hmax=1,J=1,a=0,drive=linear,t0=0,tmax=200)





