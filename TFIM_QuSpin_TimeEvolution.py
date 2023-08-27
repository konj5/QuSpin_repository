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
    states = H.evolve(v0=[0 if x != 0 else 1 for x in range(2**N)], t0 = t0, times=ts)
    Es = []

    for i in range(len(ts)):
        state = states[:,i]

        E = np.conjugate(state).dot(H(time=ts[i]).dot(state))
        assert(np.abs(np.imag(E)) < 0.0001)
        E = np.real(E)

        Es.append(E)

    #E = np.conjugate(endstate).dot(endstate)

    return Es

def energyComparison(N:int, hmax:float, J:float, a:float, drive:object):
    
    #exact diagonalization
    (E0_exact, trash) = exactDiag(N, hmax, J)
    E0_exact = E0_exact[0]


    #evolution
    Es = energyTimeEvolution(N,hmax, J, a, drive.drive, drive.t0, drive.tend)
    E0_evolved = Es[-1]

    return (f"Exact ground state: {E0_exact}, Evolved: {E0_evolved}, difference: {np.abs(E0_exact-E0_evolved)}", [E0_exact, E0_evolved, np.abs(E0_exact-E0_evolved)])

class linearDriveDeprecated:
    t0 = 0

    def __init__(self, tmax, tend) -> None:
        self.tmax = tmax
        self.tend = tend

    def drive(self, t:float, a:float) -> float:
        if t < 100: return t/100

        return 1
    
class linearDrive:

    def __init__(self, t0:float, tstart:float, tmax: float, tend:float) -> None:
        assert(t0<=tstart)
        assert(tstart<=tmax)
        assert(tmax<=tend)
        
        self.tmax = tmax
        self.tend = tend
        self.t0 = t0
        self.tstart = tstart

    def drive(self, t:float, a:float) -> float:
        if t < self.tstart:
            return 0
        if t < self.tmax:
            return (t-self.tstart) / (self.tmax - self.tstart)
        
        return 1
        
    
class exponentialDriveDeprecated:

    #In this setup optimal a appears to be ~0.0469
    
    def __init__(self, t0:float, tend:float) -> None:
        self.t0=t0
        self.tend=tend

    def drive(self, t:float, a:float) -> float:
        if t < self.t0 + 10:
            return 0
        elif t < 0:
            return np.exp(a*t)
        else:
            return 1
        
class exponentialDrive:
    
    def __init__(self, t0:float, tstart:float, tmax: float, tend:float) -> None:
        assert(t0<=tstart)
        assert(tstart<=tmax)
        assert(tmax<=tend)
        
        self.tmax = tmax
        self.tend = tend
        self.t0 = t0
        self.tstart = tstart

    def drive(self, t:float, a:float) -> float:
        if t < self.tstart:
            return 0
        
        if t < self.tmax:
            return np.exp(a * (t -self.tstart - self.tmax) / (self.tmax - self.tstart))
        
        return 1
        
class fermiDiracDriveDeprecated:

    #optimum appears to be a ~ 0.049

    def __init__(self, t0:float, tend:float) -> None:
        self.t0=t0
        self.tend=tend

    def drive(self, t:float, a:float) -> float:
        if t < self.t0 + 10:
            return 0
        elif t < self.tend - 10:
            return 1 / (1 + np.exp(-a*t))
        else:
            return 1
        
class fermiDiracDrive:

    def __init__(self, t0:float, tstart:float, tmax: float, tend:float) -> None:
        assert(t0<=tstart)
        assert(tstart<=tmax)
        assert(tmax<=tend)
        
        self.tmax = tmax
        self.tend = tend
        self.t0 = t0
        self.tstart = tstart

    def drive(self, t:float, a:float) -> float:
        if t < self.tstart:
            return 0
        
        if t < self.tmax:
            return 1 / (1 + np.exp(-a * (t - (self.tstart + self.tmax)/2)))
        
        return 1

"""
N = 8
resultLinear = energyComparison(N=N,hmax=1,J=1,a=0,drive=linearDriveDeprecated(100,200))
resultExponential = energyComparison(N=N,hmax=1,J=1,a=0.0469,drive=exponentialDriveDeprecated(-100,100))
resultFD = energyComparison(N=N,hmax=1,J=1,a=0.049,drive=fermiDiracDriveDeprecated(-100,100))

print(f"Linear -> {resultLinear[0]}")
print(f"Exponential -> {resultExponential[0]}")
print(f"FermiDirac -> {resultFD[0]}")
"""

def varyParameterA():
    diff = []
    As = np.linspace(0.102,0.106,100)
    for a in As:
        diff.append(energyComparison(N=12,hmax=1,J=1,a=a,drive=fermiDiracDrive(-100,100))[1][2])
        print(a)
    
    plt.plot(As, diff)
    plt.scatter(As, diff)
    plt.show()

#varyParameterA()

def varyParameterN():
    diffLin = []
    diffExp = []
    diffFD = []
    ns = [i for i in range(1,16,1)]
    for N in ns:
        diffLin.append(energyComparison(N=12,hmax=1,J=1,a=0,drive=linearDrive(100,200))[1][2])
        diffExp.append(energyComparison(N=N,hmax=1,J=1,a=0.0469,drive=exponentialDrive(-100,100))[1][2])
        diffFD.append(energyComparison(N=N,hmax=1,J=1,a=0.049,drive=fermiDiracDrive(-100,100))[1][2])
        print(N)
    
    plt.plot(ns, diffLin, label = "linear")
    plt.scatter(ns, diffLin)

    plt.plot(ns, diffExp, label = "exponential")
    plt.scatter(ns, diffExp)

    plt.plot(ns, diffFD, label = "fermi-dirac")
    plt.scatter(ns, diffFD)
    
    plt.legend()
    plt.show()

#varyParameterN()




def evolveEnergy(N:int, J:float, hmax:float, a:list, drives:list):
    #exact diagonalization
    (E0_exact, trash) = exactDiag(N, hmax, J)
    E0_exact = E0_exact[0]


    #evolution
    Ess = []
    dss = []
    for i in range(len(drives)):
        Es = energyTimeEvolution(N,hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)
        ts = np.linspace(drives[i].t0, drives[i].tend, 100)
        
        ds = []
        for t in ts:
            ds.append(drives[i].drive(t,a[i]))
            
        Ess.append(Es)
        dss.append(ds)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    for i in range(len(Ess)):
        ax1.plot(ts, dss[i])
        ax2.plot(ts, Ess[i])
        ax3.plot(ts[-10:-1], Ess[i][-10:-1])
        
    ax1.set_title("Gonilna funkcija")
    ax2.set_title("E(t), črtkana črta je točna vrednost")
    ax3.set_title("Končne energije")
        
    ax2.axhline(y = E0_exact, linestyle = "dashed")
    
    
    plt.show()
    
    
evolveEnergy(N=8,J=1,hmax=1,a=[0,4,0.13],drives=[linearDrive(0,1,100,110), exponentialDrive(0,1,100,110), fermiDiracDrive(0,1,100,110)])
#evolveEnergy(N=14,J=1,hmax=1,a=[0,4,0.01],drives=[linearDrive(0,1,1000,1010), exponentialDrive(0,1,1000,1010), fermiDiracDrive(0,1,1000,1010)])
    