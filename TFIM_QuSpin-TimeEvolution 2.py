from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from scipy import optimize as optim #gradient descent library




def bin_array(num:int, m:int) -> list:
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def exactDiag(N:int, h:float, J:float, JT:float) -> tuple:

    #Veriga 1 [0 : N], veriga 2 [N:2N]
    h_field = [[-h, i] for i in range(2*N)] #Transverzalno polje

    J_interaction = [[-J,i,i+1] for i in range(2*N-1) if i != N-1]  #Isingova interakcija

    JT_interaction =[[-JT,i,N+i] for i in range(N)] #sklopitev med verigama

    static_spin = [["zz", J_interaction], ["x", h_field], ["zz", JT_interaction]]
    dynamic_spin = []
    spin_basis = spin_basis_1d(2*N)

    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis,dtype=np.float64)

    E, eigvect = H.eigsh(k=1,which = "SA")

    return (E, eigvect)

OsnovnaStanja = dict()

def getOsnovnoStanje(N):
    if N in OsnovnaStanja.keys():
        return OsnovnaStanja[N]
    else:
        _, eigv = exactDiag(N, h = 0, J = 1, JT = 0)

        OsnovnaStanja[N] = eigv[:,0]
        
        return eigv[:,0]
    





def timeEvolution(N:int, hmax:float, J:float, JT:float, a:float, b:float, driveH:callable, driveJT:callable, t0:float, tmax:float):
    #Veriga 1 [0 : N], veriga 2 [N:2N]
    h_field = [[-hmax, i] for i in range(2*N)] #Transverzalno polje

    J_interaction = [[-J,i,i+1] for i in range(2*N-1) if i != N-1]  #Isingova interakcija

    JT_interaction =[[-JT,i,N+i] for i in range(N)] #sklopitev med verigama

    static_spin = [["zz", J_interaction]]
    dynamic_spin = [["x", h_field, driveH,[a]], ["zz", JT_interaction, driveJT, [b]]]
    spin_basis = spin_basis_1d(2*N)
    
    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis,dtype=np.float64)

    return(H.evolve(v0=[0 if x != 0 else 1 for x in range(2**N)], t0 = 0, times=np.linspace(t0,tmax,100)))
    

def energyTimeEvolution(N:int, hmax:float, J:float, JT:float, a:float, b:float, driveH:callable, driveJT:callable, t0:float, tmax:float, startstate:list):
    #Veriga 1 [0 : N], veriga 2 [N:2N]
    h_field = [[-hmax, i] for i in range(2*N)] #Transverzalno polje

    J_interaction = [[-J,i,i+1] for i in range(2*N-1) if i != N-1]  #Isingova interakcija

    JT_interaction =[[-JT,i,N+i] for i in range(N)] #sklopitev med verigama

    static_spin = [["zz", J_interaction]]
    dynamic_spin = [["x", h_field, driveH,[a]], ["zz", JT_interaction, driveJT, [b]]]
    spin_basis = spin_basis_1d(2*N)
    
    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis,dtype=np.float64)

    #print(spin_basis[0])
    #print(bin_array(spin_basis[0], m = N))

    ts = np.linspace(t0,tmax,100)
    states = H.evolve(v0=startstate, t0 = t0, times=ts)
    Es = []

    for i in range(len(ts)):
        state = states[:,i]

        E = np.conjugate(state).dot(H(time=ts[i]).dot(state))
        assert(np.abs(np.imag(E)) < 0.0001)
        E = np.real(E)

        Es.append(E)

    #E = np.conjugate(endstate).dot(endstate)

    return Es

def energyComparison(N:int, hmax:float, J:float, JT:float, a:float, b:float, drive:object):
    
    #exact diagonalization
    (E0_exact, basestate) = exactDiag(N, hmax, J, JT)
    E0_exact = E0_exact[0]                                         




    #evolution

    basestate = getOsnovnoStanje(N)

    Es = energyTimeEvolution(N,hmax, J, JT, a, b, drive.driveH, drive.driveJT, drive.t0, drive.tend, basestate)
    E0_evolved = Es[-1]

    return (f"Exact ground state: {E0_exact}, Evolved: {E0_evolved}, difference: {np.abs(E0_exact-E0_evolved)}", [E0_exact, E0_evolved, np.abs(E0_exact-E0_evolved)])

class linearDrive:
    t0 = 0

    def __init__(self, tmax:float, tend:float, mode:str) -> None:
        self.tmax = tmax
        self.tend = tend
        self.mode = mode 
        #instructions for mode
        #simul - both at he same time
        #hJT - h first, JT second
        #JTh -JT first, h second

    def driveH(self, t:float, a:float) -> float:
        if self.mode == "JTh": 
            if t < 50: 
                return 0
            elif t < 100:
                return (t-50)/50

            return 1
        
        elif self.mode == "hJT": 
            if t < 50: return t/50

            return 1
        
        else:
            if t < 100: return t/100

            return 1

    
    def driveJT(self, t:float, b:float) -> float:
        if self.mode == "hJT": 
            if t < 50: 
                return 0
            elif t < 100:
                return (t-50)/50

            return 1
        
        elif self.mode == "JTh": 
            if t < 50: return t/50

            return 1
        
        else:
            if t < 100: return t/100

            return 1
    
class exponentialDrive:

    
    
    def __init__(self, t0:float, tend:float, mode:str) -> None:
        self.t0=t0
        self.tend=tend
        self.mode = mode

    def driveH(self, t:float, a:float) -> float:
        if self.mode == "JTh":
            tmid = (self.t0+self.tend)/2

            if t < tmid:
                return 0
            elif t > tmid and t < 0:
                return np.exp(a*t)
            else:
                return 1

        elif self.mode == "hJT":
            tmid = (self.t0+self.tend)/2

            if t < self.t0+10:
                return 0
            elif t < tmid:
                return np.exp(a*(t+tmid))
            else:
                return 1

        else:
            if t < self.t0 + 10:
                return 0
            elif t < 0:
                return np.exp(a*t)
            else:
                return 1
            
    def driveJT(self, t:float, b:float) -> float:
        if self.mode == "hJT":
            tmid = (self.t0+self.tend)/2

            if t < tmid:
                return 0
            elif t > tmid and t < 0:
                return np.exp(b*t)
            else:
                return 1

        elif self.mode == "JTh":
            tmid = (self.t0+self.tend)/2

            if t < self.t0+10:
                return 0
            elif t < tmid:
                return np.exp(b*(t+tmid))
            else:
                return 1

        else:
            if t < self.t0 + 10:
                return 0
            elif t < 0:
                return np.exp(b*t)
            else:
                return 1
        
class fermiDiracDrive:

    def __init__(self, t0:float, tend:float, mode:str) -> None:
        self.t0=t0
        self.tend=tend
        self.mode=mode

    def driveH(self, t:float, a:float) -> float:
        if self.mode == "JTh":
            tmid = (self.t0+self.tend)/2

            if t < tmid:
                return 0
            elif t > tmid and t < 0:
                return 1 / (1 + np.exp(-a*t))
            else:
                return 1

        elif self.mode == "hJT":
            tmid = (self.t0+self.tend)/2

            if t < self.t0+10:
                return 0
            elif t < tmid:
                return 1 / (1 + np.exp(-a*(t+tmid)))
            else:
                return 1

        else:
            if t < self.t0 + 10:
                return 0
            elif t < 0:
                return 1 / (1 + np.exp(-a*t))
            else:
                return 1
            
    def driveJT(self, t:float, b:float) -> float:
        if self.mode == "hJT":
            tmid = (self.t0+self.tend)/2

            if t < tmid:
                return 0
            elif t > tmid and t < 0:
                return 1 / (1 + np.exp(-b*t))
            else:
                return 1

        elif self.mode == "JTh":
            tmid = (self.t0+self.tend)/2

            if t < self.t0+10:
                return 0
            elif t < tmid:
                return 1 / (1 + np.exp(-b*(t+tmid)))
            else:
                return 1

        else:
            if t < self.t0 + 10:
                return 0
            elif t < 0:
                return 1 / (1 + np.exp(-b*t))
            else:
                return 1

"""
N = 4
resultLinear = energyComparison(N=N,hmax=1,J=1,JT=1,a=0, b=0,drive=linearDrive(100,200, mode="hJT"))
resultExponential = energyComparison(N=N,hmax=1,J=1,JT=1,a=0.0469, b=0.0469,drive=exponentialDrive(-100,100, mode="hJT"))
resultFD = energyComparison(N=N,hmax=1,J=1, JT=1,a=0.049, b=0.049,drive=exponentialDrive(-100,100, mode = "hJT"))

print(f"Linear -> {resultLinear[0]}")
print(f"Exponential -> {resultExponential[0]}")
print(f"Exponential -> {resultFD[0]}")
"""

def minimizeParametersAB():

    def f(arg):
        a = arg[0]
        b = arg[1]

        return(energyComparison(N=1,hmax=1,J=1,JT=1,a=a,b=b,drive=exponentialDrive(-100,100, mode="hJT"))[1][2])

    print(optim.minimize(fun=f, x0=[0.49,1]))

#minimizeParametersAB()

def varryParametersAB():  ##########################   KJE STA V RESNICI ZDAJ OPTIMALNA a in b

    def f(arg):
        a = arg[0]
        b = arg[1]

        return(energyComparison(N=2,hmax=1,J=1,JT=2,a=a,b=b,drive=exponentialDrive(-100,100, mode="simult"))[1][2])
    
    As = np.linspace(0.25,0.75,10)
    Bs = np.linspace(0,10,10)
    diffs = []
    xs = []
    ys = []
    for a in As:
       for b in Bs:
           xs.append(a)
           ys.append(b)
           diffs.append(f([a,b]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.plot_trisurf(xs, ys, diffs,cmap='viridis', edgecolor='none')
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    fig.add_axes(ax)
    plt.show()

    
varryParametersAB()

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


