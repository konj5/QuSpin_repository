from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
from scipy import interpolate # polynomial interpolation library


GreatCounter_TheMagnificscent = 0

hz = 0.01


def bin_array(num:int, m:int) -> list:
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def exactDiag(N:int, hx:float, J:float) -> tuple:

    h_field = [[-hx,i] for i in range(N)] #Transverzalno polje

    #NO PERIODIC BORDER
    J_interaction = [[-J,i,i+1] for i in range(N-1)] #Isingova interakcija

    static_spin = [["zz", J_interaction], ["z",[[hz,i] for i in range(N)]], ["x", h_field]]
    dynamic_spin = []

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis,dtype=np.float64)

    E, eigvect = H.eigsh(k=2,which = "SA")

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> float:
    
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i])**2 * np.sum(2 *bin_array(i,N) - 1) / N
    
    return M

def anti_magnetization(ground_state:np.ndarray) -> float:
    
    AM = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        AMi = 0
        statei = 2 *bin_array(i,N) - 1
        for j in range(N):
            AMi += statei[j] * (-1)**j
        
        AM += (np.abs(ground_state[i]**2)) * np.abs(AMi)/N
    
    return AM


def timeEvolution(N:int, hx:float, J:float, a:float, drive:callable, t0:float, tmax:float):
    J_interaction = [[-J,i,i+1] for i in range(N-1)]
    static_spin = [["zz", J_interaction], ["z",[[hz,i] for i in range(N)]], ["x",[[-hx,i] for i in range(N)]]]

    dynamic_list = [["x",[[-1,i] for i in range(N)],drive,[a]]]

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis,dtype=np.float64)

    #print(spin_basis[0])
    #print(bin_array(spin_basis[0], m = N))
    
    groundstate = exactDiag(N,hx + drive(garbage=a,t=t0),J)[1][:,0]

    #groundstate = [0 if i not in (0,2**N-1) else 1/np.sqrt(2) for i in range(2**N)]

    return(H.evolve(v0=groundstate, t0 = t0, times=np.linspace(t0,tmax,1000)))
    

endtolerance = 10**-1

class HarmonicDrive:

    def __init__(self, t0:float,tend:float) -> None:
        self.tmax = tend
        self.tend = tend
        self.t0 = t0
        self.tstart = t0
        self.c = tend * (endtolerance)


    def drive(self, t:float, garbage:float) -> float:
        if t == 0:
            t = 10**-3


        return self.c / t
    
class RootHarmonicDrive:

    def __init__(self, t0:float, tend:float) -> None:
        self.tmax = tend
        self.tend = tend
        self.t0 = t0
        self.tstart = t0
        self.c = np.sqrt(tend) * (endtolerance)

    def drive(self, t:float, garbage:float) -> float:
        if t == 0:
            t = 10**-3

        return self.c / np.sqrt(t)
    
class LogHarmonicDrive:

    def __init__(self, t0:float, tend:float) -> None:
        self.tmax = tend
        self.tend = tend
        self.t0 = t0
        self.tstart = t0
        self.c = np.log(tend+1) * (endtolerance)

    def drive(self, t:float, garbage:float) -> float:
        if t == 0:
            t = 10**-3

        return self.c / np.log(t+1)
    

E0_exactdict=dict()
def getE0_exact(data):
        data = tuple(data)
        	
        if data in E0_exactdict.keys():
            return E0_exactdict[data]

        (E0_exact, basestate) = exactDiag(data[0], data[1], data[2])
        E0_exactdict[data] = (E0_exact[0],basestate[:,0])
        return (E0_exact[0],basestate[:,0])



def evolvePQA_st(N:int, J:float, hx:float, a:list, drives:list):
    #exact diagonalization
    basestate = getE0_exact((N,hx,J))[1]


    #evolution
    exactdotss = []
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0,a[i])
        
        print(i)
        vs = timeEvolution(N,hx, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)


        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        exactstates = np.zeros_like(vs)
        for j in range(len(ts)):
            print(ts[j])
            exactstates[:,j] = exactDiag(N=N,J=J, hx = hx + drives[i].drive(ts[j],a[i]))[1][:,0]
        
        
        dots = []
        exactdots = []
        for j in range(len(vs[0,:])):
            dots.append(np.abs(vs[:,j].dot(basestate))**2)
            exactdots.append(np.abs(exactstates[:,j].dot(basestate))**2)
        
        
        ds = []
        for t in ts:
            ds.append(drives[i].drive(t,a[i]))
            
        dotss.append(dots)
        exactdotss.append(exactdots)
        dss.append(ds)
        tss.append(ts)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    colors = ["red", "blue", "orange", "green", "purple", "pink"]
    
    tempe = []
    for i in range(len(dotss)):
        tempe.append(dotss[i][-1])
        tempe.append(exactdotss[i][-1])

    for i in range(len(dotss)):
        ax1.plot(tss[i], dss[i], color = colors[i])
        ax2.plot(tss[i], dotss[i], color = colors[i])
        ax2.plot(tss[i], exactdotss[i], color = colors[i], linestyle = "dashed")
        ax3.axhline(y = dotss[i][-1], label = f"{a[i]}", color = colors[i])
        ax3.axhline(y = exactdotss[i][-1], label = f"{a[i]}", color = colors[i], linestyle = "dashed")
        #ax3.legend()

    ax1.set_title("Gonilna funkcija")
    ax2.set_title("P (polna črta), P_st (črtkana)")
    ax3.set_title("Končne vrednosti")
        
    #ax2.axhline(y = 1, linestyle = "dashed")
    ax3.axhline(y = 1, linestyle = "dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe))/2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe))/2))

    plt.show()


def evolvePQA_st_data(N:int, J:float, hx:float, a:float, drive:object):
    #exact diagonalization
    basestate = getE0_exact((N,hx,J))[1]


    #evolution
    vs = timeEvolution(N,hx, J, a, drive.drive, drive.t0, drive.tend)


    ts = np.linspace(drive.t0, drive.tend, 100)

    exactstates = np.zeros_like(vs)
    for j in range(len(ts)):
        print(ts[j])
        exactstates[:,j] = exactDiag(N=N,J=J, hx = hx + drive.drive(ts[j],a))[1][:,0]
    
    
    dots = []
    exactdots = []
    for j in range(len(vs[0,:])):
        dots.append(np.abs(vs[:,j].dot(basestate))**2)
        exactdots.append(np.abs(exactstates[:,j].dot(basestate))**2)
    
    
    ds = []
    for t in ts:
        ds.append(drive.drive(t,a))
        
    
    return (ts, ds, dots, exactdots)

def getK2toFitTmax(tstart:float, tmax:float):
    A = np.log(1/10**-8 - 1)
    B = np.log(1/(1-10**-8) - 1)
    
    k1 = (tstart * B - tmax * A) / (B - A)
    k2 = - A / (tstart - k1)

    return(k2)

#evolvePQA_st(N = 4, J=1, hx=0, a = [""], drives=[HarmonicDrive(t0 = 0.1, tend=1000)])
#evolvePQA_st(N = 4, J=1, hx=0, a = [""], drives=[RootHarmonicDrive(t0 = 0.1, tend=1000)])
#evolvePQA_st(N = 4, J=1, hx=0, a = [""], drives=[LogHarmonicDrive(t0 = 0.1, tend=1000)])

#Kako izbirati a in tend-t0 ==> zih neka povezava
# tend določi a, tako da je hx(tend) = hx(inf) + endtolerance
#za harm in rootharm je dobro 10**-1, za logharm pa 10**-3 pri tmax = 1000
