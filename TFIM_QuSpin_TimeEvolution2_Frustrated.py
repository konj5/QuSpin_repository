from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

hz = 0.01

def bin_array(num:int, m:int) -> list:
    """
    Pretvori pozitivno celo število `num` v m-bitni bitni vektor.

    Args:
        num (int): Pozitivno celo število za pretvorbo.
        m (int): Število bitov v bitnem vektorju.

    Returns:
        list: M-bitni bitni vektor v obliki seznama.
    """
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def exactDiag(hx:float, J:float) -> tuple:
    """
    Izračuna točno rešitev kvantnega modela s pomočjo diagonalizacije Hamiltoniana.

    Args:
        hx (float): Moč transverzalnega magnetnega polja.
        J (float): Moč Isingove interakcije.

    Returns:
        tuple: Vrne nizkoenergijske lastne vrednosti in ustrezen lastni vektor Hamiltoniana.
    """

    h_field = [[-hx,i] for i in range(8)] # Transverzalno polje

    # Brez periodičnih robov
    J_interaction = [[-J,0,3],[-J,0,5],[-J,3,1],[-J,3,6],[J,3,4],[-J,1,4],[-J,6,4],[-J,4,2],[-J,4,7],[-J,2,7]] # Isingova interakcija

    static_spin = [["zz", J_interaction], ["z",[[hz,i] for i in range(8)]], ["x", h_field]]
    dynamic_spin = []

    spin_basis = spin_basis_1d(8)

    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis, dtype=np.float64)

    E, eigvect = H.eigsh(k=2, which="SA")

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> float:
    """
    Izračuna magnetizacijo osnovnega stanja.

    Args:
        ground_state (np.ndarray): Osnovno stanje sistema.

    Returns:
        float: Magnetizacija osnovnega stanja.
    """
    
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i])**2 * np.sum(2 * bin_array(i,N) - 1) / N
    
    return M

def anti_magnetization(ground_state:np.ndarray) -> float:
    """
    Izračuna antiparalelno magnetizacijo osnovnega stanja.

    Args:
        ground_state (np.ndarray): Osnovno stanje sistema.

    Returns:
        float: Antiparalelna magnetizacija osnovnega stanja.
    """
    
    AM = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        AMi = 0
        statei = 2 * bin_array(i,N) - 1
        for j in range(N):
            AMi += statei[j] * (-1)**j
        
        AM += (np.abs(ground_state[i]**2)) * np.abs(AMi)/N
    
    return AM

def timeEvolution(hx:float, J:float, a:float, drive:callable, t0:float, tmax:float):
    """
    Izvaja časovni razvoj sistema s pomočjo Hamiltoniana z dinamičnim členom.

    Args:
        hx (float): Moč transverzalnega magnetnega polja.
        J (float): Moč Isingove interakcije.
        a (float): Parameter dinamičnega člena.
        drive (callable): Funkcija za izračun dinamičnega člena v odvisnosti od časa.
        t0 (float): Začetni čas časovnega razvoja.
        tmax (float): Končni čas časovnega razvoja.

    Returns:
        np.ndarray: Seznam stanj sistema v odvisnosti od časa.
    """
    J_interaction = [[-J,0,3],[-J,0,5],[-J,3,1],[-J,3,6],[J,3,4],[-J,1,4],[-J,6,4],[-J,4,2],[-J,4,7],[-J,2,7]]
    static_spin = [["zz", J_interaction], ["z",[[hz,i] for i in range(8)]], ["x",[[-hx,i] for i in range(8)]]]

    dynamic_list = [["x",[[-1,i] for i in range(8)],drive,[a]]]

    spin_basis = spin_basis_1d(8)

    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis, dtype=np.float64)
    
    groundstate = exactDiag(hx + drive(garbage=a,t=t0),J)[1][:,0]

    return H.evolve(v0=groundstate, t0=t0, times=np.linspace(t0,tmax,1000))

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
    

E0_exactdict = dict()

def getE0_exact(data):
    """
    Izračuna energijo osnovnega stanja in pripadajoči lastni vektor za podane parametre, pri čemer optimizira ponovno izračunavanje.

    Args:
        data: Tuple s parametri za izračun (hx, J).

    Returns:
        tuple: Energijska lastna vrednost in pripadajoči lastni vektor Hamiltoniana.
    """
    data = tuple(data)

    if data in E0_exactdict.keys():
        return E0_exactdict[data]

    (E0_exact, basestate) = exactDiag(data[0], data[1])
    E0_exactdict[data] = (E0_exact[0],basestate[:,0])
    return (E0_exact[0],basestate[:,0])

def evolvePQA_st(J:float, hx:float, a:list, drives:list):
    """
    Izvaja časovni razvoj P in P_st in prikazuje rezultate.

    Args:
        J (float): Moč Isingove interakcije.
        hx (float): Moč transverzalnega magnetnega polja.
        a (list): Seznam parametrov za dinamični člen.
        drives (list): Seznam objektov za generiranje dinamičnega člena.

    Returns:
        None
    """
    # Točna diagonalizacija
    basestate = getE0_exact((hx,J))[1]

    # Izvaja časovni razvoj za vsak dinamični člen
    exactdotss = []
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        # Začetna nastavitev a za Fermi_Dirac
        drives[i].drive(0,a[i])

        print(i)
        vs = timeEvolution(hx, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)

        ts = np.linspace(drives[i].t0, drives[i].tend, 1000)

        exactstates = np.zeros_like(vs)
        for j in range(len(ts)):
            print(ts[j])
            exactstates[:,j] = exactDiag(J=J, hx = hx + drives[i].drive(ts[j],a[i]))[1][:,0]

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
        ax2.plot(tss[i], exactdotss[i], color = colors[i+1], linestyle = "dotted")
        ax3.axhline(y = dotss[i][-1], label = f"{a[i]}", color = colors[i])
        ax3.axhline(y = exactdotss[i][-1], label = f"{a[i]}", color = colors[i+1], linestyle = "dashed")

    ax1.set_title("Gonilna funkcija")
    ax2.set_title("P (polna črta), P_st (črtkana) - Frustrirani model")
    ax3.set_title("Končne vrednosti")

    ax3.axhline(y = 1, linestyle = "dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe))/2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe))/2))

    plt.show()

def getK2toFitTmax(tstart:float, tmax:float):
    """
    Izračuna vrednost parametra K2 za prileganje končnemu času.

    Args:
        tstart (float): Začetni čas časovnega razvoja.
        tmax (float): Končni čas časovnega razvoja.

    Returns:
        float: Vrednost parametra K2.
    """
    A = np.log(1/10**-8 - 1)
    B = np.log(1/(1-10**-8) - 1)
    
    k1 = (tstart * B - tmax * A) / (B - A)
    k2 = - A / (tstart - k1)

    return(k2)

evolvePQA_st(J=1, hx=0, a = [""], drives=[HarmonicDrive(t0 = 0.1, tend=1000)])
evolvePQA_st(J=1, hx=0, a = [""], drives=[RootHarmonicDrive(t0 = 0.1, tend=1000)])
evolvePQA_st(J=1, hx=0, a = [""], drives=[LogHarmonicDrive(t0 = 0.1, tend=1000)])

#Kako izbirati a in tend-t0 ==> zih neka povezava
# tend določi a, tako da je hx(tend) = hx(inf) + endtolerance
#za harm in rootharm je dobro 10**-1, za logharm pa 10**-3 pri tmax = 1000
