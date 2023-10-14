from quspin.operators import hamiltonian  # Hamiltoniani in operatorji
from quspin.basis import spin_basis_1d  # Hilbertov prostor za spin
import numpy as np  # splošne matematične funkcije
import matplotlib.pyplot as plt  # knjižnica za risanje grafov
from scipy import interpolate  # knjižnica za polinomsko interpolacijo

hz = 0.01

def bin_array(num: int, m: int) -> list:
    """
    Pretvori pozitivno celo število `num` v m-bitni bitni vektor.

    Parameters:
        num (int): Pozitivno celo število, ki ga želite pretvoriti.
        m (int): Število bitov v bitnem vektorju.

    Returns:
        list: m-bitni bitni vektor.
    """
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

def exactDiag(N: int, hx: float, J: float) -> tuple:
    """
    Izračuna natančno diagonizacijo za kvantni model Isinga.

    Izračuna lastne vrednosti in lastne vektorje za kvantni model Isinga z natančno diagonizacijo.

    Parameters:
        N (int): Število delcev.
        hx (float): Moč zunanjega magnetnega polja.
        J (float): Moč medsebojnega delovanja.

    Returns:
        tuple: Tuple z lastnimi vrednostmi in lastnimi vektorji.
    """
    h_field = [[-hx, i] for i in range(N)]  # Transverzalno polje

    # BREZ PERIODIČNEGA ROBA
    J_interaction = [[-J, i, i + 1] for i in range(N - 1)]  # Isingova interakcija

    static_spin = [["zz", J_interaction], ["z", [[hz, i] for i in range(N)]], ["x", h_field]]
    dynamic_spin = []

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis, dtype=np.float64)

    E, eigvect = H.eigsh(k=2, which="SA")

    return (E, eigvect)

def magnetization(ground_state: np.ndarray) -> float:
    """
    Izračuna magnetizacijo kvantnega stanja.

    Izračuna magnetizacijo kvantnega stanja na osnovi lastnega stanja.

    Parameters:
        ground_state (np.ndarray): Lastno stanje.

    Returns:
        float: Magnetizacija.
    """
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i]) ** 2 * np.sum(2 * bin_array(i, N) - 1) / N

    return M

def anti_magnetization(ground_state: np.ndarray) -> float:
    """
    Izračuna anti-magnetizacijo kvantnega stanja.

    Izračuna anti-magnetizacijo kvantnega stanja na osnovi lastnega stanja.

    Parameters:
        ground_state (np.ndarray): Lastno stanje.

    Returns:
        float: Anti-magnetizacija.
    """
    AM = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        AMi = 0
        statei = 2 * bin_array(i, N) - 1
        for j in range(N):
            AMi += statei[j] * (-1) ** j

        AM += (np.abs(ground_state[i] ** 2)) * np.abs(AMi) / N

    return AM

def timeEvolution(N: int, hx: float, J: float, a: float, drive: callable, t0: float, tmax: float):
    """
    Izvaja časovno evolucijo kvantnega stanja.

    Izračuna časovno evolucijo kvantnega stanja za dani kvantni model Isinga.

    Parameters:
        N (int): Število delcev.
        hx (float): Moč zunanjega magnetnega polja.
        J (float): Moč medsebojnega delovanja.
        a (float): Parameter `a` za pogon.
        drive (callable): Funkcija za pogon.
        t0 (float): Začetni čas.
        tmax (float): Maksimalni čas.

    Returns:
        np.ndarray: Rezultat časovne evolucije.
    """
    J_interaction = [[-J, i, i + 1] for i in range(N - 1)]
    static_spin = [["zz", J_interaction], ["z", [[hz, i] for i in range(N)]], ["x", [[-hx, i] for i in range(N)]]]

    dynamic_list = [["x", [[-1, i] for i in range(N)], drive, [a]]]

    spin_basis = spin_basis_1d(N)

    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis, dtype=np.float64)

    groundstate = exactDiag(N, hx + drive(garbage=a, t=t0), J)[1][:, 0]

    return (H.evolve(v0=groundstate, t0=t0, times=np.linspace(t0, tmax*1.1, 1000)))

    

class HarmonicDrive:

    def __init__(self, t0:float,tend:float) -> None:
        self.tmax = tend
        self.tend = tend
        self.t0 = t0
        self.tstart = t0
        endtolerance = 10**-1
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
        endtolerance = 10**-1
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
        endtolerance = 10**-1
        self.c = np.log(tend+1) * (endtolerance)

    def drive(self, t:float, garbage:float) -> float:
        if t == 0:
            t = 10**-3

        return self.c / np.log(t+1)
    

E0_exactdict = dict()

def getE0_exact(data):
    """
    Vrne natančno energijo in osnovno stanje za podatke.

    Parameters:
        data: Tuple podatkov (N, hx, J), kjer je N število delcev, hx moč zunanjega magnetnega polja, J moč medsebojnega delovanja.

    Returns:
        tuple: Tuple z energijo in osnovnim stanjem.
    """
    data = tuple(data)

    if data in E0_exactdict.keys():
        return E0_exactdict[data]

    (E0_exact, basestate) = exactDiag(data[0], data[1], data[2])
    E0_exactdict[data] = (E0_exact[0], basestate[:, 0])
    return (E0_exact[0], basestate[:, 0])



def evolvePQA_st(N: int, J: float, hx: float, a: list, drives: list):
    """
    Izvaja časovno evolucijo kvantnega stanja s primerjavo s točno rešitvijo.

    Izračuna časovno evolucijo kvantnega stanja za dani kvantni model Isinga, pri čemer primerja rezultate s točno rešitvijo.

    Parameters:
        N (int): Število delcev.
        J (float): Moč medsebojnega delovanja.
        hx (float): Moč zunanjega magnetnega polja.
        a (list): Seznam parametrov a za pogon.
        drives (list): Seznam različnih tipov pogonov.

    Returns:
        None
    """
    # Natančna diagonalizacija
    basestate = getE0_exact((N, hx, J))[1]

    # Evolucija
    exactdotss = []
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        # Nastavi začetno stanje a za Fermi-Dirac
        drives[i].drive(0, a[i])

        print(i)
        vs = timeEvolution(N, hx, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)

        ts = np.linspace(drives[i].t0, drives[i].tend, 1000)

        exactstates = np.zeros_like(vs)
        for j in range(len(ts)):
            print(ts[j])
            exactstates[:, j] = exactDiag(N=N, J=J, hx=hx + drives[i].drive(ts[j], a[i]))[1][:, 0]

        dots = []
        exactdots = []
        for j in range(len(vs[0, :])):
            dots.append(np.abs(vs[:, j].dot(basestate))**2)
            exactdots.append(np.abs(exactstates[:, j].dot(basestate))**2)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

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
        ax1.plot(tss[i], dss[i], color=colors[i])
        ax1.set_yscale("log")
        ax2.plot(tss[i], dotss[i], color=colors[i])
        ax2.plot(tss[i], exactdotss[i], color=colors[i+1], linestyle="dotted")
        ax3.axhline(y=dotss[i][-1], label=f"{a[i]}", color=colors[i])
        ax3.axhline(y=exactdotss[i][-1], label=f"{a[i]}", color=colors[i+1], linestyle="dashed")

    ax1.set_title("Gonilna funkcija")
    ax2.set_title("P (polna črta), P_st (črtkana)")
    ax3.set_title("Končne vrednosti")

    ax3.axhline(y=1, linestyle="dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe))/2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe))/2))

    plt.show()


def evolvePQA_st_data(N: int, J: float, hx: float, a: float, drive: object):
    """
    Izvaja časovno evolucijo kvantnega stanja s primerjavo s točno rešitvijo.

    Izračuna časovno evolucijo kvantnega stanja za dani kvantni model Isinga, pri čemer primerja rezultate s točno rešitvijo.

    Parameters:
        N (int): Število delcev.
        J (float): Moč medsebojnega delovanja.
        hx (float): Moč zunanjega magnetnega polja.
        a (float): Parameter a za pogon.
        drive (object): Tip pogona.

    Returns:
        tuple: Tuple s časi, vrednostmi pogona, P in P_st.
    """
    # Natančna diagonalizacija
    basestate = getE0_exact((N, hx, J))[1]

    # Evolucija
    vs = timeEvolution(N, hx, J, a, drive.drive, drive.t0, drive.tend)

    ts = np.linspace(drive.t0, drive.tend, 1000)

    exactstates = np.zeros_like(vs)
    for j in range(len(ts)):
        exactstates[:, j] = exactDiag(N=N, J=J, hx=hx + drive.drive(ts[j], a))[1][:, 0]

    dots = []
    exactdots = []
    for j in range(len(vs[0, :])):
        dots.append(np.abs(vs[:, j].dot(basestate))**2)
        exactdots.append(np.abs(exactstates[:, j].dot(basestate))**2)

    ds = []
    for t in ts:
        ds.append(drive.drive(t, a))

    return (ts, ds, dots, exactdots)


def getK2toFitTmax(tstart: float, tmax: float):
    """
    Izračuna parameter k2 za prilagajanje največjemu času tmax.

    Parameters:
        tstart (float): Začetni čas.
        tmax (float): Maksimalni čas.

    Returns:
        float: Vrednost parametra k2.
    """
    A = np.log(1/10**-8 - 1)
    B = np.log(1/(1-10**-8) - 1)

    k1 = (tstart * B - tmax * A) / (B - A)
    k2 = - A / (tstart - k1)

    return k2

#evolvePQA_st(N = 4, J=1, hx=0, a = [""], drives=[HarmonicDrive(t0 = 0.1, tend=1000)])
#evolvePQA_st(N = 4, J=1, hx=0, a = [""], drives=[RootHarmonicDrive(t0 = 0.1, tend=1000)])
#evolvePQA_st(N = 4, J=1, hx=0, a = [""], drives=[LogHarmonicDrive(t0 = 0.01, tend=1000)])

#Kako izbirati a in tend-t0 ==> zih neka povezava
# tend določi a, tako da je hx(tend) = hx(inf) + endtolerance
#za harm in rootharm je dobro 10**-1, za logharm pa 10**-3 pri tmax = 1000
