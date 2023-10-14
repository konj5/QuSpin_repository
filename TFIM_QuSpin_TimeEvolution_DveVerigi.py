from quspin.operators import hamiltonian  # Hamiltoniani in operatorji
from quspin.basis import spin_basis_1d  # Spinov prostor Hilbertovega prostora
import numpy as np  # Generične matematične funkcije
import matplotlib.pyplot as plt  # Knjižnica za risanje grafov

hz = 0.01  # Jakost magnetnega polja v z smeri


def bin_array(num: int, m: int) -> list:
    """
    Pretvori pozitivno celo število `num` v m-bitni bitni vektor.

    Args:
        num (int): Pozitivno celo število za pretvorbo.
        m (int): Število bitov v bitnem vektorju.

    Returns:
        list: M-seznam, ki predstavlja m-bitni bitni vektor.
    """
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def exactDiag(N: int, hx1: float, hx2: float, J: float, JT: float) -> tuple:
    """
    Izračuna energijsko lastno vrednost in pripadajoči lastni vektor za podane parametre s kvantnim Hamiltonianom.

    Args:
        N (int): Število spinov v sistemu.
        hx1 (float): Moč transverzalnega magnetnega polja za verigo 1.
        hx2 (float): Moč transverzalnega magnetnega polja za verigo 2.
        J (float): Moč Isingove interakcije med sosednjimi spini v verigi.
        JT (float): Sklopitev med verigama.

    Returns:
        tuple: Energijska lastna vrednost in pripadajoči lastni vektor Hamiltoniana.
    """
    # Veriga 1 [0 : N], veriga 2 [N:2N]
    hx1_field = [[-hx1, i] for i in range(N)]  # Transverzalno polje
    hx2_field = [[-hx2, N + i] for i in range(N)]  # Transverzalno polje

    hz_field = [[hz, i] for i in range(2 * N)]  # Vertikalno polje

    J_interaction = [[-J, i, i + 1] for i in range(2 * N - 1) if i != N - 1]  # Isingova interakcija

    JT_interaction = [[-JT, i, N + i] for i in range(N)]  # Sklopitev med verigama

    static_list = [["zz", J_interaction], ["x", hx1_field], ["x", hx2_field], ["zz", JT_interaction], ["z", hz_field]]
    dynamic_list = []
    spin_basis = spin_basis_1d(2 * N)

    H = hamiltonian(static_list, dynamic_list, basis=spin_basis, dtype=np.float64)

    E, eigvect = H.eigsh(k=1, which="SA")

    return (E, eigvect)


def magnetization(ground_state: np.ndarray) -> tuple:
    """
    Izračuna magnetizacijo sistema na podlagi osnovnega stanja.

    Args:
        ground_state (np.ndarray): Osnovno stanje sistema.

    Returns:
        tuple: Skupna magnetizacija sistema, magnetizacija verige 1, magnetizacija verige 2.
    """
    M = 0
    M1 = 0
    M2 = 0
    N = int(np.round(np.log2(len(ground_state)))) // 2
    for i in range(len(ground_state)):
        m1 = 0
        m2 = 0
        state = (2 * bin_array(i, 2 * N) - 1)

        for j in range(0, N, 1):
            m1 += state[j]

        for j in range(N, 2 * N, 1):
            m2 += state[j]

        M += np.abs(ground_state[i] ** 2) * (m1 + m2) / (2 * N)
        M1 += np.abs(ground_state[i] ** 2) * m1 / N
        M2 += np.abs(ground_state[i] ** 2) * m2 / N

    return (M, M1, M2)


def anti_magnetization(ground_state: np.ndarray) -> tuple:
    """
    Izračuna protimagnetizacijo sistema na podlagi osnovnega stanja.

    Args:
        ground_state (np.ndarray): Osnovno stanje sistema.

    Returns:
        tuple: Skupna protimagnetizacija sistema, protimagnetizacija verige 1, protimagnetizacija verige 2.
    """
    M = 0
    M1 = 0
    M2 = 0
    N = int(np.round(np.log2(len(ground_state)))) // 2
    for i in range(len(ground_state)):
        m1 = 0
        m2 = 0
        state = (2 * bin_array(i, 2 * N) - 1)

        for j in range(0, N, 1):
            m1 += state[j] * (-1) ** j

        for j in range(N, 2 * N, 1):
            m2 += state[j] * (-1) ** j

        M += np.abs(ground_state[i] ** 2) * (m1 + m2) / (2 * N)
        M1 += np.abs(ground_state[i] ** 2) * m1 / N
        M2 += np.abs(ground_state[i] ** 2) * m2 / N

    return (M, M1, M2)


def timeEvolution(N: int, J: float, hxdrive: object, JTdrive: object, k: float):
    """
    Izvede časovni razvoj sistema s kvantnim Hamiltonianom.

    Args:
        N (int): Število spinov v sistemu.
        J (float): Moč Isingove interakcije med sosednjimi spini v verigi.
        hxdrive (object): Objekt, ki predstavlja transverzalno magnetno polje za verigo 1.
        JTdrive (object): Objekt, ki predstavlja sklopitev med verigama.
        k (float): Parameter, ki določa obliko pulza.

    Returns:
        np.ndarray: Seznam časovnih korakov za časovni razvoj sistema.
    """
    t0 = hxdrive.t0
    tend = hxdrive.tend

    # Veriga 1 [0 : N], veriga 2 [N:2N]
    hx_field1 = [[-1, i] for i in range(N)]  # Transverzalno polje
    hx_field2 = [[-hxdrive.hx2, N + i] for i in range(N)]  # Transverzalno polje

    hz_field = [[hz, i] for i in range(2 * N)]  # Vertikalno polje

    J_interaction = [[-J, i, i + 1] for i in range(2 * N - 1) if i != N - 1]  # Isingova interakcija

    JT_interaction = [[-1, i, N + i] for i in range(N)]  # Sklopitev med verigama

    static_list = [["zz", J_interaction], ["z", hz_field], ["x", hx_field2]]

    dynamic_list = [["x", hx_field1, hxdrive.drive, ["garbageparameter"]],
                    ["zz", JT_interaction, JTdrive.drive, [k]]]

    spin_basis = spin_basis_1d(2 * N)

    H = hamiltonian(static_list, dynamic_list, basis=spin_basis, dtype=np.float64)

    groundstate = exactDiag(N=N, J=J, hx1=hxdrive.hx0, hx2=hxdrive.hx2, JT=0)[1][:, 0]

    return (H.evolve(v0=groundstate, t0=t0, times=np.linspace(t0, tend *1.1, 1000)))


class hxDrive:
    def __init__(self, t0: float, tend: float, hx0: float, hxend: float, hx2: float, a: float) -> None:
        """
        Konstruktor za objekt, ki predstavlja transverzalno magnetno polje za verigo 1.

        Args:
            t0 (float): Začetni čas transverzalnega magnetnega polja.
            tend (float): Končni čas transverzalnega magnetnega polja.
            hx0 (float): Moč transverzalnega magnetnega polja na začetku.
            hxend (float): Moč transverzalnega magnetnega polja na koncu.
            hx2 (float): Moč transverzalnega magnetnega polja za verigo 2.
            a (float): Parameter za določanje oblike pulza.
        """
        self.hx2 = hx2  # hx2 je moč transverzalnega magnetnega polja za drugo verigo
        self.tmax = tend
        self.tend = tend
        self.t0 = t0
        self.tstart = t0
        self.hx0 = hx0
        self.hxend = hxend
        self.a = a
        self.b = a / (hx0 - hxend) * (tend - t0)

    def drive(self, t: float, garbage) -> float:
        """
        Izračuna vrednost transverzalnega magnetnega polja v času `t`.

        Args:
            t (float): Čas za izračun transverzalnega magnetnega polja.
            garbage: Nepomemben parameter, ki ga ne uporabljamo.

        Returns:
            float: Moč transverzalnega magnetnega polja v času `t`.
        """
        if t < self.t0:
            return self.hx0

        if t > self.tend:
            return self.hxend

        return self.a / ((t - self.t0) ** 2 + self.b) * (self.tend - t) + self.hxend


class JTDrive:
    def __init__(self, JT: float, w: float, hxstart: float, hxdrive: callable) -> None:
        """
        Konstruktor za objekt, ki predstavlja sklopitev med verigama.

        Args:
            JT (float): Sklopitev med verigama.
            w (float): Kotna frekvenca.
            hxstart (float): Začetna moč transverzalnega magnetnega polja.
            hxdrive (callable): Funkcija, ki predstavlja transverzalno magnetno polje za verigo 1.
        """
        A = hxstart - hxdrive.hxend
        B = hxdrive.a - 2 * A * hxdrive.t0
        C = A * (hxdrive.t0 ** 2 + hxdrive.b) - hxdrive.a * hxdrive.tend

        self.tstart = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        self.tmid = 1 / 2 * (self.tstart + hxdrive.tend)
        self.c = (hxdrive.tend - self.tmid)
        self.tend = hxdrive.tend

        self.JT = JT
        self.w = w

    def drive(self, t: float, k: float) -> float:
        """
        Izračuna vrednost sklopitve med verigama v času `t`.

        Args:
            t (float): Čas za izračun sklopitve med verigama.
            k (float): Parameter, ki določa obliko pulza.

        Returns:
            float: Vrednost sklopitve med verigama v času `t`.
        """
        if t < self.tstart:
            return 0

        if t > self.tend:
            return 0

        return self.JT * np.cos(self.w * t) * np.exp(-(t - self.tmid) ** 2 / (2 * (k * self.c) ** 2))
    
#JTDrive, ki se ga da enostavno uporabiti v izmeničniJTDrive
class JTDriveNew:
    def __init__(self, JT:float, w:float, tstart:float, tend:float) -> None:
        self.c =1/2 * (tend-tstart)
        self.tmid = 1/2 * (tstart + tend)
        self.tstart = tstart
        self.tend = tend
        self.JT = JT
        self.w = w

    def drive(self, t: float, k: float) -> float:
        if t < self.tstart:
            return 0

        if t > self.tend:
            return 0

        return self.JT * np.cos(self.w * t) * np.exp(-(t - self.tmid) ** 2 / (2 * (k * self.c) ** 2))
    
class ConstDrive():

    def __init__(self, value:float) -> None:
        self.value = value

    def drive(self, t:float, garbage) -> float:
        return self.value
    
class IzmeničniHxDrive():
    
    def __init__(self, splittimes:list, edgevalues:list, hx2:float, As:list) -> None:
        driveObjects = []
        driveTypes = []

        for i in range(len(splittimes)-1):
            tstart = splittimes[i]
            tend = splittimes[i+1]

            hstart = edgevalues[i]
            hend = edgevalues[i+1]
            
            if hstart != hend:
                driveObjects.append(hxDrive(t0=tstart,tend=tend,hx0=hstart,hxend=hend, hx2=hx2, a=As[i])) #Tu lahko HxDrive zamenjamo z katerokoli funkcijo ki poveže začetno in končno točko
                driveTypes.append("Non_Constant")
            else:
                driveObjects.append(ConstDrive(hstart))
                driveTypes.append("Constant")

        self.driveObjets = driveObjects
        self.splittimes = splittimes
        self.driveTypes = driveTypes
        self.t0 = splittimes[0]
        self.tend = splittimes[-1]
        self.hx2 = hx2
        self.hx0 = edgevalues[0]
        self.hxend = edgevalues[-1]
    
    def drive(self,t:float, garbage) -> float:
        if t < self.t0: return self.hx0
        if t > self.tend: return self.hxend

        I = len(self.splittimes)-1
        for i in range(len(self.splittimes)-1):
            if t >= self.splittimes[i] and t <= self.splittimes[i+1]:
                I = i
                break

        return self.driveObjets[I].drive(t,"garbage")
    
class IzmeničniJTDrive():
    def __init__(self, JTs:list, ws:list, IzmeničniHxDrive:IzmeničniHxDrive) -> None:
        driveObjects = []

        for i in range(len(IzmeničniHxDrive.driveTypes)):
            if IzmeničniHxDrive.driveTypes[i] != "Constant":
                driveObjects.append(ConstDrive(0))
            else:
                driveObjects.append(JTDriveNew(JT=JTs[i], w=ws[i], tstart=IzmeničniHxDrive.splittimes[i], tend=IzmeničniHxDrive.splittimes[i+1]))

        self.driveObjects = driveObjects
        self.IzmeničniHxDrive = IzmeničniHxDrive

    def drive(self,t:float, k:float) -> float:
        if t < self.IzmeničniHxDrive.t0: return 0
        if t > self.IzmeničniHxDrive.tend: return 0
        
        I = len(self.IzmeničniHxDrive.splittimes)-1
        for i in range(len(self.IzmeničniHxDrive.splittimes)-1):
            if t >= self.IzmeničniHxDrive.splittimes[i] and t <= self.IzmeničniHxDrive.splittimes[i+1]:
                I = i
                break

        return self.driveObjects[I].drive(t,k)


                
            

        
    
#DEMONSTRACIJE UPORABE RAZLIČNIH DRIVE OBJEKTOV


# ISTOČASNA IMPLEMENTACIJA
"""
ts = np.linspace(0,10,1000)

y1s = [hxDrive(0,10,10,0,100,10).drive(t,"trash") for t in ts]
y2s = [JTDrive(1,10,1,hxDrive(0,10,10,0,100,10)).drive(t,"trash") for t in ts]


plt.plot(ts, y1s, label = "$h_x$")
plt.plot(ts, y2s, label = "$J_T$")
plt.legend()
plt.show()
"""


# ZLEPLJENA HX FUNKCIJA
"""
ts = np.linspace(0,100,1000)

Drive = IzmeničniHxDrive([0,12,40,100], [10,2,1,0], 12222, [10,10,10,10])

y1s = [Drive.drive(t,"trash") for t in ts]
plt.plot(ts, y1s, label = "$h_x$")
plt.show()
"""


# JT IN HX SE SPREMINJATA IZMENIČNO
"""
ts = np.linspace(0,100,1000)

DriveHx = IzmeničniHxDrive([0,12,40,100], [10,2,2,0], 12222, [10,10,10,10])
DriveJT = IzmeničniJTDrive(["",2,"",""], ["",2,"",""],DriveHx)

y1s = [DriveHx.drive(t,"trash") for t in ts]
y2s = [DriveJT.drive(t,1/4) for t in ts]
plt.plot(ts, y1s, label = "$h_x$")
plt.plot(ts, y2s, label = "$h_x$")
plt.show()
"""
