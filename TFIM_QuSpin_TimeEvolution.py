# Import necessary libraries
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
import numpy as np  # Generic math functions
import matplotlib.pyplot as plt  # Plotting library
from scipy import interpolate  # Polynomial interpolation library

# Define the strength of the transverse magnetic field
hz = 0.01


def bin_array(num: int, m: int) -> list:
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def exactDiag(N: int, h: float, J: float) -> tuple:
    """
    Compute the exact diagonalization of a quantum spin chain.

    Parameters:
        N (int): Number of spins in the chain.
        h (float): Strength of the transverse magnetic field.
        J (float): Strength of the Ising interaction between neighboring spins.

    Returns:
        tuple: A tuple containing the eigenvalues (energies) and eigenvectors of the Hamiltonian.
    """

    # Construct the static and dynamic parts of the Hamiltonian
    h_field = [[-h, i] for i in range(N)]  # Transverse magnetic field

    # No periodic boundary conditions
    J_interaction = [[-J, i, i + 1] for i in range(N - 1)]  # Ising interaction

    static_spin = [["zz", J_interaction], ["z", [[hz, i] for i in range(N)]], ["x", h_field]]
    dynamic_spin = []

    # Create the spin basis
    spin_basis = spin_basis_1d(N)

    # Create the Hamiltonian operator
    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis, dtype=np.float64)

    # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
    E, eigvect = H.eigsh(k=2, which="SA")

    return (E, eigvect)


def magnetization(ground_state: np.ndarray) -> float:
    """
    Compute the magnetization of a quantum spin chain's ground state.

    Parameters:
        ground_state (np.ndarray): The ground state of the quantum spin chain.

    Returns:
        float: The magnetization of the ground state.
    """

    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i]) ** 2 * np.sum(2 * bin_array(i, N) - 1) / N

    return M


def anti_magnetization(ground_state: np.ndarray) -> float:
    """
    Compute the anti-magnetization of a quantum spin chain's ground state.

    Parameters:
        ground_state (np.ndarray): The ground state of the quantum spin chain.

    Returns:
        float: The anti-magnetization of the ground state.
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



def timeEvolution(N: int, hmax: float, J: float, a: float, drive: callable, t0: float, tmax: float):
    """
    Perform time evolution of a quantum spin chain under the given parameters.

    Parameters:
        N (int): Number of spins in the chain.
        hmax (float): Maximum strength of the transverse magnetic field.
        J (float): Strength of the Ising interaction between neighboring spins.
        a (float): Parameter for the drive.
        drive (callable): Callable function representing the drive.
        t0 (float): Initial time.
        tmax (float): Maximum time for the time evolution.

    Returns:
        np.ndarray: Array of evolved states at different time points.
    """

    # Define the Ising interaction terms
    J_interaction = [[-J, i, i + 1] for i in range(N - 1)]
    
    # Define the static and dynamic parts of the Hamiltonian
    static_spin = [["zz", J_interaction], ["z", [[hz, i] for i in range(N)]]]
    dynamic_list = [["x", [[-hmax, i] for i in range(N)], drive, [a]]]

    # Create the spin basis
    spin_basis = spin_basis_1d(N)

    # Create the Hamiltonian operator
    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis, dtype=np.float64)

    # Compute the ground state using exact diagonalization
    groundstate = exactDiag(N, 0, J)[1][:, 0]

    # Perform time evolution
    return H.evolve(v0=groundstate, t0=t0, times=np.linspace(t0, tmax, 1000))


def energyTimeEvolution(N: int, hmax: float, J: float, a: float, drive: callable, t0: float, tmax: float):
    """
    Compute the energy time evolution of a quantum spin chain.

    Parameters:
        N (int): Number of spins in the chain.
        hmax (float): Maximum strength of the transverse magnetic field.
        J (float): Strength of the Ising interaction between neighboring spins.
        a (float): Parameter for the drive.
        drive (callable): Callable function representing the drive.
        t0 (float): Initial time.
        tmax (float): Maximum time for the time evolution.

    Returns:
        list: List of energy values at different time points.
    """

    # Define the Ising interaction terms
    J_interaction = [[-J, i, i + 1] for i in range(N - 1)]
    
    # Define the static and dynamic parts of the Hamiltonian
    static_spin = [["zz", J_interaction], ["z", [[0, i] for i in range(N)]]]
    dynamic_list = [["x", [[-hmax, i] for i in range(N)], drive, [a]]]

    # Create the spin basis
    spin_basis = spin_basis_1d(N)

    # Create the Hamiltonian operator
    H = hamiltonian(static_spin, dynamic_list, basis=spin_basis, dtype=np.float64)

    # Compute the time points
    ts = np.linspace(t0, tmax, 1000)

    # Initialize a list to store energy values
    Es = []

    for i in range(len(ts)):
        state = H.evolve(v0=[0 if x != 0 else 1 for x in range(2 ** N)], t0=t0, times=ts[:, i])

        # Compute the energy and add it to the list
        E = np.conjugate(state).dot(H(time=ts[i]).dot(state))
        assert(np.abs(np.imag(E)) < 0.0001)
        E = np.real(E)
        Es.append(E)

    return Es


def energyComparison(N: int, hmax: float, J: float, a: float, drive: object):
    """
    Compare the ground state energy obtained via exact diagonalization and time evolution.

    Parameters:
        N (int): Number of spins in the chain.
        hmax (float): Maximum strength of the transverse magnetic field.
        J (float): Strength of the Ising interaction between neighboring spins.
        a (float): Parameter for the drive.
        drive (object): Drive object containing drive parameters.

    Returns:
        tuple: A tuple containing a string with the comparison results and a list of energy values.
    """
    
    # Compute the exact ground state energy using exact diagonalization
    (E0_exact, trash) = exactDiag(N, hmax, J)
    E0_exact = E0_exact[0]

    # Compute the ground state energy using time evolution
    Es = energyTimeEvolution(N, hmax, J, a, drive.drive, drive.t0, drive.tend)
    E0_evolved = Es[-1]

    # Compute the absolute difference between the two energy values
    energy_difference = np.abs(E0_exact - E0_evolved)

    # Create a formatted string for the comparison results
    comparison_result = f"Exact ground state: {E0_exact}, Evolved: {E0_evolved}, difference: {energy_difference}"

    return (comparison_result, [E0_exact, E0_evolved, energy_difference])

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
        # f(tstart) = a
        
        k1 = np.log(a)/(self.tstart-self.tmax)
        
        if t < self.tstart:
            return 0
        
        if t < self.tmax:
            return np.exp(k1 * (t - self.tmax))
        
        return 1
               
class fermiDiracDriveDeprecated:

    def __init__(self, t0:float, tstart:float, tmax: float, tend:float) -> None:
        assert(t0<=tstart)
        assert(tstart<=tmax)
        assert(tmax<=tend)
        
        self.tmax = tmax
        self.tend = tend
        self.t0 = t0
        self.tstart = tstart

    def drive(self, t:float, a:float) -> float:
        #Začetna in končna točka sta f(tstart) = a in f(tmax) = 1-a
        
        A = np.log(1/a - 1)
        B = np.log(1/(1-a) - 1)
        
        k1 = (self.tstart * B - self.tmax * A) / (B - A)
        k2 = - A / (self.tstart - k1)
        
        if t < self.tstart:
            return 0
        
        if t < self.tmax:
            return 1 / (1 + np.exp(-k2 * (t - k1)))
        
        return 1
    
class fermiDiracDrive:

    def __init__(self, t0:float, tstart:float, _: float, __:float) -> None:
        assert(t0<=tstart)
        self.t0 = t0
        self.tstart = tstart

    def drive(self, t:float, k2:float) -> float:
        #Začetna in končna točka sta f(tstart) = 10**-8 in f(tmax) = 1-10**-8
        
        A = np.log(1/10**-8 - 1)
        B = np.log(1/(1-10**-8) - 1)
        
        k1 = self.tstart + A/k2
        self.tmax = (self.tstart * B - (B-A)*k1) / A
        self.tend = self.tmax + 10
        
        if t < self.tstart:
            return 0
        
        if t < self.tmax:
            return 1 / (1 + np.exp(-k2 * (t - k1)))
        
        return 1

def varyParameterA():
    """
    Varies the parameter 'a' and plots the energy difference.

    It computes the energy difference for a range of 'a' values and plots the results.

    Parameters:
        None

    Returns:
        None
    """
    diff = []
    As = np.linspace(0.102, 0.106, 100)
    for a in As:
        diff.append(energyComparison(N=12, hmax=1, J=1, a=a, drive=fermiDiracDrive(-100, 100))[1][2])
        print(a)

    plt.plot(As, diff)
    plt.scatter(As, diff)
    plt.show()


def varyParameterN():
    """
    Varies the parameter 'N' and plots the energy difference for different drives.

    It computes the energy difference for different 'N' values and different drive types (linear, exponential, Fermi-Dirac) and plots the results.

    Parameters:
        None

    Returns:
        None
    """
    diffLin = []
    diffExp = []
    diffFD = []
    ns = [i for i in range(1, 16, 1)]
    for N in ns:
        diffLin.append(energyComparison(N=12, hmax=1, J=1, a=0, drive=linearDrive(100, 200))[1][2])
        diffExp.append(energyComparison(N=N, hmax=1, J=1, a=0.0469, drive=exponentialDrive(-100, 100))[1][2])
        diffFD.append(energyComparison(N=N, hmax=1, J=1, a=0.049, drive=fermiDiracDrive(-100, 100))[1][2])
        print(N)

    plt.plot(ns, diffLin, label="linear")
    plt.scatter(ns, diffLin)

    plt.plot(ns, diffExp, label="exponential")
    plt.scatter(ns, diffExp)

    plt.plot(ns, diffFD, label="fermi-dirac")
    plt.scatter(ns, diffFD)

    plt.legend()
    plt.show()


def getE0_exact(data):
    """
    Get the exact ground state energy and state for given parameters.

    It computes the exact ground state energy and state for a given set of parameters and stores the results in a dictionary for future use.

    Parameters:
        data (tuple): A tuple containing N, hmax, and J parameters.

    Returns:
        tuple: A tuple containing the exact ground state energy and state.
    """
    data = tuple(data)

    if data in E0_exactdict.keys():
        return E0_exactdict[data]

    (E0_exact, basestate) = exactDiag(data[0], data[1], data[2])
    E0_exactdict[data] = (E0_exact[0], basestate[:, 0])
    return (E0_exact[0], basestate[:, 0])


def evolveEnergy(N: int, J: float, hmax: float, a: list, drives: list):
    """
    Compute and plot the energy evolution over time for different drives.

    It computes the energy evolution over time for different drive types and plots the results.

    Parameters:
        N (int): Number of particles.
        J (float): Interaction strength.
        hmax (float): Maximum field strength.
        a (list): List of parameters 'a' for different drives.
        drives (list): List of drive objects for different drives.

    Returns:
        None
    """
    #exact diagonalization
    E0_exact = getE0_exact((N, hmax, J))[0]

    #evolution
    Ess = []
    dss = []
    tss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0, a[i])

        Es = energyTimeEvolution(N, hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)
        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

        Ess.append(Es)
        dss.append(ds)
        tss.append(ts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    colors = ["red", "blue", "orange", "green", "purple", "pink"]

    tempe = [1]
    for i in range(len(Ess)):
        tempe.append(Ess[i][-1])

    for i in range(len(Ess)):
        ax1.plot(tss[i], dss[i], color=colors[i])
        ax2.plot(tss[i], Ess[i], color=colors[i])
        ax3.axhline(y=Ess[i][-1], label=f"{a[i]}", color=colors[i])

    ax1.set_title("Drive Function")
    ax2.set_title("E(t), Dashed Line is Exact Value")
    ax3.set_title("Final Energies")

    ax3.axhline(y=E0_exact, linestyle="dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe)) / 2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe)) / 2))

    plt.show()


def evolveEnergyData(N: int, J: float, hmax: float, a: list, drives: list):
    """
    Compute and return the energy evolution over time for different drives.

    It computes the energy evolution over time for different drive types and returns the results.

    Parameters:
        N (int): Number of particles.
        J (float): Interaction strength.
        hmax (float): Maximum field strength.
        a (list): List of parameters 'a' for different drives.
        drives (list): List of drive objects for different drives.

    Returns:
        tuple: A tuple containing lists of energy evolution, drive functions, and the exact ground state energy.
    """
    #exact diagonalization
    E0_exact = getE0_exact((N, hmax, J))[0]

    #evolution
    Ess = []
    dss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0, a[i])

        print(i)

        Es = energyTimeEvolution(N, hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)
        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

        Ess.append(Es)
        dss.append(ds)

    return (Ess, dss, E0_exact)


def evolveDotProduct(N: int, J: float, hmax: float, a: list, drives: list):
    """
    Compute and plot the dot product evolution over time for different drives.

    It computes the dot product evolution over time for different drive types and plots the results.

    Parameters:
        N (int): Number of particles.
        J (float): Interaction strength.
        hmax (float): Maximum field strength.
        a (list): List of parameters 'a' for different drives.
        drives (list): List of drive objects for different drives.

    Returns:
        tuple: A tuple containing lists of dot product evolution and time.
    """
    #exact diagonalization
    basestate = getE0_exact((N, hmax, J))[1]

    #evolution
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0, a[i])

        print(i)
        vs = timeEvolution(N, hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)

        dots = []
        for j in range(len(vs[0, :])):
            dots.append(np.abs(vs[:, j].dot(basestate)) ** 2)

        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

        dotss.append(dots)
        dss.append(ds)
        tss.append(ts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    colors = ["red", "blue", "orange", "green", "purple", "pink"]

    tempe = [1]
    for i in range(len(dotss)):
        tempe.append(dotss[i][-1])

    for i in range(len(dotss)):
        ax1.plot(tss[i], dss[i], color=colors[i])
        ax2.plot(tss[i], dotss[i], color=colors[i])
        ax3.axhline(y=dotss[i][-1], label=f"{a[i]}", color=colors[i])

    ax1.set_title("Drive Function")
    ax2.set_title("DotProduct^2(t)")
    ax3.set_title("Final Values")

    ax3.axhline(y=1, linestyle="dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe)) / 2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe)) / 2))

    plt.show()
    return (dotss, tss)


def evolveMagnetization(N: int, J: float, hmax: float, a: list, drives: list):
    """
    Compute and plot the magnetization evolution over time for different drives.

    It computes the magnetization evolution over time for different drive types and plots the results.

    Parameters:
        N (int): Number of particles.
        J (float): Interaction strength.
        hmax (float): Maximum field strength.
        a (list): List of parameters 'a' for different drives.
        drives (list): List of drive objects for different drives.

    Returns:
        None
    """
    #exact diagonalization
    basestate = getE0_exact((N, hmax, J))[1]

    #evolution
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0, a[i])

        print(i)
        vs = timeEvolution(N, hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)

        dots = []
        for j in range(len(vs[0, :])):
            dots.append(magnetization(vs[:, j]))

        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

        dotss.append(dots)
        dss.append(ds)
        tss.append(ts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    colors = ["red", "blue", "orange", "green", "purple", "pink"]

    tempe = [1]
    for i in range(len(dotss)):
        tempe.append(dotss[i][-1])

    for i in range(len(dotss)):
        ax1.plot(tss[i], dss[i], color=colors[i])
        ax2.plot(tss[i], dotss[i], color=colors[i])
        ax3.axhline(y=dotss[i][-1], label=f"{a[i]}", color=colors[i])

    ax1.set_title("Drive Function")
    ax2.set_title("Magnetization")
    ax3.set_title("Final Values")

    ax3.axhline(y=1, linestyle="dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe)) / 2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe)) / 2))

    plt.show()


def evolveAntiMagnetization(N: int, J: float, hmax: float, a: list, drives: list):
    """
    Compute and plot the anti-magnetization evolution over time for different drives.

    It computes the anti-magnetization evolution over time for different drive types and plots the results.

    Parameters:
        N (int): Number of particles.
        J (float): Interaction strength.
        hmax (float): Maximum field strength.
        a (list): List of parameters 'a' for different drives.
        drives (list): List of drive objects for different drives.

    Returns:
        None
    """
    #exact diagonalization
    basestate = getE0_exact((N, hmax, J))[1]

    #evolution
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0, a[i])

        print(i)
        vs = timeEvolution(N, hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)

        dots = []
        for j in range(len(vs[0, :])):
            dots.append(anti_magnetization(vs[:, j]))

        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

        dotss.append(dots)
        dss.append(ds)
        tss.append(ts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    colors = ["red", "blue", "orange", "green", "purple", "pink"]

    tempe = [1]
    for i in range(len(dotss)):
        tempe.append(dotss[i][-1])

    for i in range(len(dotss)):
        ax1.plot(tss[i], dss[i], color=colors[i])
        ax2.plot(tss[i], dotss[i], color=colors[i])
        ax3.axhline(y=dotss[i][-1], label=f"{a[i]}", color=colors[i])

    ax1.set_title("Drive Function")
    ax2.set_title("Anti-Magnetization")
    ax3.set_title("Final Values")

    ax3.axhline(y=1, linestyle="dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe)) / 2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe)) / 2))

    plt.show()


def evolvePQA_st(N: int, J: float, hmax: float, a: list, drives: list):
    """
    Compute and plot the evolution of P, Q, and A_st over time for different drives.

    It computes the evolution of P, Q, and A_st over time for different drive types and plots the results.

    Parameters:
        N (int): Number of particles.
        J (float): Interaction strength.
        hmax (float): Maximum field strength.
        a (list): List of parameters 'a' for different drives.
        drives (list): List of drive objects for different drives.

    Returns:
        None
    """
    #exact diagonalization
    basestate = getE0_exact((N, hmax, J))[1]

    #evolution
    exactdotss = []
    dotss = []
    dss = []
    tss = []
    for i in range(len(drives)):
        #initialize a for Fermi_Dirac
        drives[i].drive(0, a[i])

        print(i)
        vs = timeEvolution(N, hmax, J, a[i], drives[i].drive, drives[i].t0, drives[i].tend)

        ts = np.linspace(drives[i].t0, drives[i].tend, 100)

        exactstates = np.zeros_like(vs)
        for j in range(len(ts)):
            print(ts[j])
            exactstates[:, j] = exactDiag(N=N, J=J, h=hmax * drives[i].drive(ts[j], a[i]))[1][:, 0]

        dots = []
        exactdots = []
        for j in range(len(vs[0, :])):
            dots.append(np.abs(vs[:, j].dot(basestate)) ** 2)
            exactdots.append(np.abs(exactstates[:, j].dot(basestate)) ** 2)

        ds = []
        for t in ts:
            ds.append(drives[i].drive(t, a[i]))

        dotss.append(dots)
        exactdotss.append(exactdots)
        dss.append(ds)
        tss.append(ts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    colors = ["red", "blue", "orange", "green", "purple", "pink"]

    tempe = [1]
    for i in range(len(dotss)):
        tempe.append(dotss[i][-1])

    for i in range(len(dotss)):
        ax1.plot(tss[i], dss[i], color=colors[i])
        ax2.plot(tss[i], dotss[i], color=colors[i])
        ax2.plot(tss[i], exactdotss[i], color=colors[i], linestyle="dashed")
        ax3.axhline(y=dotss[i][-1], label=f"{a[i]}", color=colors[i])

    ax1.set_title("Drive Function")
    ax2.set_title("P (solid line), P_st (dashed line)")
    ax3.set_title("Final Values")

    ax3.axhline(y=1, linestyle="dashed")
    ax3.set_ylim((np.amin(tempe) - (np.amax(tempe) - np.amin(tempe)) / 2, np.amax(tempe) + (np.amax(tempe) - np.amin(tempe)) / 2))

    plt.show()


def getK2toFitTmax(tstart: float, tmax: float):
    """
    Compute the parameter k2 to fit a given tmax value.

    Parameters:
        tstart (float): Start time.
        tmax (float): Maximum time.

    Returns:
        float: The computed parameter k2.
    """
    A = np.log(1 / 10 ** -8 - 1)
    B = np.log(1 / (1 - 10 ** -8) - 1)

    k1 = (tstart * B - tmax * A) / (B - A)
    k2 = - A / (tstart - k1)

    return k2




"""
N = 8
resultLinear = energyComparison(N=N,hmax=1,J=1,a=0,drive=linearDriveDeprecated(100,200))
resultExponential = energyComparison(N=N,hmax=1,J=1,a=0.0469,drive=exponentialDriveDeprecated(-100,100))
resultFD = energyComparison(N=N,hmax=1,J=1,a=0.049,drive=fermiDiracDriveDeprecated(-100,100))

print(f"Linear -> {resultLinear[0]}")
print(f"Exponential -> {resultExponential[0]}")
print(f"FermiDirac -> {resultFD[0]}")
"""
    
    
#evolveEnergy(N=8,J=1,hmax=1,a=[0,0.001,getK2toFitTmax(1,100)],drives=[linearDrive(0,1,100,110), exponentialDrive(0,1,100,110), fermiDiracDrive(0,1,100,110)])
#evolveEnergy(N=8,J=1,hmax=0.1,a=[0,0.01,getK2toFitTmax(1,1000)],drives=[linearDrive(0,1,1000,1010), exponentialDrive(0,1,1000,1010), fermiDiracDrive(0,1,1000,1010)])
#evolveEnergy(N=8,J=1,hmax=0.1,a=[0,0.01,getK2toFitTmax(1,10000)],drives=[linearDrive(0,1,10000,10100), exponentialDrive(0,1,10000,10100), fermiDiracDrive(0,1,10000,10100)])

#evolveEnergy(N=8,J=1,hmax=0.1,a=[0.01, 0.1, 1, 10],drives=[fermiDiracDrive(0,1,"lol","pointless") for _ in range(4)])



"""
ks = np.linspace(2,0.01,200)
(Ess, trash, E0_exact) = evolveEnergyData(8,1,0.1,ks,[fermiDiracDrive(0,1,0,0) for _ in range(len(ks))])
Ess = np.array(Ess)
dE = Ess[:,-1] - E0_exact

plt.plot(ks, dE)
#plt.scatter(ks,dE, s = 3, color = "black")
plt.xlabel("k2")
plt.ylabel("E-E0")
plt.savefig("K2.svg")
plt.show()
"""


t0 = 0
tstart = 1
tmax = 8000
tend = tmax + 10


evolveDotProduct(N=4,J=1,hmax=1,a=[0,0.001,getK2toFitTmax(tstart,tmax)],drives=[linearDrive(t0,tstart,tmax,tend), exponentialDrive(t0,tstart,tmax,tend), fermiDiracDrive(t0,tstart,tmax,tend)])

evolveDotProduct(N=8,J=2,hmax=0.1,a=[getK2toFitTmax(1,i) for i in np.linspace(50,1000,4)],drives=[fermiDiracDrive(t0,tstart,tmax,tend) for i in range(4)])

"""
nJ,nh = 10,10
a = getK2toFitTmax(1,1000)
Js = np.linspace(-1,1,nJ)
hs = np.linspace(-1,1,nh)

xdata,ydata,zdata = ([],[],[])
for J in Js:
    for hmax in hs:
        print((J,hmax))

        #states = timeEvolution(N=8, hmax=hmax, J=J,a=a,drive=fermiDiracDrive(0,1,"","").drive, t0=0, tmax=1010)
        #endstate = states[-1]
        
        #M = magnetization(endstate)

        (Ess,trash,E0_exact) = evolveEnergyData(8,J,hmax,[a],[fermiDiracDrive(0,1,0,0)])
        Ess = np.array(Ess)
        E = Ess[0,-1] - E0_exact

        xdata.append(J)
        ydata.append(hmax)
        zdata.append(E)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(xdata, ydata, zdata,cmap='viridis', edgecolor='none')
ax.set_xlabel("J")
ax.set_ylabel("hmax")
ax.set_zlabel("E - E0")
ax.view_init(20, 200)        

fig.add_axes(ax)
plt.show()
"""

"""
basestate = getE0_exact((8,0.1,1))[1]
tmax = 400
vs = timeEvolution(N=8,hmax=0.1, J=1, a=getK2toFitTmax(1,tmax), drive = fermiDiracDrive(0,1,-1,-1).drive, t0=0, tmax=tmax)

dot = np.abs(vs[:,0].dot(basestate))**2
M = magnetization(vs[:,0])
AM = anti_magnetization(vs[:,0])

print(f"{dot}, {M}, {AM}")
"""

#evolveMagnetization(N=8,J=2,hmax=0.5,a=[getK2toFitTmax(1,i) for i in np.linspace(50,1000,4)],drives=[fermiDiracDrive(t0,tstart,tmax,tend) for i in range(4)])
#evolveAntiMagnetization(N=8,J=2,hmax=0.5,a=[getK2toFitTmax(1,i) for i in np.linspace(50,1000,4)],drives=[fermiDiracDrive(t0,tstart,tmax,tend) for i in range(4)])

#evolvePQA_st(N=8,J=1,hmax=3,a=[getK2toFitTmax(tstart,tmax)], drives=[fermiDiracDrive(t0,tstart,tmax,tend)])
