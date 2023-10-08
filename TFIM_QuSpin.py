# Import necessary libraries
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Define a function to convert a positive integer into an m-bit bit vector
def bin_array(num: int, m: int) -> list:
    """
    Convert a positive integer num into an m-bit bit vector.
    Args:
        num (int): Positive integer to convert.
        m (int): Number of bits in the resulting bit vector.
    Returns:
        list: An array representing the bit vector.
    """
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

# Define a function for exact diagonalization of the TFIM Hamiltonian
def exactDiag(N: int, h: float, J: float) -> tuple:
    """
    Compute the exact diagonalization of the Transverse Field Ising Model (TFIM) Hamiltonian.
    Args:
        N (int): Number of spins.
        h (float): Transverse field strength.
        J (float): Interaction strength between neighboring spins.
    Returns:
        tuple: A tuple containing the eigenvalues (E) and eigenvectors (eigvect) of the Hamiltonian.
    """
    # Define the Hamiltonian terms
    h_field = [[-h, i] for i in range(N)]  # Transverse field
    J_interaction = [[-J, i, i + 1] for i in range(N - 1)]  # Ising interaction
    
    # Specify static and dynamic operators
    static_spin = [["zz", J_interaction], ["x", h_field]]
    dynamic_spin = []

    # Create a spin basis
    spin_basis = spin_basis_1d(N)

    # Initialize the Hamiltonian
    H = hamiltonian(static_spin, dynamic_spin, basis=spin_basis, dtype=np.float64)

    # Diagonalize the Hamiltonian to find eigenvalues and eigenvectors
    E, eigvect = H.eigh()

    return E, eigvect

# Define a function to calculate the magnetization of a ground state
def magnetization(ground_state: np.ndarray) -> float:
    """
    Calculate the magnetization of a ground state.
    Args:
        ground_state (np.ndarray): Ground state wavefunction.
    Returns:
        float: The magnetization of the ground state.
    """
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i] ** 2 * np.sum(2 * bin_array(i, N) - 1)) / N
    
    return M

# Define a function to calculate the antiferromagnetic order parameter of a ground state
def anti_magnetization(ground_state: np.ndarray) -> float:
    """
    Calculate the antiferromagnetic order parameter of a ground state.
    Args:
        ground_state (np.ndarray): Ground state wavefunction.
    Returns:
        float: The antiferromagnetic order parameter of the ground state.
    """
    AM = 0
    N = int(np.round(np.log2(len(ground_state))))
    
    for i in range(len(ground_state)):
        AMi = 0
        statei = 2 * bin_array(i, N) - 1
        
        for j in range(N):
            AMi += statei[j] * (-1) ** j
        
        AM += (ground_state[i] ** 2) * np.abs(AMi) / N
    
    return AM

# Define the main function for plotting the magnetization vs. transverse field strength
def main():
    """
    Main function for plotting magnetization vs. transverse field strength for the TFIM.
    """
    Ms = []
    hs = np.linspace(0, 2, 100)
    J = 1
    N = 10
    
    for h in hs:
        Ms.append(magnetization(exactDiag(N, h, J)[1][:, 0]))

    plt.plot(hs, Ms)
    plt.title(f"Standard TFIM (J = {J}, N = {N})")
    plt.xlabel("h")
    plt.ylabel("M")
    plt.show()

# Define the main function for plotting the antiferromagnetic order parameter vs. transverse field strength
def main_antiferomag():
    """
    Main function for plotting antiferromagnetic order parameter vs. transverse field strength for the antiferromagnetic TFIM.
    """
    Ms = []
    hs = np.linspace(0, 2, 100)
    J = -1
    N = 8
    
    for h in hs:
        Ms.append(anti_magnetization(exactDiag(N, h, J)[1][:, 0]))

    plt.plot(hs, Ms)
    plt.title(f"Antiferromagnetic TFIM (J = {J}, N = {N})")
    plt.xlabel("h")
    plt.ylabel("M")
    plt.show()

# Define a function for plotting a 3D surface of magnetization vs. J and h
def plot_3d():
    """
    Plot a 3D surface of magnetization vs. J and h for the TFIM.
    """
    n = 30
    Ms_2d = np.zeros((n, n))
    xdata, ydata, zdata = ([], [], [])
    hs = np.linspace(-2, 2, n)
    Js = np.linspace(-1, 1, n)
    
    for i in range(n):
        for j in range(n):
            Ms_2d[i, j] = magnetization(exactDiag(8, hs[j], Js[i])[1][:, 0])
            xdata.append(Js[i])
            ydata.append(hs[j])
            zdata.append(Ms_2d[i, j])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
    ax.set_xlabel("J")
    ax.set_ylabel("h")
    ax.set_zlabel("M")
    plt.show()

# Define a function for plotting a 3D surface of antiferromagnetic order parameter vs. J and h
def plot_3d_antiferomag():
    """
    Plot a 3D surface of antiferromagnetic order parameter vs. J and h for the antiferromagnetic TFIM.
    """
    n = 30
    Ms_2d = np.zeros((n, n))
    xdata, ydata, zdata = ([], [], [])
    hs = np.linspace(-2, 2, n)
    Js = np.linspace(-1, 1, n)
    
    for i in range(n):
        for j in range(n):
            Ms_2d[i, j] = anti_magnetization(exactDiag(8, hs[j], Js[i])[1][:, 0])
            xdata.append(Js[i])
            ydata.append(hs[j])
            zdata.append(Ms_2d[i, j])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
    ax.set_xlabel("J")
    ax.set_ylabel("h")
    ax.set_zlabel("M")
    plt.show()

# Define a function for plotting a contour plot of magnetization vs. J and h
def plot_contour():
    """
    Plot a contour plot of magnetization vs. J and h for the TFIM.
    """
    n = 100
    Ms_2d = np.zeros((n, n))
    xdata, ydata, zdata = ([], [], [])
    hs = np.linspace(-2, 2, n)
    Js = np.linspace(-1, 1, n)
    
    for i in range(n):
        for j in range(n):
            Ms_2d[i, j] = magnetization(exactDiag(8, hs[j], Js[i])[1][:, 0])
            xdata.append(Js[i])
            ydata.append(hs[j])
            zdata.append(Ms_2d[i, j])

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(Js, hs, Ms_2d, levels=20)
    fig.colorbar(cp)
    ax.set_xlabel('J')
    ax.set_ylabel('h')
    plt.show()

# Define a function for plotting magnetization vs. transverse field strength for various N
def plot_variable_N():
    """
    Plot magnetization vs. transverse field strength for various N in the TFIM.
    """
    for N in range(2, 18, 4):
        try:
            Ms = []
            hs = np.linspace(0, 2, 100)
            J = 1
            
            for h in hs:
                Ms.append(magnetization(exactDiag(N, h, J)[1][:, 0]))

            plt.plot(hs, Ms, label=f"N = {N}")
            print(N)
        except:
            break

    plt.title("Standard TFIM")
    plt.xlabel("h")
    plt.ylabel("M")
    plt.legend()
    plt.savefig("TFIM_variable_N")
    plt.show()

# Define a function for plotting a 3D surface of magnetization vs. J, h, and N
def plot_3d_variable_N():
    """
    Plot a 3D surface of magnetization vs. J, h, and N for the TFIM.
    """
    for N in range(2, 11):
        n = 30
        Ms_2d = np.zeros((n, n))
        xdata, ydata, zdata = ([], [], [])
        hs = np.linspace(-2, 2, n)
        Js = np.linspace(-1, 1, n)
        
        for i in range(n):
            for j in range(n):
                Ms_2d[i, j] = magnetization(exactDiag(N, hs[j], Js[i])[1][:, 0])
                xdata.append(Js[i])
                ydata.append(hs[j])
                zdata.append(Ms_2d[i, j])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
        ax.set_xlabel("J")
        ax.set_ylabel("h")
        ax.set_zlabel("M")
        ax.view_init(20, 200)
        
        fig.savefig(f"TFIM_3D_N={N}.pdf")
        plt.cla()

# Uncomment the desired functions to run them
# main()
# main_antiferomag()
# plot_3d()
# plot_3d_antiferomag()
# plot_contour()
# plot_variable_N()
# plot_3d_variable_N()
