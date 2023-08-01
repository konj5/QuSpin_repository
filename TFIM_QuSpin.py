from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library

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

    E, eigvect = H.eigh()

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> float:
    
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i]**2 * np.sum(2 *bin_array(i,N) - 1)) / N
    
    return M

def main():
    Ms = []
    hs = np.linspace(0,2,100)
    for h in hs:
        Ms.append(magnetization(exactDiag(8,h,1)[1][:,0]))

    plt.plot(hs,Ms)
    plt.title("Enoverižni sistem")
    plt.show()


def plot_3d():
    n = 20
    Ms_2d = np.zeros((n,n))
    xdata, ydata, zdata = ([],[],[])
    hs = np.linspace(-2,2,n)
    Js = np.linspace(-1,1,n)
    for i in range(n):
        for j in range(n):
            print(f"{i}, {j}")
            Ms_2d[i,j] = magnetization(exactDiag(8,hs[j],Js[i])[1][:,0])
            xdata.append(Js[i])
            ydata.append(hs[j])
            zdata.append(Ms_2d[i,j])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.plot_trisurf(xdata, ydata, zdata,cmap='viridis', edgecolor='none')
    ax.set_xlabel("J")
    ax.set_ylabel("h")
    ax.set_zlabel("M")
    
    

    fig.add_axes(ax)

    plt.show()    



#main()
    
#plot_3d()