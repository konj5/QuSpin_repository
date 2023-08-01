from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
import TFIM_QuSpin as OneChain

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

    E, eigvect = H.eigh()

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> tuple:
    
    M1 = 0
    M2 = 0
    N = int(np.round(np.log2(len(ground_state)))) // 2
    for i in range(len(ground_state)):
        m1 = 0
        m2 = 0
        state = (2*bin_array(i,2*N)-1)
        
        for j in range(0,N,1):
            m1 += state[j]

        for j in range(N,2*N,1):
            m2 += state[j]

        M1+= np.abs(ground_state[i]**2 * m1 / N)
        M2+= np.abs(ground_state[i]**2 * m2 / N)
        
    
    return (M1, M2)

def main():
    M1s = []
    M2s = [] #Sistem dveh verig
    J = 1
    JT = 1

    hs = np.linspace(0,2,100)
    for h in hs:
        M1s.append(magnetization(exactDiag(4,h,J,JT)[1][:,0])[0])
        M2s.append(magnetization(exactDiag(4,h,J,JT)[1][:,0])[1])

    M0s = [] #Primerjava z primerom z eno samo verigo
    for h in hs:
        M0s.append(OneChain.magnetization(OneChain.exactDiag(4,h,J)[1][:,0]))
        

    plt.plot(hs,M1s, label = "1. veriga")
    plt.plot(hs,M2s, label = "2. veriga")
    plt.plot(hs,M0s, label = "Brez dodatne verige")
    plt.title("Dvoverižni sistem")
    plt.legend()
    plt.show()


def plot_3d():
    n = 30
    Ms_2d = np.zeros((n,n))
    xdata, ydata, zdata = ([],[],[])
    hs = np.linspace(-20,20,n)
    JTs = np.linspace(-20,20,n)
    J = 1

    for i in range(len(JTs)):
        for j in range(len(hs)):
            print(f"{i}, {j}")
            Ms_2d[i,j] = magnetization(exactDiag(4,hs[j],J,JTs[i])[1][:,0])[0]
            xdata.append(JTs[i])
            ydata.append(hs[j])
            zdata.append(Ms_2d[i,j])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.plot_trisurf(xdata, ydata, zdata,cmap='viridis', edgecolor='none')
    ax.set_xlabel("JT")
    ax.set_ylabel("h")
    ax.set_zlabel("M")
    
    

    fig.add_axes(ax)

    plt.show()    

#main()

plot_3d()