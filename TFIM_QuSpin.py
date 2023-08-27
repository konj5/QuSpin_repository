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

    #E, eigvect = H.eigsh(k=1,which = "SA")
    E, eigvect = H.eigh()

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> float:
    
    M = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        M += np.abs(ground_state[i]**2 * np.sum(2 *bin_array(i,N) - 1)) / N
    
    return M

def anti_magnetization(ground_state:np.ndarray) -> float:
    
    AM = 0
    N = int(np.round(np.log2(len(ground_state))))
    for i in range(len(ground_state)):
        AMi = 0
        statei = 2 *bin_array(i,N) - 1
        for j in range(N):
            AMi += statei[j] * (-1)**j
        
        AM += (ground_state[i]**2) * np.abs(AMi)/N
    
    return AM

def main():
    Ms = []
    hs = np.linspace(0,2,100)
    J = 1
    N = 8
    for h in hs:
        Ms.append(magnetization(exactDiag(N,h,J)[1][:,0]))

    plt.plot(hs,Ms)
    plt.title(f"Standardni TFIM (J = {J}, N = {N})")
    plt.xlabel("h")
    plt.ylabel("M")
    plt.show()
    
def main_antiferomag():
    Ms = []
    hs = np.linspace(0,2,100)
    J = -1
    N = 8
    for h in hs:
        Ms.append(anti_magnetization(exactDiag(N,h,J)[1][:,0]))

    plt.plot(hs,Ms)
    plt.title(f"Standardni TFIM (J = {J}, N = {N})")
    plt.xlabel("h")
    plt.ylabel("M")
    plt.show()


def plot_3d():
    n = 30
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
    
def plot_3d_antiferomag():
    n = 30
    Ms_2d = np.zeros((n,n))
    xdata, ydata, zdata = ([],[],[])
    hs = np.linspace(-2,2,n)
    Js = np.linspace(-1,1,n)
    for i in range(n):
        for j in range(n):
            print(f"{i}, {j}")
            Ms_2d[i,j] = anti_magnetization(exactDiag(8,hs[j],Js[i])[1][:,0])
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



def plot_contour():
    n = 100
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

    #f = interpolate.interp2d(x=xdata, y=ydata, z=zdata,kind="cubic")

    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(Js, hs, Ms_2d, levels = 20)
    fig.colorbar(cp)
    ax.set_xlabel('J')
    ax.set_ylabel('h')
    plt.show()


def plot_variable_N():
    for N in range(2, 18,4):
        try:
            Ms = []
            hs = np.linspace(0,2,100)
            J = 1
            for h in hs:
                Ms.append(magnetization(exactDiag(N,h,J)[1][:,0]))

            plt.plot(hs,Ms, label = f"N = {N}")
            print(N)
        except:
            break


    plt.title("Standardni TFIM")
    plt.xlabel("h")
    plt.ylabel("M")
    plt.legend()
    plt.savefig("TFIM_variable_N")
    plt.show()


def plot_3d_variable_N():
    for N in range(2,11):
        n = 30
        Ms_2d = np.zeros((n,n))
        xdata, ydata, zdata = ([],[],[])
        hs = np.linspace(-2,2,n)
        Js = np.linspace(-1,1,n)
        for i in range(n):
            for j in range(n):
                print(f"{i}, {j}")
                Ms_2d[i,j] = magnetization(exactDiag(N,hs[j],Js[i])[1][:,0])
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
        ax.view_init(20, 200)        
        

        fig.add_axes(ax)

        fig.savefig(f"TFIM_3D_N={N}.pdf")
        plt.cla()




#main()

#main_antiferomag()

#plot_3d()

#plot_3d_antiferomag()

#plot_contour()

#plot_variable_N()

#plot_3d_variable_N()
