from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d# Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library


hz = 0.01


def bin_array(num:int, m:int) -> list:
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def exactDiag(N:int, hx1:float, hx2:float, J:float, JT:float) -> tuple:

    #Veriga 1 [0 : N], veriga 2 [N:2N]
    hx1_field = [[-hx1, i] for i in range(N)] #Transverzalno polje
    hx2_field = [[-hx2, N+i] for i in range(N)] #Transverzalno polje

    hz_field = [[hz,i] for i in range(2*N)] #Vertikalno polje

    J_interaction = [[-J,i,i+1] for i in range(2*N-1) if i != N-1]  #Isingova interakcija

    JT_interaction =[[-JT,i,N+i] for i in range(N)] #sklopitev med verigama

    static_list = [["zz", J_interaction], ["x", hx1_field],["x", hx2_field], ["zz", JT_interaction], ["z", hz_field]]
    dynamic_list = []
    spin_basis = spin_basis_1d(2*N)

    H = hamiltonian(static_list, dynamic_list, basis=spin_basis,dtype=np.float64)

    E, eigvect = H.eigsh(k=1,which = "SA")

    return (E, eigvect)

def magnetization(ground_state:np.ndarray) -> tuple:
    
    M = 0
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

        M += np.abs(ground_state[i]**2) * (m1+m2) / (2*N)
        M1+= np.abs(ground_state[i]**2) * m1 / N
        M2+= np.abs(ground_state[i]**2) * m2 / N
        
    
    return (M ,M1, M2)

def anti_magnetization(ground_state:np.ndarray) -> tuple:
    
    M = 0
    M1 = 0
    M2 = 0
    N = int(np.round(np.log2(len(ground_state)))) // 2
    for i in range(len(ground_state)):
        m1 = 0
        m2 = 0
        state = (2*bin_array(i,2*N)-1)
        
        for j in range(0,N,1):
            m1 += state[j] * (-1)**j

        for j in range(N,2*N,1):
            m2 += state[j] * (-1)**j

        M += np.abs(ground_state[i]**2) * (m1+m2) / (2*N)
        M1+= np.abs(ground_state[i]**2) * m1 / N
        M2+= np.abs(ground_state[i]**2) * m2 / N
        
    
    return (M ,M1, M2)


def timeEvolution(N:int, J:float, hxdrive:object, JTdrive:object, k:float):

    t0 = hxdrive.t0
    tend = hxdrive.tend

    #Veriga 1 [0 : N], veriga 2 [N:2N]
    hx_field1 = [[-1, i] for i in range(N)] #Transverzalno polje
    hx_field2 = [[-hxdrive.hx2, N+i] for i in range(N)] #Transverzalno polje

    hz_field = [[hz,i] for i in range(2*N)] #Vertikalno polje

    J_interaction = [[-J,i,i+1] for i in range(2*N-1) if i != N-1]  #Isingova interakcija

    JT_interaction =[[-1,i,N+i] for i in range(N)] #sklopitev med verigama

    static_list = [["zz", J_interaction], ["z", hz_field] ,["x", hx_field2]]
    

    dynamic_list = [["x",hx_field1,hxdrive.drive, ["garbageparameter"]], ["zz",JT_interaction,JTdrive.drive,[k]] ]

    spin_basis = spin_basis_1d(2*N)

    H = hamiltonian(static_list, dynamic_list, basis=spin_basis,dtype=np.float64)

    #print(spin_basis[0])
    #print(bin_array(spin_basis[0], m = N))
    
    groundstate = exactDiag(N = N, J = J, hx1 = hxdrive.hx0, hx2= hxdrive.hx2, JT = 0)[1][:,0]

    #groundstate = [0 if i not in (0,2**N-1) else 1/np.sqrt(2) for i in range(2**N)]

    return(H.evolve(v0=groundstate, t0 = t0, times=np.linspace(t0,tend,1000)))
    

class hxDrive:
    ################ PRIBLIŽNA OCENA ZA DOBRO IZBIRO a, a = tend - t0

    #druge potencialno dobre izbire a, drive(polovica časa) = polovični hx, ali pa mogoče tak da minimizita d/dt drive pri t = tend

    def __init__(self, t0:float,tend:float, hx0:float, hxend:float, hx2:float ,a:float) -> None:

        #hx2 je hx za drugo verigo
        self.hx2 = hx2
        self.tmax = tend
        self.tend = tend
        self.t0 = t0
        self.tstart = t0
        self.hx0 = hx0
        self.hxend = hxend
        self.a = a
        self.b = a / (hx0-hxend) * (tend - t0)

    

    def drive(self, t:float, garbage) -> float:
        if t < self.t0:
            return self.hx0
        
        if t > self.tend:
            return self.hxend

        return self.a / ((t-self.t0)**2 + self.b) * (self.tend - t) + self.hxend
    
class JTDrive:

    def __init__(self, JT:float, w:float, hxstart:float, hxdrive:callable, ) -> None:
        A = hxstart - hxdrive.hxend
        B = hxdrive.a - 2*A*hxdrive.t0
        C = A * (hxdrive.t0**2 + hxdrive.b) - hxdrive.a * hxdrive.tend

        self.tstart = (-B + np.sqrt(B**2-4*A*C)) / (2*A)
        self.tmid =1/2 * (self.tstart + hxdrive.tend) 
        self.c = (hxdrive.tend - self.tmid)
        self.tend = hxdrive.tend



        self.JT = JT
        self.w = w

    

    def drive(self, t:float, k:float) -> float:
        # k nam pove kako ostra je gaussovka, dobra izbira je recimo 1/3 ali manjše

        if t < self.tstart:
            return 0
        
        if t > self.tend:
            return 0

        return self.JT * np.cos(self.w * t) * np.exp(-(t-self.tmid)**2 / (2*(k * self.c)**2))
    
class JTDriveAutoW:

    def __init__(self, JT:float, hxstart:float, hxdrive:callable, ) -> None:
        A = hxstart - hxdrive.hxend
        B = hxdrive.a - 2*A*hxdrive.t0
        C = A * (hxdrive.t0**2 + hxdrive.b) - hxdrive.a * hxdrive.tend

        self.tstart = (-B + np.sqrt(B**2-4*A*C)) / (2*A)
        self.tmid =1/2 * (self.tstart + hxdrive.tend) 
        self.c = 1/3 * (hxdrive.tend - self.tmid)
        self.tend = hxdrive.tend
        self.hxdrive = hxdrive



        self.JT = JT

    

    def drive(self, t:float, garbage) -> float:

        self.w = 2* (self.hxdrive.hx2 - self.hxdrive.drive(t, "trash"))

        if t < self.tstart:
            return 0
        
        if t > self.tend:
            return 0

        return self.JT * np.cos(self.w * t) * np.exp(-(t-self.tmid)**2 / (2*self.c**2))



"""
#Drive Demonstration
ts = np.linspace(0,10,1000)

y1s = [hxDrive(0,10,10,0,100,10).drive(t,"trash") for t in ts]
y2s = [JTDrive(1,10,1,hxDrive(0,10,10,0,100,10)).drive(t,"trash") for t in ts]
#y2s = [JTDriveAutoW(1,1,hxDrive(0,10,10,0,100,10)).drive(t,"trash") for t in ts]


plt.plot(ts, y1s, label = "$h_x$")
plt.plot(ts, y2s, label = "$J_T$")
plt.legend()
plt.show()
"""
