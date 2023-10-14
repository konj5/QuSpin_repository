import TFIM_QuSpin_TimeEvolution2 as OneChain
import TFIM_QuSpin_TimeEvolution_DveVerigi as TwoChain
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Spinova osnovna baza Hilbertovega prostora
import numpy as np # Splošne matematične funkcije
import matplotlib.pyplot as plt # Knjižnica za risanje grafov
import pickle

# Število spinskih delcev v verigi
N = 4

# Vrednost izmenjalnih sklopov
J = 1

# Časovni interval
t0 = 0
tend = 10

# Začetna in končna vrednost transverzalnega polja za prvo verigo
hx0 = 10
hxend = 0

# Vrednost transverzalnega polja za drugo verigo
hx2 = 10

# Jakost sklopitve med verigama
JT = 2.1

# Vrednost transverzalnega polja pri kateri začnemo sklopitev
hxstart = 1

# Frekvenca nihanja za JTDrive
w = 2 * (hx2 - hxstart) # <----------------------------------------------------- To bi lahko dobil tudi tako, da samo zračunaš <v|H1|v> - <v|H2|v>, če prav razumem načeloma ni treba nič uganiti!

# Parameter za obliko funkcije hx drive
a = 10

# Parameter za obliko funkcije JT drive, skalira širino gaussovke (za več podatkov je vloga razvidna iz razreda JTDrive)
k = 1/5



# PODATKI ZA ENOVERIŽNI SISTEM
hxDrive = TwoChain.hxDrive(t0=t0, tend=tend, hx0=hx0, hxend=hxend, hx2=hx2, a=a)
JTDrive = TwoChain.JTDrive(JT=JT, w=w, hxstart=hxstart, hxdrive=hxDrive)

"""
Tole še ne dela
hxDrive = TwoChain.IzmeničniHxDrive(splittimes = [t0, 1/2 * (t0 + tend), tend], edgevalues = [hx0,hxstart,hxend], hx2 = hx2, As = [a,a,a])
JTDrive = TwoChain.IzmeničniJTDrive(JTs = ["",JT,""], ws=["",w,""], IzmeničniHxDrive=hxDrive)
"""

# PODATKI ZA DVOVERIŽNI SISTEM
ts = np.linspace(t0, tend *1.1, 1000)
vs2 = TwoChain.timeEvolution(N=N, J=J, hxdrive=hxDrive, JTdrive=JTDrive, k=k)

# Demonstracija drive funkcij
y1s = [hxDrive.drive(t, "trash") for t in ts]
y2s = [JTDrive.drive(t, k) for t in ts]
plt.plot(ts, y1s, label = "$h_x$")
plt.plot(ts, y2s, label = "$J_T$")
plt.legend()
plt.show()



############################ Primerjava magnetizacije
M1s = [TwoChain.magnetization(vs2[:,i])[1] for i in range(len(vs2[0,:]))]
M2s = [TwoChain.magnetization(vs2[:,i])[2] for i in range(len(vs2[0,:]))]

# Izračun magnetizacije za verigo OneChain
vs1 = OneChain.timeEvolution(N=N, J=J, drive=hxDrive.drive, hx=hxend, a="trash", t0=t0, tmax=tend)
MOneChain = [OneChain.magnetization(vs1[:,i]) for i in range(len(vs1[0,:]))]

"""
# Prikaži rezultate magnetizacije na grafu
plt.plot(ts, M1s, label="$M_1$")
plt.plot(ts, M2s, label="$M_2$")
plt.plot(ts, MOneChain, label="$M_{ena veriga}$")
plt.title("Magnetizacija")
plt.legend()
plt.show()
"""



############################ Primerjava energije za verigo 1 in 2
J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
J_interaction2 = [[-J,N+i,N+i+1] for i in range(N-1)]

# Funkcija za pridobitev hamiltonskega operatorja za verigo 1
def getHamiltonianChain1(N:int, hx1:float, J:float):
    J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
    hx1_field = [[-hx1, i] for i in range(N)] # Transverzalno polje
    spin_basis = spin_basis_1d(2*N)
    return hamiltonian([["zz", J_interaction1], ["x", hx1_field]], [], basis=spin_basis, dtype=np.float64)

# Funkcija za pridobitev hamiltonskega operatorja za verigo 2
def getHamiltonianChain2(N:int, hx2:float, J:float):
    J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
    hx2_field = [[-hx2, N+i] for i in range(N)] # Transverzalno polje
    spin_basis = spin_basis_1d(2*N)
    return hamiltonian([["zz", J_interaction1], ["x", hx2_field]], [], basis=spin_basis, dtype=np.float64)

# Izračun energije za verigo 1 in 2
E1s = [np.real(np.conjugate(vs2[:,i]).dot(getHamiltonianChain1(N=N, J=J, hx1=hxDrive.drive(ts[i],"trash")).dot(vs2[:,i]))) for i in range(len(vs2[0,:]))]
E2s = [np.real(np.conjugate(vs2[:,i]).dot(getHamiltonianChain2(N=N, J=J, hx2=hx2).dot(vs2[:,i]))) for i in range(len(vs2[0,:]))]

# Izračun energije za enoverižni primer
vs1 = OneChain.timeEvolution(N=N, J=J, drive=hxDrive.drive, hx=hxend, a="trash", t0=t0, tmax=tend)

# Funkcija za pridobitev hamiltonskega operatorja za enoverižni primer
def getHamiltonianChainOneChain(N:int, hx:float, J:float):
    J_interaction1 = [[-J,i,i+1] for i in range(N-1)]
    hx1_field = [[-hx, i] for i in range(N)] # Transverzalno polje
    spin_basis = spin_basis_1d(N)
    return hamiltonian([["zz", J_interaction1], ["x", hx1_field]], [], basis=spin_basis, dtype=np.float64)

# Izračun energije za enoverižni primer
Eonechains = [np.real(np.conjugate(vs1[:,i]).dot(getHamiltonianChainOneChain(N=N, J=J, hx=hxDrive.drive(ts[i],"trash")).dot(vs1[:,i]))) for i in range(len(vs1[0,:]))]

"""
# Prikaži energije na grafu
plt.plot(ts, E1s, label = "1. veriga")
plt.plot(ts, E2s, label = "2. veriga")
plt.plot(ts, Eonechains, label = "Enoverižni primer")
plt.title("Energija")
plt.legend()
plt.show()
"""


# Razdelitev okna za prikaz grafov
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Graf za spreminjanje vrednosti hx in JT v času
ax1.plot(ts, y1s, label="$h_x$")
ax1.plot(ts, y2s, label="$J_T$")
ax1.legend()

# Graf za magnetizacijo v času za obe verigi in enoverižni primer
ax2.plot(ts, M1s, label="$M_1$")
ax2.plot(ts, M2s, label="$M_2$")
ax2.plot(ts, MOneChain, label="Enoverižni primer")
ax2.set_title("Magnetizacija")
ax2.legend()

# Graf za energijo v času za obe verigi in enoverižni primer
ax3.plot(ts, E1s, label="1. veriga")
ax3.plot(ts, E2s, label="2. veriga")
ax3.plot(ts, Eonechains, label="Enoverižni primer")
ax3.set_title("Energija")
ax3.legend()

# Naslov celotnega grafa
fig.suptitle(f"N={N}, t0={t0}, tend={tend}, hx0={hx0}, hxend={hxend},hx2={hx2}, JT={JT}, hxstart={hxstart}")

# Prilagodi razporeditev grafov
fig.tight_layout()

# Prikaz grafov
plt.show()



