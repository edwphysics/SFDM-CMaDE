'''

Compton Mass Dark Energy & Scalar Field Dark Matter.

Combination of the SFDM and CMaDE model equations. 
Here, the density parameters of the components of the universe are calculated solving the dynamical equations using the ABMM4 method.

Modified from the original code by Luis Osvaldo Tellez Tovar for the paper "The quantum character of the Scalar Field Dark Matter" by Tonatiuh Matos

'''

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import time

class DarkM:
    def __init__(self):

        self.mass = 1.e-22 # Scalar field mass in eV
        self.H0   = 1.49e-33 # Hubble parameter in eV
        self.s    = self.mass/self.H0

        # Scale factor range
        self.NP = 100000
        self.Ni = np.log(1.e-0)
        self.Nf = np.log(1.e-6)
        self.d  = (self.Nf - self.Ni)/ self.NP
        self.t  = [self.Ni + self.d* i for i in np.arange(self.NP)]

    # Runge-Kutta 4 initial0ices the method ABM4
    def rk4(self, func, y_0, t):
        y = np.zeros([4, len(y_0)])
        y[0] = y_0

        for i, t_i in enumerate(t[:3]):

            h = self.d 
            k_1 = func(t_i, y[i])
            k_2 = func(t_i + h/2., y[i] + h/2.* k_1)
            k_3 = func(t_i + h/2., y[i] + h/2.* k_2)
            k_4 = func(t_i + h, y[i] + h* k_3)

            y[i+1] = y[i] + h/6.* (k_1 + 2.* k_2 + 2.* k_3 + k_4) # RK4 step

        return y

    # Adams-Bashforth 4/Moulton 4 Step Predictor/Corrector
    def ABM4(self, func, y_0, t):
        y = np.zeros([len(t), len(y_0)])
        
        # First 4 steps found w/ RK4 
        y[0:4] = self.rk4(func, y_0, t)
        k_1 = func(t[2], y[2])
        k_2 = func(t[1], y[1])
        k_3 = func(t[0], y[0])

        print("{:<20}\t{:<20}\t{:<20}".format("E-FOLDING", "FRIEDMANN", "OMEGA_SFDM"))

        for i in range(3, self.NP - 1):

            # Prints N, F, and Omega SFDM
            if i % 10000 == 0:
                print("{:<10}\t{:<10}\t{:<10}".format(t[i], np.sum(np.square(np.array(y[i,:-1]))), np.sum(np.square(np.array(y[i,:2])))))

            h   = self.d
            k_4 = k_3
            k_3 = k_2
            k_2 = k_1
            k_1 = func(t[i], y[i])
            #Adams-Bashforth predictor
            y[i+1] = y[i] + h* (55.* k_1 - 59.* k_2 + 37.* k_3 - 9.* k_4)/24.
            k_0    = func(t[i+1], y[i+1])
            #Adams-Moulton corrector
            y[i+1] = y[i] + h* (9.* k_0 + 19.* k_1 - 5.* k_2 + k_3)/24.

        return y

    # Initial conditions from today comsological observations
    def solver(self):
        y0 = np.array([np.sqrt(1.e-10),   # x: x0 Kinetic energy - small to avoid 1/x divergence
                       np.sqrt(0.22994),  # u: x2 Potential energy
                       np.sqrt(0.00004),  # z: x4 Radiation
                       np.sqrt(0.00002),  # n: x5 Neutrinos
                       np.sqrt(0.04),     # b: x6 Baryons
                       np.sqrt(0.73),     # l: x7 Lambda
                       1.e9])             # s: x8 Spurious Variable

        # Solve the SoE with the ABM4 or RK4 algorithms
        #y_result = self.ABM4(self.RHS, y0, self.t)
        #y_result = self.rk4(self.RHS, y0, self.t)
        result = solve_ivp(self.RHS, (self.Ni, self.Nf), y0, method = 'LSODA', t_eval = self.t)

        return result

    def RHS(self, t, y):
        x0, x2, x4, x5, x6, x7, x8 = y
        CTer = 4./3.
        Pe   = 2.* x0**2 + 4.* x4**2/3. + 4.* x5**2/3. + x6**2
        kc   = 1.
        Q    = 1.

        return np.array([-3.* x0 - x8* x2 + 1.5* x0* Pe - (Q* kc/np.pi)* np.sqrt(3/2.)* (x7**3/x0)* np.exp(-t),
                         x0* x8 + 1.5* x2* Pe,
                         1.5* x4* (Pe - CTer),
                         1.5* x5* (Pe - CTer),
                         1.5* x6* (Pe - 1.),
                         1.5* x7* Pe + (Q/np.pi)* np.sqrt(3/2.)* x7**2* np.exp(-t),
                         #-1.5* x8**(-2)])
                         1.5* Pe* x8])

#Abajo se tiene la funcion que va a imprimir las graficas.
    def plot(self):
        #En este arreglo se guardan los resultados de la funcion solver. Las variables se acomodan como en la funcion RHS.
        z0, z2, z4, z5, z6, z7, z8 = self.solver().y
        t = self.solver().t
        #x, u, z, nu, l, s, b = self.solver().T

        fig3 = plt.figure(figsize=(9,10))
        ax3  = fig3.add_subplot(111)

        fig2 = plt.figure(figsize=(9,10)) #Define una nueva ventana.
        ax2  = fig2.add_subplot(111)       #La grafica correspondiente a la nueva ventana.

        fig8 = plt.figure(figsize=(9,10)) #Define una nueva ventana.
        ax8  = fig8.add_subplot(111)       #La grafica correspondiente a la nueva ventana.

        fig9 = plt.figure(figsize=(9,10)) #Define una nueva ventana.
        ax9  = fig9.add_subplot(111)       #La grafica correspondiente a la nueva ventana.

        i = 0
        tiempo = w0 = w2 = w4 = w5 = w6 = w7 = w8 = np.array([])

        for t, aux0, aux2, aux4, aux5, aux6, aux7, aux8  in zip(t, z0, z2, z4, z5, z6, z7, z8):
            #Resolucion de las graficas
            if i % 200 == 0:
               tiempo = np.append(tiempo, np.exp(t))  # Scale factor from e-folding N
               w0 = np.append(w0, aux0)
               w2 = np.append(w2, aux2)
               w4 = np.append(w4, aux4)
               w5 = np.append(w5, aux5)
               w6 = np.append(w6, aux6)
               w7 = np.append(w7, aux7)
               w8 = np.append(w8, aux8)
            i += 1

        # Background Scalar Field
        ax2.semilogx(tiempo, np.sqrt(6.)* w2/w8, 'black', label=r"$\kappa\Phi_0$")
        ax2.set_ylabel(r'$\kappa\Phi_0(a)$', fontsize=20)
        ax2.set_xlabel(r'$a$', fontsize=15)
        ax2.legend(loc = 'best', fontsize = 'xx-large')
        fig2.savefig('Phi0.pdf')
        #plt.show()

        #Dark Matter
        ax3.semilogx(tiempo, w0**2 + w2**2, 'black', label=r"$\Omega_{SFDM}$")
        ax3.semilogx(tiempo, w4**2, 'blue', label=r"$\Omega_{\gamma}$") #radiacion
        ax3.semilogx(tiempo, w5**2, 'orange', label=r"$\Omega_{v}$") #neutrinos
        ax3.semilogx(tiempo, w6**2, 'red', label=r"$\Omega_b$")  #bariones
        ax3.semilogx(tiempo, w7**2, 'green', label=r"$\Omega_{\Lambda}$")  #constante cosmologica
        ax3.set_ylabel(r'$\Omega(a)$', fontsize=20) #original
        ax3.set_xlabel(r'$a$', fontsize=15)
        ax3.legend(loc = 'best', fontsize = 'xx-large')
        fig3.savefig('Omegas.pdf')
        #plt.show()        

        # Friedmann Restriction
        ax9.semilogx(tiempo, w0**2 + w2**2 + w4**2 + w5**2 + w6**2 + w7**2, 'black', label=r"$F$")
        ax9.set_ylabel(r'$F(a)$', fontsize=20)
        ax9.set_xlabel(r'$a$', fontsize=15)
        ax9.legend(loc = 'best', fontsize = 'xx-large')
        fig9.savefig('F.pdf')
        #plt.show()

        # Spurious Variable
        ax8.semilogx(tiempo, w8, 'black', label=r"$s$")
        ax8.set_ylabel(r'$s$', fontsize=20)
        ax8.set_xlabel(r'$a$', fontsize=15)
        ax8.legend(loc = 'best', fontsize = 'xx-large')
        fig8.savefig('s.pdf')
        #plt.show()

# Runs only if the script is self contained
if __name__ == '__main__':
    start_time = time.time()

    DM = DarkM()
    DM.plot()

    # Prints total evaluation time with 3 decimals
    print("\nEvaluation time = ", "{:.3f}".format(time.time() - start_time)," seconds")
