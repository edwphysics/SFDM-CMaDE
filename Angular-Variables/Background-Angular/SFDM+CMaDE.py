'''

Scalar Field Dark Matter and Compton Mass Dark Energy with the Alma-Ureña (2016) system of equations.

Solution of the SFDM + CMaDE model equations. 
The density parameters of the components of the universe are calculated solving the dynamical equations using the ABMM4 method.

Modified from the original code by Luis Osvaldo Tellez Tovar for the paper "The quantum character of the Scalar Field Dark Matter" by Tonatiuh Matos

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import time

class DarkM:
    def __init__(self):

        # Scalar Field Constants
        self.mass = 1.e-22                  # Scalar field mass in eV
        self.H0   = 1.49e-33                # Hubble parameter in eV
        self.y1_0 = 2.* self.mass/self.H0   # x7: y1 Mass to Hubble Ratio

        # CMaDE Constants
        self.kc    = 0.42
        self.Q     = -0.43
        self.rs    = 0.4
        self.s0    = 1.e-5      # x8: Structure Fromation 

        # Initial Conditions
        self.Th_0   = np.pi/2.  # x1: Th Theta - to avoid Cot(0/2.)
        self.OmDM_0 = 0.22994   # x2: Om Omega_DM
        self.z_0    = 0.00004   # x3:  z Radiation 
        self.nu_0   = 0.00002   # x4: nu Neutrinos 
        self.b_0    = 0.04      # x5:  b Baryons
        self.OmDE_0 = 0.729     # x6:  l Lambda
        self.Omk_0  = 0.001     # Curvature

        # Scale factor range
        self.NP = 100000
        self.Ni = np.log(1.e-0)
        self.Nf = np.log(1.e-6)
        self.d  = (self.Nf - self.Ni)/ self.NP
        self.t  = [self.Ni + self.d* i for i in np.arange(self.NP)]

    # Runge-Kutta 4 initialices the method ABM4
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

        print("{:<20}\t{:<20}\t{:<20}".format("E-FOLDING", "FRIEDMANN", "OMEGA_LS"))

        for i in range(3, self.NP - 1):

            # Prints N, F, and Omega SFDM
            if i % 5000 == 0:
                k_Term = self.Omk_0* (y[i,6]/self.y1_0)**2* np.exp(-2* t[i]) 
                print("{:<10}\t{:<10}\t{:<10}".format(t[i], y[i,1] + y[i,2]**2 + y[i,3]**2 + y[i,4]**2 + y[i,5]**2 + y[i,7]**2 + k_Term, y[i,7]))

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
        y0 = np.array([np.sqrt(self.Th_0),       
                       self.OmDM_0,           
                       np.sqrt(self.z_0),  
                       np.sqrt(self.nu_0),  
                       np.sqrt(self.b_0),     
                       np.sqrt(self.OmDE_0* (0/5.)),     
                       self.y1_0,
                       np.sqrt(self.OmDE_0* (5/5.))])

        # Solve the SoE with the ABM4 or RK4 algorithms
        y_result = self.ABM4(self.RHS, y0, self.t)
        #y_result = self.rk4(self.RHS, y0, self.t)

        return y_result

    # System of Equations
    def RHS(self, t, y):
        x1, x2, x3, x4, x5, x6, x7, x8 = y

        # Parameters
        CTer = 4/3.
        kc   = self.kc
        Q    = self.Q
        rs   = self.rs

        # CMaDE Factors
        CMF  = (Q/np.pi)* np.sqrt(3/2.)* np.exp(-t)
        SF   = (rs/np.pi)* np.sqrt(3/2.)* np.exp(-t)
        gamm = CMF* (kc/x2)* x6**3

        # Contributions 
        k_Term     = (2/3.)* self.Omk_0* (x7/self.y1_0)**2* np.exp(-2* t)
        SF_Term    = -(2/3.)* SF* x8**3
        CMaDE_Term = (kc - 1.)* (2/3.)* CMF* x6**3

        # Hubble Parameter Evolution 
        Pe = 2.* x2* np.sin(x1/2.)**2 + CTer* x3**2 + CTer* x4**2 + x5**2 + CMaDE_Term + k_Term + SF_Term

        return np.array([-3.* np.sin(x1) + x7 - 2.* gamm/ np.tan(x1/2.),
                         3.* (Pe - 1. + np.cos(x1))* x2 - gamm* x2,
                         1.5* x3* (Pe - CTer),
                         1.5* x4* (Pe - CTer),
                         1.5* x5* (Pe - 1.),
                         1.5* x6* Pe + CMF* x6**2,
                         1.5* x7* Pe,
                         1.5* x8* Pe + SF* x8**2])

    # Plotting Function
    def plot(self):
        #En este arreglo se guardan los resultados de la funcion solver. Las variables se acomodan como en la funcion RHS.
        z1, z2, z3, z4, z5, z6, z7, z8 = self.solver().T
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
        tiempo = w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = np.array([])

        for t, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8  in zip(self.t, z1, z2, z3, z4, z5, z6, z7, z8):
            #Resolucion de las graficas
            if i % 200 == 0:
               tiempo = np.append(tiempo, np.exp(t))  # Scale factor from e-folding N
               w1 = np.append(w1, aux1)
               w2 = np.append(w2, aux2)
               w3 = np.append(w3, aux3)
               w4 = np.append(w4, aux4)
               w5 = np.append(w5, aux5)
               w6 = np.append(w6, aux6)
               w7 = np.append(w7, aux7)
               w8 = np.append(w8, aux8)
            i += 1

        # Background Scalar Field
        ax2.semilogx(tiempo, -2* np.sqrt(6.)* np.sqrt(w2)* np.cos(w1/2.)/w7, 'black', label=r"$\kappa\Phi_0$")
        ax2.set_ylabel(r'$\kappa\Phi_0(a)$', fontsize=20)
        ax2.set_xlabel(r'$a$', fontsize=15)
        ax2.legend(loc = 'best', fontsize = 'xx-large')
        fig2.savefig('Phi0.pdf')
        #plt.show()

        #Dark Matter
        ax3.semilogx(tiempo, w2, 'black', label=r"$\Omega_{SFDM}$")
        ax3.semilogx(tiempo, w3**2, 'blue', label=r"$\Omega_{\gamma}$")     # Radiation
        ax3.semilogx(tiempo, w4**2, 'orange', label=r"$\Omega_{v}$")        # Neutrinos
        ax3.semilogx(tiempo, w5**2, 'red', label=r"$\Omega_b$")             # Baryons
        ax3.semilogx(tiempo, w6**2 + w8**2, 'green', label=r"$\Omega_{\Lambda}$")   # Lambda
        ax3.set_ylabel(r'$\Omega(a)$', fontsize=20)                         
        ax3.set_xlabel(r'$a$', fontsize=15)
        ax3.legend(loc = 'best', fontsize = 'xx-large')
        fig3.savefig('Omegas.pdf')
        #plt.show()        

        # Friedmann Restriction
        k_Term = self.Omk_0* (w7/self.y1_0)**2* np.exp(-2* np.log(tiempo))
        ax9.semilogx(tiempo, w2 + w3**2 + w4**2 + w5**2 + w6**2 + w8**2 + k_Term, 'black', label=r"$F$")
        ax9.set_ylabel(r'$F(a)$', fontsize=20)
        ax9.set_xlabel(r'$a$', fontsize=15)
        ax9.legend(loc = 'best', fontsize = 'xx-large')
        fig9.savefig('F.pdf')
        #plt.show()

        # Spurious Variable
        ax8.semilogx(tiempo, w7, 'black', label=r"$y1$")
        ax8.set_ylabel(r'$y1$', fontsize=20)
        ax8.set_xlabel(r'$a$', fontsize=15)
        ax8.legend(loc = 'best', fontsize = 'xx-large')
        fig8.savefig('y1.pdf')
        #plt.show()

# Runs only if the script is self contained
if __name__ == '__main__':
    start_time = time.time()

    DM = DarkM()
    DM.plot()

    # Prints total evaluation time with 3 decimals
    print("\nEvaluation time = ", "{:.3f}".format(time.time() - start_time)," seconds")
