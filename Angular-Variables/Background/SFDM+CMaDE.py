'''

Scalar Field Dark Matter and Compton Mass Dark Energy model (CMaDE+SFDM) 
Simulation with the Ureña--Gonzalez (2016) Decomposition Method for the SFDM system.
Numerical Method: ABM4 Fourth-Order Adams-Bashford Predictor Adams-Moulton Corrector  

Modified from the original code by Luis Osvaldo Tellez Tovar 
Written for the paper "The quantum character of the Scalar Field Dark Matter" by Tonatiuh Matos

Edwin Pérez
MSc Student 
Cinvestav

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Enable LaTeX font rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')  # Set font to LaTeX default serif

class DarkM:
    def __init__(self):

        # Scalar Field Constants
        self.mass = 1.e-20                  # Scalar field mass in eV
        self.H0   = 1.42919e-33             # Hubble constant in eV
        self.y1_0 = 2.* self.mass/self.H0   # x7: y1 Mass to Hubble Ratio

        # CMaDE Constants
        # Turn off CMaDE with kc = Q = 0
        self.kc = 0.42
        self.Q  = -0.43

        # Initial Conditions
        self.OmDM_0 = 0.2501       # x2: Om Omega_DM
        self.Omz_0  = 5.50234e-5   # x3:  z Radiation 
        self.Omnu_0 = 3.74248e-5   # x4: nu Neutrinos 
        self.Omb_0  = 0.0498998    # x5:  b Baryons
        self.Omk_0  = 0.001        # Curvature

        # x6: CMaDE Variable -- Friedmann Restriction  
        self.OmDE_0 = 1. - self.Omk_0 - self.OmDM_0 - self.Omz_0 - self.Omnu_0 - self.Omb_0 

        # x1: Th Theta -- From Eq. 2.14 Ureña-Gonzalez
        t0 = 6.60996e32 # Age of the Universe
        self.Th_0   = 2.* self.mass* t0

        # Scale factor range
        self.NP = 100000
        self.Ni = np.log(1.e-0)
        self.Nf = np.log(1.e-6)
        self.d  = (self.Nf - self.Ni)/ self.NP
        self.t  = [self.Ni + self.d* i for i in np.arange(self.NP)]

        # Plots saved at...
        self.directory = '../../../Report/mastersthesis/plots/SFDM+CMade-UGD/'

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

        print("{:<20}\t{:<20}".format("E-FOLDING", "FRIEDMANN"))

        for i in range(3, self.NP - 1):

            # Prints N and F
            if i % (self.NP/10) == 0:
                k_Term = self.Omk_0* (y[i,6]/self.y1_0)**2* np.exp(-2* t[i]) 
                print("{:<10}\t{:<10}".format(t[i], y[i,1] + y[i,2]**2 + y[i,3]**2 + y[i,4]**2 + y[i,5]**2 + k_Term))

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
        y0 = np.array([self.Th_0,       
                       self.OmDM_0,           
                       np.sqrt(self.Omz_0),  
                       np.sqrt(self.Omnu_0),  
                       np.sqrt(self.Omb_0),     
                       np.sqrt(self.OmDE_0),     
                       self.y1_0])

        # Solve the SoE with the ABM4 or RK4 algorithms
        y_result = self.ABM4(self.RHS, y0, self.t)
        #y_result = self.rk4(self.RHS, y0, self.t)

        return y_result

    # System of Equations
    def RHS(self, t, y):
        x1, x2, x3, x4, x5, x6, x7 = y

        # Parameters
        CTer = 4/3.
        kc   = self.kc
        Q    = self.Q

        # CMaDE Factors
        CMF  = (Q/np.pi)* np.sqrt(3/2.)* np.exp(-t)
        gamm = CMF* (kc/x2)* x6**3

        # Contributions 
        k_Term     = (2/3.)* self.Omk_0* (x7/self.y1_0)**2* np.exp(-2* t)
        CMaDE_Term = (kc - 1.)* (2/3.)* CMF* x6**3

        # Hubble Parameter Evolution 
        Pe = 2.* x2* np.sin(x1/2.)**2 + CTer* x3**2 + CTer* x4**2 + x5**2 + k_Term + CMaDE_Term

        return np.array([-3.* np.sin(x1) + x7 - 2.* gamm/ np.tan(x1/2.),
                         3.* (Pe - 1. + np.cos(x1))* x2 - 2* gamm* x2,
                         1.5* x3* (Pe - CTer),
                         1.5* x4* (Pe - CTer),
                         1.5* x5* (Pe - 1.),
                         1.5* x6* Pe + CMF* x6**2,
                         1.5* x7* Pe])

    # Plotting Function
    def plot(self):
        #En este arreglo se guardan los resultados de la funcion solver. Las variables se acomodan como en la funcion RHS.
        z1, z2, z3, z4, z5, z6, z7 = self.solver().T
        #x, u, z, nu, l, s, b = self.solver().T

        fig2 = plt.figure(figsize=(10,6)) # Define una nueva ventana
        ax2  = fig2.add_subplot(111)      # La grafica correspondiente a la nueva ventana

        fig3 = plt.figure(figsize=(10,6)) 
        ax3  = fig3.add_subplot(111)       

        fig4 = plt.figure(figsize=(10,6)) 
        ax4  = fig4.add_subplot(111)

        fig8 = plt.figure(figsize=(10,6)) 
        ax8  = fig8.add_subplot(111)      

        fig9 = plt.figure(figsize=(10,6)) 
        ax9  = fig9.add_subplot(111)      

        i = 0
        tiempo = w1 = w2 = w3 = w4 = w5 = w6 = w7 = np.array([])

        for t, aux1, aux2, aux3, aux4, aux5, aux6, aux7  in zip(self.t, z1, z2, z3, z4, z5, z6, z7):
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
            i += 1

        # Saving Results
        data = np.column_stack((tiempo, w1, w2, w3, w4, w5, w6, w7))
        np.savetxt("data.dat", data)

        # Background Scalar Field
        ax2.semilogx(tiempo, -2* np.sqrt(6.)* np.sqrt(w2)* np.cos(w1/2.)/w7, 'black')
        ax2.set_ylabel(r'$\kappa\Phi_0(a)$', fontsize=18, fontweight='bold')
        ax2.set_xlabel(r'$a$', fontsize=18, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        #ax2.legend(loc = 'best', fontsize = 'xx-large')
        fig2.savefig(self.directory + "Phi0.pdf")
        #plt.show()

        # Parameter Densities
        ax3.semilogx(tiempo, w2, 'black', label=r"$\Omega_{\rm SFDM}$")         # Dark Matter
        ax3.semilogx(tiempo, w3**2, 'blue', label=r"$\Omega_{\gamma}$")     # Radiation
        ax3.semilogx(tiempo, w4**2, 'orange', label=r"$\Omega_{\nu}$")        # Neutrinos
        ax3.semilogx(tiempo, w5**2, 'red', label=r"$\Omega_b$")             # Baryons
        ax3.semilogx(tiempo, w6**2, 'green', label=r"$\Omega_{\Lambda}$")   # Lambda
        ax3.set_ylabel(r'$\Omega(a)$', fontsize=18, fontweight='bold')                         
        ax3.set_xlabel(r'$a$', fontsize=18, fontweight='bold')
        ax3.tick_params(axis='both', which='major', labelsize=13)
        ax3.legend(loc = 'upper left', fontsize = '12')
        fig3.savefig(self.directory + 'Omegas.pdf')
        #plt.show()       

        # Effective EoS Parameter wphi
        ax4.semilogx(tiempo, -np.cos(w1), 'black')
        ax4.set_ylabel(r'$w_{\phi}(a)$', fontsize=18, fontweight='bold')
        ax4.set_xlabel(r'$a$', fontsize=18, fontweight='bold')
        #ax4.legend(loc = 'best', fontsize = 'xx-large')
        ax4.tick_params(axis='both', which='major', labelsize=13)
        fig4.savefig(self.directory + 'wphi.pdf')
        #plt.show()

        # Friedmann Restriction
        k_Term = self.Omk_0* (w7/self.y1_0)**2* np.exp(-2* np.log(tiempo))
        ax9.semilogx(tiempo, w2 + w3**2 + w4**2 + w5**2 + w6**2 + k_Term, 'black')
        ax9.set_ylabel(r'$F(a)$', fontsize=20)
        ax9.set_xlabel(r'$a$', fontsize=15)
        ax9.tick_params(axis='both', which='major', labelsize=13)
        #ax9.legend(loc = 'best', fontsize = 'xx-large')
        fig9.savefig(self.directory + 'F.pdf')
        #plt.show()

        # Spurious Variable
        ax8.semilogx(tiempo, w7, 'black', label=r"$y1$")
        ax8.set_ylabel(r'$y1(a)$', fontsize=18, fontweight='bold')
        ax8.set_xlabel(r'$a$', fontsize=18, fontweight='bold')
        ax8.tick_params(axis='both', which='major', labelsize=13)
        #ax8.legend(loc = 'best', fontsize = 'xx-large')
        fig8.savefig(self.directory + 'y1.pdf')
        #plt.show()

# Runs only if the script is self contained
if __name__ == '__main__':
    start_time = time.time()

    DM = DarkM()
    DM.plot()

    # Prints total evaluation time with 3 decimals
    print("\nEvaluation time = ", "{:.3f}".format(time.time() - start_time)," seconds")
