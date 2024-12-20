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
        self.mass = 1.e-22                  # Scalar field mass in eV
        self.H0   = 1.42919e-33             # Hubble constant in eV
        self.y1_0 = 2.* self.mass/self.H0   # x7: y1 Mass to Hubble Ratio

        # Cut-off Parameter
        self.theta_star = 1.e2

        # CMaDE Constants
        # Turn off CMaDE with kc = Q = 0
        self.kc = 0.
        self.Q  = 0.

        # Initial Conditions a=1
        self.a_0    = 1.e-0
        self.OmDM_0 = 0.2501       # x2: Om Omega_DM
        self.Omz_0  = 5.50234e-5   # x3:  z Radiation 
        self.Omnu_0 = 3.74248e-5   # x4: nu Neutrinos 
        self.Omb_0  = 0.0498998    # x5:  b Baryons
        self.Omk_0  = 0.001        # Curvature

        # Total Radiation & Matter
        self.OmR_0  = self.Omnu_0 + self.Omz_0
        self.OmM_0  = self.OmDM_0 + self.Omb_0

        # Initial Conditions a_i
        self.a_i    = 1.e-14
        self.scH    = np.sqrt(self.OmR_0* self.a_i**(-4) + self.OmM_0* self.a_i**(-3)) # Adimensional Hubble Parameter at RD
        self.OmDM_i = (self.OmDM_0/self.scH**2)* self.a_i**(-3) 
        self.Omb_i  = (self.Omb_0/self.scH**2)* self.a_i**(-3)
        self.Omz_i  = (self.Omz_0/self.scH**2)* self.a_i**(-4)
        self.Omnu_i = (self.Omnu_0/self.scH**2)* self.a_i**(-4)
        self.Omk_i  = (self.Omk_0/self.scH**2)* self.a_i**(-2)

        # x1: Th Theta -- From Eq. 2.14 & 2.16 Ureña-Gonzalez
        t0 = 6.60996e32 # Age of the Universe in eV
        self.Th_0   = 2.* self.mass* t0
        self.Th_i   = (1/5.)* (self.y1_0/np.sqrt(self.OmR_0))* self.a_i**2

        # x2: OmegaDM_i from OmegaDM_0
        #self.OmDM_i = self.a_i* (self.OmDM_0/self.OmR_0)* (4.* self.Th_i**2/np.pi**2)**(3/4)* ((9. + np.pi**2/4.)/(9. + self.Th_i**2))**(3/4)

        # x6: CMaDE Variable -- Friedmann Restriction  
        self.OmDE_0 = 1. - self.Omk_0 - self.OmDM_0 - self.Omz_0 - self.Omnu_0 - self.Omb_0 
        #self.OmDE_i = 1. - self.Omk_i - self.OmDM_i - self.Omz_i - self.Omnu_i - self.Omb_i
        self.OmDE_i = self.OmDE_0/self.scH**2

        # x7: SF Mass to Hubble Parameter Ratio at a_i
        #self.y1_i   = self.y1_0/ self.scH
        # From Eq. 2.9a Ureña-Gonzalez
        self.y1_i   = 5* self.Th_i

        # Scale factor range
        self.NP = 100000
        self.Ni = np.log(self.a_i)
        self.Nf = np.log(self.a_0)
        self.d  = (self.Nf - self.Ni)/ self.NP
        self.t  = [self.Ni + self.d* i for i in np.arange(self.NP)]

        # Plots saved at...
        self.directory = '../../../Report/mastersthesis/plots/SFDM-UGD-FWD/'

        # Save Initial Conditions
        with open(self.directory + "IC.txt", "w") as file:
        # Write the constants to the file
            file.write(f"N: \t\t{self.NP}\n\n")
            file.write(f"CMaDE Q: \t{self.Q}\n")
            file.write(f"CMaDE kc: \t{self.kc}\n\n")
            file.write(f"Theta_i: \t{self.Th_i}\n")
            file.write(f"OmegaDM_i: \t{self.OmDM_i}\n")
            file.write(f"Omegab_i: \t{self.Omb_i}\n")
            file.write(f"Omegaz_i: \t{self.Omz_i}\n")
            file.write(f"Omeganu_i: \t{self.Omnu_i}\n")
            file.write(f"Omegak_i: \t{self.Omk_i}\n")
            file.write(f"OmegaDE_i: \t{self.OmDE_i}\n")
            file.write(f"y1_i: \t\t{self.y1_i}\n")
            file.write(f"\n")
            file.write(f"Theta_0: \t{self.Th_0}\n")
            file.write(f"OmegaDM_0: \t{self.OmDM_0}\n")
            file.write(f"Omegab_0: \t{self.Omb_0}\n")
            file.write(f"Omegaz_0: \t{self.Omz_0}\n")
            file.write(f"Omeganu_0: \t{self.Omnu_0}\n")
            file.write(f"Omegak_0: \t{self.Omk_0}\n")
            file.write(f"OmegaDE_0: \t{self.OmDE_0}\n")
            file.write(f"y1_0: \t\t{self.y1_0}\n")

    # Cut-off Trigonometric Functions
    def sin_star(self, theta):
        return (1/2.)* (1. - np.tanh(theta**2 - self.theta_star**2))* np.sin(theta)

    def cos_star(self, theta):
        return (1/2.)* (1. - np.tanh(theta**2 - self.theta_star**2))* np.cos(theta)

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
                print("{:<10}\t{:<10}".format(t[i], y[i,1] + y[i,2] + y[i,3] + y[i,4] + y[i,5] + k_Term))

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
        y0 = np.array([self.Th_i,       
                       self.OmDM_i,           
                       self.Omz_i,
                       self.Omnu_i,
                       self.Omb_i,
                       self.OmDE_i,
                       self.y1_i])

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
        gamm = CMF* (kc/x2)* x6**(3/2)

        # Contributions 
        k_Term     = (2/3.)* self.Omk_0* (x7/self.y1_0)**2* np.exp(-2* t)
        CMaDE_Term = (kc - 1.)* (2/3.)* CMF* x6**(3/2)

        # Hubble Parameter Evolution 
        Pe = 2.* x2* self.sin_star(x1/2.)**2 + CTer* (x3 + x4) + x5 + k_Term + CMaDE_Term

        return np.array([-3.* self.sin_star(x1) + x7 - 2.* gamm* self.cos_star(x1/2.)/ np.sin(x1/2.),
                         3.* (Pe - 1. + self.cos_star(x1))* x2 - 2* gamm* x2,
                         3.* x3* (Pe - CTer),
                         3.* x4* (Pe - CTer),
                         3.* x5* (Pe - 1.),
                         3.* x6* Pe + 2.* CMF* x6**(3/2),
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

        fig5 = plt.figure(figsize=(10,6)) 
        ax5  = fig5.add_subplot(111)       

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
        ax2.semilogx(tiempo, -2* np.sqrt(6.)* np.sqrt(w2)* self.cos_star(w1/2.)/w7, 'black')
        ax2.set_ylabel(r'$\kappa\Phi_0(a)$', fontsize=20, fontweight='bold')
        ax2.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=18)
        #ax2.legend(loc = 'best', fontsize = 'xx-large')
        fig2.savefig(self.directory + "Phi0.pdf")
        #plt.show()

        # Parameter Densities
        ax3.semilogx(tiempo, w2, 'black', label=r"$\Omega_{\rm SFDM}$")     # Dark Matter
        ax3.semilogx(tiempo, w3, 'blue', label=r"$\Omega_{\gamma}$")     # Radiation
        ax3.semilogx(tiempo, w4, 'orange', label=r"$\Omega_{\nu}$")      # Neutrinos
        ax3.semilogx(tiempo, w5, 'red', label=r"$\Omega_b$")             # Baryons
        ax3.semilogx(tiempo, w6, 'green', label=r"$\Omega_{\Lambda}$")   # Lambda
        ax3.set_ylabel(r'$\Omega(a)$', fontsize=20, fontweight='bold')                         
        ax3.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.legend(loc = 'upper left', fontsize = '12')
        fig3.savefig(self.directory + 'Omegas.pdf')
        #plt.show()       

        # Effective EoS Parameter wphi
        ax4.semilogx(tiempo, -self.cos_star(w1), 'black')
        ax4.set_ylabel(r'$w_{\phi}(a)$', fontsize=20, fontweight='bold')
        ax4.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
        #ax4.legend(loc = 'best', fontsize = 'xx-large')
        ax4.tick_params(axis='both', which='major', labelsize=18)
        fig4.savefig(self.directory + 'wphi.pdf')
        #plt.show()

        # x1: Theta - Angular Variable
        ax5.semilogx(tiempo, w1, 'black')
        ax5.set_ylabel(r'$\theta(a)$', fontsize=20, fontweight='bold')
        ax5.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
        ax5.tick_params(axis='both', which='major', labelsize=18)
        fig5.savefig(self.directory + 'theta.pdf')

        # Friedmann Restriction
        k_Term = self.Omk_0* (w7/self.y1_0)**2* np.exp(-2* np.log(tiempo))
        ax9.semilogx(tiempo, w2 + w3 + w4 + w5 + w6 + k_Term, 'black')
        ax9.set_ylabel(r'$F(a)$', fontsize=20)
        ax9.set_xlabel(r'$a$', fontsize=20)
        ax9.tick_params(axis='both', which='major', labelsize=18)
        #ax9.legend(loc = 'best', fontsize = 'xx-large')
        fig9.savefig(self.directory + 'F.pdf')
        #plt.show()

        # Spurious Variable
        ax8.semilogx(tiempo, w7, 'black', label=r"$y1$")
        ax8.set_ylabel(r'$y1(a)$', fontsize=20, fontweight='bold')
        ax8.set_xlabel(r'$a$', fontsize=20, fontweight='bold')
        ax8.tick_params(axis='both', which='major', labelsize=18)
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
