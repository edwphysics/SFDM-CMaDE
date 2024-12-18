'''

Scalar Field Dark Matter with the Alma-Ureña (2016) system of equations.

Solution of the SFDM model equations. Good solving parameters: NP = 10000000 and mass = 1.e-22 
The density parameters of the components of the universe are calculated solving the dynamical equations using the ABMM4 method.

Modified from the original code by Luis Osvaldo Tellez Tovar for the paper "The quantum character of the Scalar Field Dark Matter" by Tonatiuh Matos

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import time

class DarkM:
    def __init__(self):

        # System Constants
        self.mass = 1.e-22                  # Scalar field mass in eV
        self.H0   = 1.49e-33                # Hubble parameter in eV
        self.y1_0 = 2.* self.mass/self.H0   # Mass to Hubble Ratio
        self.km   = 1.e-12                  # Wavenumber to Mass Ratio

        # CMaDE Constants
        self.kc = 0.42
        self.Q  = -0.43

        # Initial Conditions
        self.OmDM_0 = 0.27         # x2: Om Omega_DM
        self.Omz_0  = 0.00005539   # x3:  z Radiation 
        self.Omnu_0 = 0.00004      # x4: nu Neutrinos 
        self.Omb_0  = 0.044        # x5:  b Baryons
        self.OmDE_0 = 0.605905     # x6:  l Lambda
        self.Omk_0  = 0.08         # Curvature

        # x1: Th Theta - From Eq. 2.16 Ureña-Gonzalez
        self.Th_0   = self.y1_0/ (5.* np.sqrt(self.Omnu_0 + self.Omz_0))  

        self.h0     = 0.        # x8:   h Metric Contiunity
        self.dh0    = 0.        # x9:  h' Diff. Metric Contiunity
        self.nu0    = 0.        # x10: th Pert. Oscillation
        self.alpha0 = 0.        # x11:  a Pert. Amplitude

        # Scale factor range
        self.NP = 100000
        self.Ni = np.log(1.e-0)
        self.Nf = np.log(1.e-6)
        self.d  = (self.Nf - self.Ni)/ self.NP
        self.t  = [self.Ni + self.d* i for i in np.arange(self.NP)]

        # Background Results
        self.BGND = self.solver_bg()

    # Runge-Kutta 4 initialices the method ABM4
    def rk4(self, func, y_0, t, z):
        y = np.zeros([4, len(y_0)])
        y[0] = y_0

        for i, t_i in enumerate(t[:3]):

            h = self.d 
            k_1 = func(t_i, y[i], z[i])
            k_2 = func(t_i + h/2., y[i] + h/2.* k_1, z[i])
            k_3 = func(t_i + h/2., y[i] + h/2.* k_2, z[i])
            k_4 = func(t_i + h, y[i] + h* k_3, z[i])

            y[i+1] = y[i] + h/6.* (k_1 + 2.* k_2 + 2.* k_3 + k_4) # RK4 step

        return y

    # Adams-Bashforth 4/Moulton 4 Step Predictor/Corrector
    def ABM4(self, func, y_0, t, z):
        y = np.zeros([len(t), len(y_0)])
        
        # First 4 steps found w/ RK4 
        y[0:4] = self.rk4(func, y_0, t, z)
        k_1 = func(t[2], y[2], z[2])
        k_2 = func(t[1], y[1], z[1])
        k_3 = func(t[0], y[0], z[0])

        print("{:<20}\t{:<20}".format("E-FOLDING", "FRIEDMANN"))

        for i in range(3, self.NP - 1):

            # Prints N and F
            if i % (self.NP/10) == 0 and func == self.RHS_bg:
                k_Term = self.Omk_0* (y[i,6]/self.y1_0)**2* np.exp(-2* t[i]) 
                print("{:<10}\t{:<10}".format(t[i], y[i,1] + y[i,2]**2 + y[i,3]**2 + y[i,4]**2 + y[i,5]**2 + k_Term))
            elif i % (self.NP/10) == 0 and func == self.RHS_pb:
                print("{:<10}".format(t[i]))

            h   = self.d
            k_4 = k_3
            k_3 = k_2
            k_2 = k_1
            k_1 = func(t[i], y[i], z[i])
            #Adams-Bashforth predictor
            y[i+1] = y[i] + h* (55.* k_1 - 59.* k_2 + 37.* k_3 - 9.* k_4)/24.
            k_0    = func(t[i+1], y[i+1], z[i+1])
            #Adams-Moulton corrector
            y[i+1] = y[i] + h* (9.* k_0 + 19.* k_1 - 5.* k_2 + k_3)/24.

        return y

    # Initial conditions from today cosmological observations
    def solver_bg(self):
        y0 = np.array([self.Th_0,       
                       self.OmDM_0,           
                       np.sqrt(self.Omz_0),  
                       np.sqrt(self.Omnu_0),  
                       np.sqrt(self.Omb_0),     
                       np.sqrt(self.OmDE_0),     
                       self.y1_0
                       ])        

        # Solve the SoE with the ABM4 or RK4 algorithms
        y_result = self.ABM4(self.RHS_bg, y0, self.t, np.zeros(self.NP))
        #y_result = self.rk4(self.RHS, y0, self.t)

        return y_result

    # System of Equations: Background
    def RHS_bg(self, t, y, z):
        x1, x2, x3, x4, x5, x6, x7 = y

        # Parameters
        kc   = self.kc
        Q    = self.Q

        # CMaDE Factors
        CMF  = (Q/np.pi)* np.sqrt(3/2.)* np.exp(-t)
        gamm = CMF* (kc/x2)* x6**3

        # Contributions 
        k_Term     = (2/3.)* self.Omk_0* (x7/self.y1_0)**2* np.exp(-2* t)
        CMaDE_Term = (kc - 1.)* (2/3.)* CMF* x6**3

        # Hubble Parameter Evolution 
        Pe = 2.* x2* np.sin(x1/2.)**2 + (4/3.)* x3**2 + (4/3.)* x4**2 + x5**2 + k_Term + CMaDE_Term

        return np.array([-3.* np.sin(x1) + x7 - 2.* gamm/ np.tan(x1/2.),
                         3.* (Pe - 1. + np.cos(x1))* x2 - 2* gamm* x2,
                         1.5* x3* (Pe - 4/3.),
                         1.5* x4* (Pe - 4/3.),
                         1.5* x5* (Pe - 1.),
                         1.5* x6* Pe + CMF* x6**2,
                         1.5* x7* Pe])

    # Initial conditions from VERIFY
    def solver_pb(self):
        y0 = np.array([self.h0,
                       self.dh0,
                       self.nu0,
                       self.alpha0])        

        # Solve the SoE with the ABM4 or RK4 algorithms
        y_result = self.ABM4(self.RHS_pb, y0, self.t[::-1], self.BGND[::-1])
        #y_result = self.rk4(self.RHS, y0, self.t)

        return y_result

    # System of Equations: Perturbations
    def RHS_pb(self, t, y, z):
        x8, x9, x10, x11 = y
        x1, x2, x3, x4, x5, x6, x7 = z

        CTer = 4./3.
        Pe   = 2.* x2* np.sin(x1/2.)**2 + CTer* x3**2 + CTer* x4**2 + x5**2
        km   = self.km 
        w    = km**2* x7* np.exp(-2*t)/2.

        return np.array([
                        # Perturbations
                        x9,
                        (1.5* Pe - 2.)* x9 + 6.* x2* np.exp(x11)* np.sin(x1/2.)* np.cos((x1 - x10)/2.),
                        -3.* (np.sin(x1) + np.sin(x1 - x10)) - w* (1 - np.cos(x1 - x10)) + np.exp(-x11)* x9* (np.cos(x10/2.) - np.cos(x1 - x10/2.)),
                        -(3/2.)* (np.cos(x1 - x10) + np.cos(x1)) - w* np.sin(x1 - x10)/2. + np.exp(-x11)* x9* (np.sin(x10/2.) + np.sin(x1 - x10/2.))/2.
                        ])

    # Plotting Function
    def plot(self):
        # Save the BG and PB results as lists
        z1, z2, z3, z4, z5, z6, z7 = self.BGND.T
        z8, z9, z10, z11 = self.solver_pb().T

        fig3 = plt.figure(figsize=(9,10))   
        ax3  = fig3.add_subplot(111)         

        fig2 = plt.figure(figsize=(9,10)) 
        ax2  = fig2.add_subplot(111)       

        fig8 = plt.figure(figsize=(9,10)) 
        ax8  = fig8.add_subplot(111)       

        fig9 = plt.figure(figsize=(9,10)) 
        ax9  = fig9.add_subplot(111)       

        fig10 = plt.figure(figsize=(9,10)) 
        ax10  = fig10.add_subplot(111)       

        fig11 = plt.figure(figsize=(9,10)) 
        ax11  = fig11.add_subplot(111)       

        i = 0
        tiempo = w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = w9 = w10 = w11 = np.array([])

        for t, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, aux9, aux10, aux11  in zip(self.t, z1, z2, z3, z4, z5, z6, z7, z8[::-1], z9[::-1], z10[::-1], z11[::-1]):
            #Resolucion de las graficas
            if i % 200 == 0:
               tiempo = np.append(tiempo, np.exp(t))  # Scale factor from e-folding N
               w1  = np.append(w1, aux1)
               w2  = np.append(w2, aux2)
               w3  = np.append(w3, aux3)
               w4  = np.append(w4, aux4)
               w5  = np.append(w5, aux5)
               w6  = np.append(w6, aux6)
               w7  = np.append(w7, aux7)
               w8  = np.append(w8, aux8)
               w9  = np.append(w9, aux9)
               w10 = np.append(w10, aux10)
               w11 = np.append(w11, aux11)
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
        ax3.semilogx(tiempo, w3**2, 'blue', label=r"$\Omega_{\gamma}$")     #radiation
        ax3.semilogx(tiempo, w4**2, 'orange', label=r"$\Omega_{v}$")        #neutrinos
        ax3.semilogx(tiempo, w5**2, 'red', label=r"$\Omega_b$")             #baryons
        ax3.semilogx(tiempo, w6**2, 'green', label=r"$\Omega_{\Lambda}$")   #Lambda
        ax3.set_ylabel(r'$\Omega(a)$', fontsize=20) #original
        ax3.set_xlabel(r'$a$', fontsize=15)
        ax3.legend(loc = 'best', fontsize = 'xx-large')
        fig3.savefig('Omegas.pdf')
        #plt.show()        

        # Spurious Variable
        ax8.semilogx(tiempo, w7, 'black', label=r"$y1$")
        ax8.set_ylabel(r'$y1$', fontsize=20)
        ax8.set_xlabel(r'$a$', fontsize=15)
        ax8.legend(loc = 'best', fontsize = 'xx-large')
        fig8.savefig('y1.pdf')
        #plt.show()

        # Friedmann Restriction
        ax9.semilogx(tiempo, w2 + w3**2 + w4**2 + w5**2 + w6**2, 'black', label=r"$F$")
        ax9.set_ylabel(r'$F(a)$', fontsize=20)
        ax9.set_xlabel(r'$a$', fontsize=15)
        ax9.legend(loc = 'best', fontsize = 'xx-large')
        fig9.savefig('F.pdf')
        #plt.show()

        # Angle Difference 
        ax10.plot(np.log10(tiempo), w10, 'black', label=r"$\tilde{\vartheta}$")
        ax10.set_ylabel(r'$\tilde{\vartheta}$', fontsize=20)
        ax10.set_xlabel(r'$Log(a)$', fontsize=15)
        ax10.legend(loc = 'best', fontsize = 'xx-large')
        fig10.savefig('varth.pdf')
        #plt.show()

        # Perturbation Amplitude 
        ax11.plot(np.log10(tiempo), w11, 'black', label=r"$\alpha$")
        ax11.set_ylabel(r'$\alpha$', fontsize=20)
        ax11.set_xlabel(r'$Log(a)$', fontsize=15)
        ax11.legend(loc = 'best', fontsize = 'xx-large')
        fig11.savefig('alpha.pdf')
        #plt.show()

# Runs only if the script is self contained
if __name__ == '__main__':
    start_time = time.time()

    DM = DarkM()
    DM.plot()

    # Prints total evaluation time with 3 decimals
    print("\nEvaluation time = ", "{:.3f}".format(time.time() - start_time)," seconds")
