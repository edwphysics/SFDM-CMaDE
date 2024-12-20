'''

Compton Mass Dark Energy & Scalar Field Dark Matter: Perturbations.

Combination of the SFDM and CMaDE model equations. Perturbation equations with T<<Tc.
Here, the density parameters of the components of the universe are calculated solving the dynamical equations using the ABMM4 method.

Comments: Solve the background equations backwards in time. Then, it uses the result to solve the perturbation equations forward in time.

Modified from the original code by Luis Osvaldo Tellez Tovar for the paper The quantum character of the Scalar Field Dark Matter.

'''

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

class DarkM:
    def __init__(self, mquin=1, name='LO'):

        self.name   = name
        self.cte    = 3./2.0
        #self.k2 = 0.1                  # Tamaño de la fluctuación que se estudia k^2/m^2
        #self.Lambda = 0.0              # Constante de autointeraccion de la materia oscura
        self.masa   = 1.0e-23
        self.H0     = 1.0e-33           # Valor del parametro de hubble en eV, viene de H=70
        self.s      = self.masa/self.H0
        self.k      = 2.009e-29         # Wavenumber Initial Value
        self.km     = self.k/self.masa

        # Rango del factor de escala
        self.NP = 100000
        self.Ni = np.log(1.0e-0)
        self.Nf = np.log(1.0e-6)
        self.d  = (self.Nf - self.Ni)/ self.NP
        self.t  = [self.Ni + self.d* i for i in np.arange(self.NP)]

    def something(self):
        print(self.name)

    # Runge-Kutta 4 para inicializar el metodo ABM4.
    def rk4(self, func, y_0, t, z, args={}):
        # Inicia el arreglo de las aproximaciones
        y = np.zeros([4, len(y_0)])
        y[0] = y_0
        for i, t_i in enumerate(t[:3]): #Revisar el contenido del enumerate

            h = self.d #t[i+1] - t_i
            k_1 = func(t_i, y[i], z[i], args)
            k_2 = func(t_i+h/2., y[i]+h/2.*k_1, z[i], args)
            k_3 = func(t_i+h/2., y[i]+h/2.*k_2, z[i], args)
            k_4 = func(t_i+h, y[i]+h*k_3, z[i], args)

            y[i+1] = y[i] + h/6.*(k_1 + 2.*k_2 + 2.*k_3 + k_4) # RK4 step

        return y

    # Adams-Bashforth 4/Moulton 4 Step Predictor/Corrector
    def ABM4(self, func, y_0, t, z, args={}):
        y = np.zeros([len(t), len(y_0)])
        #Se calcularan los primeros pasos con rk4
        y[0:4] = self.rk4(func,y_0, t, z)
        k_1 = func(t[2], y[2], z[2], args)
        k_2 = func(t[1], y[1], z[1], args)
        k_3 = func(t[0], y[0], z[0], args)
        for i in range(3,self.NP-1):
            h = self.d
            k_4 = k_3
            k_3 = k_2
            k_2 = k_1
            k_1 = func(t[i], y[i], z[i], args)
            #Adams Bashforth predictor
            y[i+1] = y[i] + h*(55.*k_1 - 59.*k_2 + 37.*k_3 - 9.*k_4)/24.
            k_0 = func(t[i+1],y[i+1],z[i+1],args)
            #Adams Moulton corrector
            y[i+1] = y[i] + h*(9.*k_0 + 19.*k_1 - 5.*k_2 + k_3)/24.
        return y

    # Background: Condiciones iniciales para las variables, valores que se tienen al dia de hoy. 
    def solverBG(self):
        y0 = np.array([1.0e-5,                # Small initial kinetic energy to avoid 1/x divergence
                       np.sqrt(0.22994),      # u Initial Condition
                       np.sqrt(0.00004),      # z Initial Condition
                       np.sqrt(0.00002),      # nu Initial Condition
                       np.sqrt(0.04),         # b Initial Condition
                       np.sqrt(0.73),         # l Initial Condition
                       1.0e3])                 # s Initial Condition               

        y_result = self.ABM4(self.RHSBG, y0, self.t, np.zeros(self.NP))
        #y_result = self.rk4(self.RHS, y0, self.t)
        return y_result

    # Perturbations: Condiciones iniciales para las variables en a ~ 1e-6. 
    def solverPB(self):
        y0 = np.array([1.0e-30,                # l1 Initial Condition
                       1.0e-30,                # l2 Initial Condition
                       1.0e-30,                # z1 Initial Condition
                       1.0e-30])               # z2 Initial Condition   

        y_result = self.ABM4(self.RHSPB, y0, self.t[::-1], self.solverBG()[::-1])
        #y_result = self.rk4(self.RHS, y0, self.t)
        return y_result

    '''

    En la siguiente funcion se introduce el lado derecho de las ecuaciones diferenciales que se quieren resolver.
    En el arreglo y se ponen las variables que se quieren hallar. La forma en la que se acomodan es la siguiente.
    x0 -> Parte cinetica del campo escalar.                        x en el pdf
    x2 -> Parte potencial del campo escalar.                       u en el pdf
    x4 -> Radiacion.                                               z en el pdf
    x5 -> Neutrinos.                                               nu en el pdf
    x6 -> Bariones.                                                b en el pdf
    x7 -> Constante cosmologica.                                   l en el pdf
    x8 -> Variable espuria.                                        s en el pdf
    x9 -> Gravitational Potential.                                 l1 en el pdf
    x10 -> Graitational Potential Derivative.                      l2 en el pdf
    x11 -> Field Perturbations.                                    z1 en el pdf
    x12 -> Field Perturbations Derivative.                         z2 en el pdf


    '''

    # Background
    def RHSBG(self, t, y, z, args={}):
        x0, x2, x4, x5, x6, x7, x8 = y
        CTer = 4.0/3.0
        Ter  = 1.0/3.0
        k2   = 1e-4
        Pe   = 2.0* x0*x0 + 4.0* x4*x4/3.0 + 4.0* x5*x5/3.0 + x6*x6
        kN   = 2.0
        kc   = 1.0
        Q    = 1.0

        #wef = x0*x0 + x1*x1 - x3*x3 + x4*x4/3.0 - x6*x6
        #Pe = 1 + wef
        return np.array([-3.0* x0 - x8*x2 + 1.5* x0*Pe - (Q* kc/np.pi)* np.sqrt(3/2)* (x7**3/x0)* np.exp(-t),
                         x0*x8 + 1.5*x2*Pe,
                         1.5* x4* (Pe - CTer),
                         1.5* x5* (Pe - CTer),
                         1.5* x6* (Pe - 1.0),
                         1.5* x7* Pe + (Q/np.pi)* np.sqrt(3/2.)* x7**2* np.exp(-t),
                         -1.5*x8**(-kN)])

    # Perturbations
    def RHSPB(self, t, y, z, args={}):
        x9, x10, x11, x12 = y
        x0, x2, x4, x5, x6, x7, x8 = z
        Pe   = 2.0* x0*x0 + 4.0* x4*x4/3.0 + 4.0* x5*x5/3.0 + x6*x6
        kc   = 1.0
        Q    = 1.0

        #wef = x0*x0 + x1*x1 - x3*x3 + x4*x4/3.0 - x6*x6
        #Pe = 1 + wef
        return np.array([x10,
                         3.0* x10* (Pe/2.0 - 2.0) + x9* (3.0* Pe - 4.0) - 6.0* x11* x2* x8 - self.km**2* x8**2* x9* np.exp(-2.0* t),
                         x12,   
                         3.0* x12* (Pe/2.0 - 1.0) - x11* x8**2* (self.km**2* np.exp(-2.0* t) + 1.0) - 2.0* x2* x8* x9 + 4.0* x10* x0])

    # Abajo se tiene la funcion que va a imprimir las graficas.
    def plot(self):
        # En este arreglo se guardan los resultados de la funcion solver. Las variables se acomodan como en la funcion RHS.
        z0, z2, z4, z5, z6, z7, z8 = self.solverBG().T
        z9, z10, z11, z12 = self.solverPB().T
        #x, u, z, nu, l, s, b = self.solver().T

        fig0 = plt.figure(figsize=(9,10))
        ax0  = fig0.add_subplot(111)

        fig2 = plt.figure(figsize=(9,10))
        ax2  = fig2.add_subplot(111)

        fig3 = plt.figure(figsize=(9,10))
        ax3  = fig3.add_subplot(111)

        fig8 = plt.figure(figsize=(9,10))   # Define una nueva ventana.
        ax8  = fig8.add_subplot(111)        # La grafica correspondiente a la nueva ventana.

        fig9 = plt.figure(figsize=(9,10))   # Define una nueva ventana.
        ax9  = fig9.add_subplot(111)        # La grafica correspondiente a la nueva ventana.

        fig11 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax11  = fig11.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        fig13 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax13  = fig13.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        fig15 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax15  = fig15.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        fig16 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax16  = fig16.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        fig17 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax17  = fig17.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        fig18 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax18  = fig18.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        fig19 = plt.figure(figsize=(9,10))  # Define una nueva ventana.
        ax19  = fig19.add_subplot(111)      # La grafica correspondiente a la nueva ventana.

        i = 0
        tiempo = []
        w0 = []
        w2 = []
        w4 = []
        w5 = []
        w6 = []
        w7 = []
        w8 = []
        w9 = []
        w10 = []
        w11 = []
        w12 = []

        for t, aux0, aux2, aux4, aux5, aux6, aux7, aux8, aux9, aux10, aux11, aux12  in zip(self.t, z0, z2, z4, z5, z6, z7, z8, z9[::-1], z10[::-1], z11[::-1], z12[::-1]):
            # Resolucion de las graficas
            if (i % 200 == 0):
               tiempo.append(np.exp(t))  # e-folding N.
               w0.append(aux0)
               w2.append(aux2)
               w4.append(aux4)
               w5.append(aux5)
               w6.append(aux6)
               w7.append(aux7)
               w8.append(aux8)
               w9.append(aux9)
               w10.append(aux10)
               w11.append(aux11)
               w12.append(aux12)
            i += 1

        tiempo_np = np.array(tiempo)
            
        # Kinetic Energy
        ax0.semilogx(tiempo, np.array(w0)*np.array(w0) , 'black', label=r'$x^2$')
        ax0.set_ylabel(r'$x^2(a)$', fontsize=20) #original
        ax0.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #plt.legend()
        #ax3.legend()
        ax0.legend(loc = 'best', fontsize = 'xx-large')
        fig0.savefig('x2.pdf')
        #plt.show()       

        # Scalar Field
        ax2.semilogx(tiempo, np.sqrt(6)* np.array(w2)/np.array(w8) , 'black', label=r'$\kappa\Phi_0$')
        ax2.set_ylabel(r'$\kappa\Phi_0(a)$', fontsize=20) #original
        ax2.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #plt.legend()
        #ax3.legend()
        ax2.legend(loc = 'best', fontsize = 'xx-large')
        fig2.savefig('sf.pdf')
        #plt.show()        

        # Energy Density Evolution
        ax3.semilogx(tiempo, np.array(w0)*np.array(w0) + np.array(w2)*np.array(w2), 'black', label=r'$\Omega_{SFDM}$')
        ax3.semilogx(tiempo, np.array(w4)*np.array(w4), 'blue', label=r'$\Omega_{\gamma}$') #radiacion
        ax3.semilogx(tiempo, np.array(w5)*np.array(w5), 'orange', label=r"$\Omega_{v}$") #neutrinos
        ax3.semilogx(tiempo, np.array(w6)*np.array(w6), 'red', label=r"$\Omega_b$")  #bariones
        ax3.semilogx(tiempo, np.array(w7)*np.array(w7), 'green', label=r"$\Omega_{\Lambda}$")  #constante cosmologica
        ax3.set_ylabel(r'$\Omega(a)$', fontsize=20) #original
        ax3.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #plt.legend()
        #ax3.legend()
        ax3.legend(loc = 'best', fontsize = 'xx-large')
        fig3.savefig('sfdm_Complejo.pdf')
        #plt.show()        

        # Variable Espuria
        ax8.semilogx(tiempo, np.array(w8), 'black', label=r"$s$")
        ax8.set_ylabel(r'$s(a)$', fontsize=20)
        ax8.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax8.legend()
        ax8.legend(loc = 'best', fontsize = 'xx-large')
        fig8.savefig('s.pdf')
        #plt.show()

        # Gravitational Potential
        ax9.semilogx(tiempo, np.array(w9), 'black', label=r"$\phi$")
        ax9.set_ylabel(r'$\phi(a)$', fontsize=20)
        ax9.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax9.legend(loc = 'best', fontsize = 'xx-large')
        fig9.savefig('phi.pdf')
        #plt.show()

        # Perturbed Scalar Field
        ax11.semilogx(tiempo, np.sqrt(6)* np.array(w11), 'black', label=r"$\kappa\delta\Phi$")
        ax11.set_ylabel(r'$\kappa\delta\Phi(a)$', fontsize=20)
        ax11.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax11.legend(loc = 'best', fontsize = 'xx-large')
        fig11.savefig('dPhi.pdf')
        #plt.show()

        # SF Energy Density Distortions
        ax13.semilogx(tiempo, 2* (np.array(w0)* (np.array(w12) - np.array(w0)* np.array(w9)) + np.array(w8)* np.array(w2)* np.array(w11))/(np.array(w0)**2 + np.array(w2)**2), 'black', label=r"$\delta$")
        ax13.set_ylabel(r'$\delta(a)$', fontsize=20)
        ax13.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax13.legend(loc = 'best', fontsize = 'xx-large')
        fig13.savefig('delta.pdf')
        #plt.show()

        # Equation of State
        ax15.semilogx(tiempo, (np.array(w0)**2 - np.array(w2)**2)/(np.array(w0)**2 + np.array(w2)**2), 'black', label=r"$\omega_{\Phi}$")
        ax15.set_ylabel(r'$\omega_{\Phi}(a)$', fontsize=20)
        ax15.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax15.legend(loc = 'best', fontsize = 'xx-large')
        fig15.savefig('omega.pdf')
        #plt.show()

        # Pressure to Density Perturbations Ratio
        ax16.semilogx(tiempo, (np.array(w0)* np.array(w12) - np.array(w0)**2* np.array(w9) - np.array(w2)* np.array(w8)* np.array(w11))/(np.array(w0)* np.array(w12) - np.array(w0)**2* np.array(w9) + np.array(w2)* np.array(w8)* np.array(w11)), 'black', label=r"$\delta P/\delta\rho$")
        ax16.set_ylabel(r'$\delta P/\delta\rho$', fontsize=20)
        ax16.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax16.legend(loc = 'best', fontsize = 'xx-large')
        fig16.savefig('dPdrho.pdf')
        #plt.show()

        # Pressure Perturbation
        ax17.semilogx(tiempo, np.array(w0)* np.array(w12) - np.array(w0)**2* np.array(w9) - np.array(w2)* np.array(w8)* np.array(w11), 'black', label=r"$\delta P$")
        ax17.set_ylabel(r'$\delta P$', fontsize=20)
        ax17.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax17.legend(loc = 'best', fontsize = 'xx-large')
        fig17.savefig('dP.pdf')
        #plt.show()

        # G/H Function
        ax18.semilogx(tiempo, (2.0/3.0)* self.km**2* np.array(w8)**2* (np.array(w9) + np.array(w10))* np.exp(-2.0* tiempo_np)/(np.array(w0)**2 + np.array(w2)**2), 'black', label=r"$G_\Phi/H$")
        ax18.set_ylabel(r'$G_\Phi/H$', fontsize=20)
        ax18.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax9.legend()
        ax18.legend(loc = 'best', fontsize = 'xx-large')
        fig18.savefig('GH.pdf')
        #plt.show()

        # Friedmann Restriction
        ax19.semilogx(tiempo, np.array(w0)*np.array(w0) + np.array(w2)*np.array(w2) + np.array(w4)*np.array(w4) + np.array(w5)*np.array(w5) + np.array(w6)*np.array(w6) + np.array(w7)*np.array(w7), 'black', label=r"$F$")
        ax19.set_ylabel(r'$F(a)$', fontsize=20)
        ax19.set_xlabel(r'$a$', fontsize=15)
        #plt.ylabel('n (a)', fontsize=20) #original
        #original
        #plt.ylabel(r'$\omega (a)$', fontsize=20) #ecuacion de estado
        #plt.legend((r'$\Omega_{dm}$', '$\Omega_{\gamma}$', '$\Omega_{\Lambda}$', '$\Omega_b$' ))
        #ax8.legend()
        ax19.legend(loc = 'best', fontsize = 'xx-large')
        fig19.savefig('F.pdf')
        #plt.show()

if __name__ == '__main__':
    DM = DarkM(mquin=2.)
    print(DM.plot())