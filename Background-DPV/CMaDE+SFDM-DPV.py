'''

Compton Mass Dark Energy & Scalar Field Dark Matter.

Combination of the SFDM and CMaDE model equations. 
Here, the density parameters of the components of the universe are calculated solving the dynamical equations using the BDF method.
In this code, the parameter densities are used as variables of the system.

Note: The solution is not right. The solver finds complex values when OmegaV > OmegaDM.

'''

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# System Constants
OmegaB0  = 0.044
OmegaDM0 = 0.27
OmegaR0  = 9.539e-5
OmegaK0  = 0.001
H0    	 = 1.0 
OmegaV0  = OmegaDM0
s0 		 = 6.65e9 

# Interaction Constants
Q  = 1.0
kc = 1.0

# Initial DE Density Parameter
OmegaM0 = 1.0 - OmegaB0 - OmegaK0 - OmegaDM0 - OmegaR0

# Initial Conditions
ICCMaDE = [OmegaB0, OmegaR0, OmegaDM0, OmegaM0, H0, OmegaV0]
ICLCDM  = [OmegaB0, OmegaR0, OmegaDM0, H0]

# e-folding Span
NMin 	  = 0.0
NMax 	  = -17.0
NInterval = (NMin, NMax)
NPoints   = 1000
NArray 	  = np.linspace(NMin, NMax, num = NPoints)

# System of Equations CMaDE+SFDM
def SoECMaDE(N, Omega):

	# Differential Equations
	dOmegaB  = -3* Omega[0]
	dOmegaR  = -4* Omega[1]
	dOmegaM  = Q* (np.sqrt(6)/np.pi)* (np.exp(-N)/Omega[4])* Omega[3]**(3/2)
	dOmegaDM = -kc* dOmegaM - 6* (Omega[2] - Omega[5])
	dOmegaV  = 2* s0* np.sqrt(abs(Omega[5]* (Omega[2] - Omega[5])))/ Omega[4]

	# Friedmann Constraint
	dHdt 	= (-2* OmegaK0* np.exp(-2* N) + dOmegaR + dOmegaM + dOmegaB + dOmegaDM)/ (2* Omega[4])

	return [dOmegaB, dOmegaR, dOmegaDM, dOmegaM, dHdt, dOmegaV]

# Solution of the DAE System
SolutionCMaDE = solve_ivp(SoECMaDE, NInterval, ICCMaDE, method = 'BDF', t_eval = NArray)

# System of Equations LCDM
def SoELCDM(N, Omega):

	# Differential Equations
	dOmegaB  = -3* Omega[0]
	dOmegaR  = -4* Omega[1]	
	dOmegaDM = -3* Omega[2]

	# Friedmann Constraint
	dHdt 	 = (-2* OmegaK0* np.exp(- 2* N) + dOmegaR + dOmegaB + dOmegaDM)/ (2* Omega[3])

	return [dOmegaB, dOmegaR, dOmegaDM, dHdt]

# Solution of the DAE System
SolutionLCDM = solve_ivp(SoELCDM, NInterval, ICLCDM, method = 'BDF', t_eval = NArray)

#Convert e-folding into Redshift and Scale Factor
zArray = np.exp(-SolutionCMaDE.t) - 1
aArray = np.exp(SolutionCMaDE.t)

# Equation of State
def weff(N, H, OmegaM):
	return -Q* (2/3)**(1/2)* (np.exp(-N)/np.pi)* np.sqrt(OmegaM)/H - 1 

# Models Comparison
def HubbleComparison(HCMaDE, HLCDM):
	return 1 - (HCMaDE/HLCDM)

# Results Array
weffArray 			  = weff(SolutionCMaDE.t, SolutionCMaDE.y[4], SolutionCMaDE.y[3])
HubbleComparisonArray = HubbleComparison(SolutionCMaDE.y[4], SolutionLCDM.y[3])

# Plot the EoS
plt.plot(zArray, weffArray)
plt.xscale('log')
plt.xlabel('z')
plt.ylabel(r'$\omega(z)$')
plt.title('Equation of State')
plt.legend()
plt.show()

# Plot the Hubble Parameter
plt.plot(zArray, HubbleComparisonArray)
plt.xscale('log')
plt.xlabel('z')
plt.ylabel(r'1-$\mathcal{H}_{CMaDE}/\mathcal{H}_{LCDM}$')
plt.title('Hubble Parameter')
plt.legend()
plt.show()

# Plot the Hubble Parameter
plt.plot(aArray, SolutionCMaDE.y[5]/SolutionCMaDE.y[4])
plt.xscale('log')
plt.xlabel('a')
plt.ylabel(r'OmegaV')
plt.title('Potential')
plt.legend()
plt.show()