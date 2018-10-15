import numpy as np
from scipy.integrate import quad

Omega_DM = 0.266
Omega_B = 0.0492
Omega_M = Omega_DM + Omega_B
Omega_R = 5.43e-5
Omega_L = 1. - Omega_M - Omega_R

h_little = 0.678
H0 = h_little * 1e2 / 2.998e5 #Mpc^-1

T0_CMB = 2.778 # K
kbolt = 8.617e-5 # eV * K^-1

Mass_MW = 1.5e12 # SM

sigma_T = 6.65e-25
n_e_0 = 2.503e-7 * (0.245 + (1. - 0.245)*2.)

nu0 = 1.4204e9 # 21cm line, Hz

clight = 2.998e8

def hubble(z):
    return H0 * np.sqrt(Omega_L + Omega_M * (1+z)**3. + Omega_R * (1+z)**4.)


def optical_depth(E, z, compute=False):
    if not compute:
        return 0.
    prefac = sigma_T * n_e_0 / H0
    int_fac = quad(lambda a: 1./a**4./np.sqrt(Omega_R*a**-4+Omega_M**-3.+Omega_L), 1./(1.+z) ,1.)
    return prefac*int_fac[0]

def critical_rho(z):
    return (hubble(z) / hubble(0.))**2. * 2.7753e11 * h_little**2. # M_odot / Mpc^3

def comoving_chi(z_emit):
    chi = quad(lambda z: 1./H0, 0., z_emit)
    return chi[0]

def cosmic_t_to_z(tval):
    #arxiv 0506079, t in Gyr
    return np.sqrt(28. / tval - 1.) - 1.

def z_to_cosmic_t(z):
    return 28. / (1. + (1. + z)**2.)
