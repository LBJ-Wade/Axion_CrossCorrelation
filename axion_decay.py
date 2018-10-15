import numpy as np
from cosmology import *
from dm_profiles import *
from scipy.integrate import quad
from scipy.special import sici
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

hbar = 6.582e-16 # eV * s

class Axion_Decay(object):
    def __init__(self, m_a, g_a, Mmin, Mmax, stim_e=1., synch=1.):
        self.m_a = m_a
        self.g_a = g_a
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.haloF = Halo_Functions(Mmin, Mmax, type_HMF='ST')
        
        self.stim_e = stim_e
        self.synch = synch
        return
    
    def bias(self, m):
        return 1.

    def window(self, z):
        tau = optical_depth(self.m_a / 2., z, compute=False)
        prefactor = 1. / (256. * np.pi**2. * nu0)
        zterm = self.m_a**3 * self.g_a**2. / 1e18 / (hubble(z) * (1. + z))  # eV * Mpc
        units = 1.728e32 # 1 / cm^2 / s
        return  prefactor * zterm  / (6.58e-16) * Omega_DM * critical_rho(0) * units # 1 / cm^2 / s

    def gamma_phasespace(self, Mhalo, r, z):
        haloC = NFW(Mhalo, z)
        cmb_term = 2. / (np.exp(self.m_a / (2.*T0_CMB*(1.+z)*kbolt)) - 1.)
        nu = self.m_a / (4. * np.pi) / hbar / 1e9 # No z dependence! This is at production.
        synch_term = self.beta_coeff(z) * self.SFR(Mhalo, z)**1.77 / Mhalo**0.77 / (1. + r / (0.015 * haloC.r200))**2.
        return cmb_term * self.stim_e, synch_term * self.stim_e * self.synch
    
    def beta_coeff(self, z):
        nu =  self.m_a / (4. * np.pi) / hbar / 1e9
        z_term = 2. * 2e3 * nu**-3.7 * (1.+z)**0.85 * (hubble(z) / H0)**0.57
        mass_term = Mass_MW ** 0.77 / self.SFR(Mass_MW, 0.)**1.77
        return z_term * mass_term * self.stim_e * self.synch

    def M_star(self, Mhalo):
        # computes stellar mass of Mhalo at z
        prefac = 0.05
        Mass_c = 1.6e11
        return prefac * Mhalo / ((Mhalo / Mass_c)**(-1.) + (Mhalo / Mass_c)**(0.3))
    
    def M_accret(self, M_0, z):
        # Given halo of mass M0 at z=0, what is mass pre accretion at z
        def M13(zz):
            return 10.**13.276 * (1. + zz)**3. * (1. + zz/2.)**(-6.11)*np.exp(-5.03*zz)
        def a0(M0):
            return 0.205 - np.log10(1. + (10.**9.649 / M0) ** 0.18)
        def Gma(zz, M0):
            return 1. + np.exp(-4.651 * ( 1./(1.+zz) - a0(M0)))
        f = np.log10(M_0 / M13(0.)) * Gma(0., M_0) / Gma(z, M_0)
        return M13(z) * 10.**f

    def check_SFR(self, z):
        dn_dm_tab = self.haloF.dn_dm_tabular(self.Mmin, self.Mmax, z)
        dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
        dn_dm = interp1d(np.log10(dn_dm_tab[:,0]), np.log10(dn_dm_tab[:,1]), kind='linear',
                            bounds_error=False, fill_value=-100)
        term_int = quad(lambda m: 10.**dn_dm(np.log10(m)) * self.SFR(m, z), self.Mmin, self.Mmax, epsrel=1e-4)
        return term_int[0]
    
    def SFR(self, Mhalo, z):
        # Using 1207.6105
        Mstar = self.M_star(Mhalo)
        zdepend =  1. / (10.**(-0.997 * (z - 1.243)) + 10.**(0.241 * (z - 1.243)))
        return zdepend * Mstar

    def mean_g(self, z):
        term0 = Omega_DM * critical_rho(0) # DM density only term
        dn_dm_tab = self.haloF.dn_dm_tabular(self.Mmin, self.Mmax, z)
        dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
        integrand = np.zeros_like(dn_dm_tab[:,0])
        for i in range(len(dn_dm_tab[:,0])):
            integrand[i] = dn_dm_tab[i, 1] * self.SFR(dn_dm_tab[i,0], z)**1.77 / dn_dm_tab[i,0]**0.77 * \
                         NFW(dn_dm_tab[i,0], z).b_field_integral() * self.beta_coeff(z)
        term_int = np.trapz(integrand, dn_dm_tab[:,0])
        
        return ((1. + self.gamma_phasespace(1., 1., z)[0]) + term_int / term0)
        
    def diff_mean_g(self, z, M):
        term0 = Omega_DM * critical_rho(0) # DM density only term
        term1 = self.SFR(M, z)**1.77 / M**0.77 * \
                NFW(M, z).b_field_integral() * self.beta_coeff(z)
        return ((1. + self.gamma_phasespace(1., 1., z)[0]) + term1 / term0)

    def g_function(self, r, M, z):
        term0 = Omega_DM * critical_rho(0) # DM density only term
        dm_prof = NFW(M, z)
        return dm_prof.density(r) * (1. + 2. * np.sum(self.gamma_phasespace(Mhalo, r, z))) / term0

    def FT_gfunc(self, k, z, M):
        norm = Omega_DM * critical_rho(0) # DM density only term
        DM_p = NFW(M, z)
        meanG = self.mean_g(z)
        term0 = DM_p.FT_density(k)
        term1 = DM_p.FT_density_Bterm(k) * self.beta_coeff(z) * self.SFR(M, z)**1.77 / \
                M**0.77 * self.stim_e * self.synch
        FT_prof = (term0 * (1. + self.gamma_phasespace(1., 1., z)[0]) + term1) / norm
        
        return FT_prof / meanG

    def mean_window(self, zmax):
        z_list = np.linspace(0, zmax, 200)
        window = np.zeros_like(z_list)
        for i,z in enumerate(z_list):
            window[i] = self.window(z) * self.mean_g(z)
        norm = np.trapz(window, z_list)
        return np.column_stack((z_list, window / norm))
        
    def averageI(self, zmax):
        z_list = np.linspace(0, zmax, 200)
        window = np.zeros_like(z_list)
        for i,z in enumerate(z_list):
            window[i] = self.window(z) * self.mean_g(z)
        return np.trapz(window, z_list)

    def autoPS(self, k, z):
        dn_dm_tab = self.haloF.dn_dm_tabular(self.Mmin, self.Mmax, z)
        dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
        integrand_1h = np.zeros_like(dn_dm_tab[:,1])
        integrand_2h = np.zeros_like(dn_dm_tab[:,1], dtype=complex)
        ft = np.zeros_like(dn_dm_tab[:,1], dtype=complex)
        ft2 = np.zeros_like(dn_dm_tab[:,1], dtype=complex)
        for i, dndm in enumerate(dn_dm_tab[:,1]):
            ft[i] = self.FT_gfunc(k, z, dn_dm_tab[i, 0])
#            print ft[i], np.abs(np.real(ft[i]) * dndm)
            integrand_1h[i] = np.real(ft[i] * np.conj(ft[i])) * dndm
            integrand_2h[i] = ft[i] * dndm
        ps1h = np.trapz(integrand_1h, dn_dm_tab[:,0])
        halo_integ2h = np.trapz(integrand_2h, dn_dm_tab[:,0])
        ps2h = np.real(halo_integ2h * np.conj(halo_integ2h)) * 10.**self.haloF.PowerLin_Dim(np.log10(k))
        return ps1h, ps2h


