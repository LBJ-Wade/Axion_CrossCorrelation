import numpy as np
from cosmology import *
from dm_profiles import *
from scipy.integrate import quad
from scipy.special import sici
from scipy.interpolate import interp1d

class Galaxy_Survey(object):

    def __init__(self, survey, Mmin, Mmax):
        self.g_survey = survey
        self.set_ngal_dz()

        self.haloF = Halo_Functions(Mmin, Mmax, type_HMF='ST')
        self.Mmin = Mmin
        self.Mmax = Mmax
        
        return

    def bias(self, m):
        return 1.4
    
    def set_ngal_dz(self):
        if self.g_survey == '2MASS':
            self.N_gals = 43500.
            self.zmin = 0.
            self.zmax = 0.1
            self.dn_dz = lambda z: 7.9895e7 * z * np.exp(- (z/0.033)**2.)
        elif self.g_survey == '2MXSC':
            self.N_gals = 770000.
            self.zmin = 0.
            self.zmax = 0.3
            self.dn_dz = lambda z: 3.34e9 * z**1.9 * np.exp(- (z / 0.07)**1.75)
        return

    def window(self, z):
        # Function of redshift
        return self.dn_dz(z) * hubble(z)

    def mean_g(self, z):
        dn_dm_tab = self.haloF.dn_dm_tabular(self.Mmin, self.Mmax, z)
        meanGal = self.haloF.avg_n_gal(z, dn_dm_tab=dn_dm_tab)
        dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
        integ_val = np.trapz(dn_dm_tab[:,1] * dn_dm_tab[:,0] *
                            (self.haloF.N_central(dn_dm_tab[:,0]) + self.haloF.N_sat(dn_dm_tab[:,0])) / meanGal,
                            dn_dm_tab[:,0])
        return integ_val

    def FT_gfunc(self, k, z, M):
        DM_p = NFW(M, z)
        FT_prof = DM_p.FT_density(k)
        meanN = self.haloF.N_central(M) + self.haloF.N_sat(M)
        dn_dm_tab = self.haloF.dn_dm_tabular(self.Mmin, self.Mmax, z)
        dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
        meanGal = self.haloF.avg_n_gal(z, dn_dm_tab=dn_dm_tab)
        meanG = self.mean_g(z)
        return FT_prof * meanN / meanG / meanGal

    def mean_I(self, zmax):
        return quad(lambda z: self.mean_g(z)*self.window(z) / hubble(z), 0., zmax, epsrel=1e-4, limit=50)[0]

    def averageI(self, zm):
        z_M = np.min([self.zmax, zm])
        z_list = np.linspace(0, z_M, 200)
        window = np.zeros_like(z_list)
        for i,z in enumerate(z_list):
            window[i] = self.window(z) * self.mean_g(z)
        return np.trapz(window, z_list)
