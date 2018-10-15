import numpy as np
from cosmology import *
from dm_profiles import *
from galaxies import *
from scipy.integrate import quad

class Compute_Cls(object):
    
    def __init__(self, type1, type2, l_min=1, lmax=1e3, ltot=100, 
                 type_HMF='ST', m_a=1e-5, g_a=1e-10, survey='2MASS', zmax=0.1,
                 Mmin=1e6, Mmax=1e14):
        self.type1 = type1
        self.type2 = type2
        
        self.l_min = l_min
        self.l_max = l_max
        self.ltot = ltot

        self.haloF = Halo_Functions(Mmin, Mmax, type_HMF='ST')

        viable_types = ['Axion', 'Galaxy']
        if type1 not in viable_types:
            print 'Invalid input for cross correlation...\n'
            print 'Input types: ', type1, ' and type: ', type2
            exit()

        if type1 == 'Axion':
            self.class_1 = Axion_Decay(m_a, g_a)
        elif type1 == 'Galaxy':
            self.class_1 = Galaxy_Survey(survey)
        if type2 == 'Axion':
            self.class_2 = Axion_Decay(m_a, g_a)
        elif type2 == 'Galaxy':
            self.class_2 = Galaxy_Survey(survey)

        P_linear = np.loadtxt('import_files/power_spectrum.dat')
        self.PowerLin = interp1d(np.log10(P_linear[:,0]*h_little), np.log10(P_linear[:,1]*(P_linear[:,0]*h_little)**3./(2.*np.pi**2.)), kind='linear', bounds_error=False, fill_value='extrapolate') # Mpc


    def compute_1halo(self, Mmin, Mmax, z):
        dn_dm_tab = self.haloF.dn_dm_tabular(Mmin, Mmax, z)
        dn_dm = interpolate(np.log10(dn_dm_tab[:,0]), np.log10(dn_dm_tab[:,1]), kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        k_list = np.logspace(-5., 0., 30)
        ps_1h = np.zeros_like(k_list)
        for i,k in enumerate(k_list):
            ps_1h[i] = quad(lambda m: 10.**dn_dm(np.log10(m))*self.class_1.FT_gfunc(k, z, m) *
                            self.class_2.FT_gfunc(k, z, m), Mmin, Mmax, epsrel=1e-4, limit=50)[0]
        
        return np.column_stack((k_list, ps_1))

    def compute_2halo(self, Mmin, Mmax, z):
        dn_dm_tab = self.haloF.dn_dm_tabular(Mmin, Mmax, z)
        dn_dm = interpolate(np.log10(dn_dm_tab[:,0]), np.log10(dn_dm_tab[:,1]), kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        k_list = np.logspace(-5., 0., 30)
        ps_1 = np.zeros_like(k_list)
        ps_2 = np.zeros_like(k_list)
        for i,k in enumerate(k_list):
            ps_1[i] = quad(lambda m: 10.**dn_dm(np.log10(m))*self.class_1.FT_gfunc(k, z, m) *
                            self.class_2.bias(m), Mmin, Mmax, epsrel=1e-4, limit=50)[0]
            ps_2[i] = quad(lambda m: 10.**dn_dm(np.log10(m))*self.class_2.FT_gfunc(k, z, m) *
                            self.class_1.bias(m), Mmin, Mmax, epsrel=1e-4, limit=50)[0]

        return np.column_stack((k_list, ps_1*ps_2*10.**self.PowerLin(np.log10(k_list))))


    def CLs(self):
        ell_tab = np.logspace(np.log10(self.l_min), np.log10(self.l_max), self.ltot)
        norm = 1. / (self.class_1.mean_I(zmax)*self.class_2.mean_I(zmax))

        return
