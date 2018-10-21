from galaxies import *
from dm_profiles import *
from axion_decay import *
from cosmology import *
from scipy.optimize import fsolve

class Cross_Corr(object):

    def __init__(self, windows, meanI, FT, bias, fname, zmax=10, Mmin=1e1, Mmax=1e16):
        self.windows = windows
        self.meanI = meanI
        self.FT = FT
        self.bias = bias

        self.Mmin = Mmin
        self.Mmax = Mmax
        
        self.fname = fname
        
        self.ell_list = np.logspace(0., 4, 10)
        self.redshifts = np.linspace(0., zmax, 30)
        self.chi_list = np.logspace(-1, 5, 30)
        self.haloF = Halo_Functions(Mmin, Mmax, type_HMF='ST')
    
    

    def normalize(self):
        meanI_z1 = self.meanI[0](self.redshifts[-1])
        meanI_z2 = self.meanI[1](self.redshifts[-1])
        return 1. / (meanI_z1 * meanI_z2)

    def power_spectrum(self, z, ell, dn_dm_tab):
        integrand_1h = np.zeros_like(dn_dm_tab[:,1])
        integrand_2h_1 = np.zeros_like(dn_dm_tab[:,1])
        integrand_2h_2 = np.zeros_like(dn_dm_tab[:,1])
        ft_1 = np.zeros_like(dn_dm_tab[:,1])
        ft_2 = np.zeros_like(dn_dm_tab[:,1])
        chi = comoving_chi(z)
        k = ell / chi
        for i, dndm in enumerate(dn_dm_tab[:,1]):
            ft_1[i] = self.FT[0](k, z, dn_dm_tab[i, 0], dn_dm_tab=dn_dm_tab)
            ft_2[i] = self.FT[1](k, z, dn_dm_tab[i, 0], dn_dm_tab=dn_dm_tab)
            integrand_1h[i] = ft_1[i] * ft_2[i] * dndm
            integrand_2h_1[i] = ft_1[i] * dndm * self.bias[0](dn_dm_tab[i, 0])
            integrand_2h_2[i] = ft_2[i] * dndm * self.bias[1](dn_dm_tab[i, 0])
        
#            print i, integrand_1h[i], integrand_2h_1[i], integrand_2h_2[i]
        ps1h = np.trapz(integrand_1h, dn_dm_tab[:,0])
        if ps1h < 0:
            ps1h = 0.
        halo_integ2h = np.trapz(integrand_2h_1, dn_dm_tab[:,0]) * np.trapz(integrand_2h_2, dn_dm_tab[:,0])
        if halo_integ2h < 0:
            halo_integ2h = 0
        return ps1h, halo_integ2h * 10.**self.haloF.PowerLin_Dim(np.log10(k))

    def comoving_integrate(self, ell):
        integrand = np.zeros_like(self.chi_list)
        for i,chi in enumerate(self.chi_list):
            findz = fsolve(lambda z: comoving_chi(z) - chi, 1)[0]
            dn_dm_tab = self.haloF.dn_dm_tabular(self.Mmin, self.Mmax, findz)
            dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
            ps = self.power_spectrum(findz, ell, dn_dm_tab)
            integrand[i] = self.windows[0](findz)*self.windows[1](findz) / chi**2. * np.sum(ps)

        cl_val = self.normalize() * np.trapz(integrand, self.chi_list)
        return ell, cl_val

    def compute_angularPS(self, cl):
        cls = np.zeros_like(self.ell_list)
        for i, cl in enumerate(self.ell_list):
            cls[i] = self.comoving_integrate(cl)
            print cl, cls[i]
        np.savetxt('outputs/' + self.fname, np.column_stack((self.ell_list, cls)))
        return




