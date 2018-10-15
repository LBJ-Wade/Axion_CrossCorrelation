import numpy as np
from cosmology import *
from scipy.special import erf, sici
from mpmath import ei
from scipy.interpolate import interp1d
from scipy.integrate import quad

class NFW(object):
    
    def __init__(self, m200, z):
        self.m200 = m200
        self.redshift = z

        delta_overdense = 18.*np.pi**2. + 82.*(Omega_M*(1.+z)**3.*(H0/hubble(z))**2. - 1.) -\
                          39.*(Omega_M*(1.+z)**3.*(H0/hubble(z))**2. - 1.)**2.
        rho_crit = (hubble(z) / hubble(0.))**2. * 2.7753e11 * h_little**2.  # M_odot / Mpc^3
        self.r200 = ((3.*m200) / (4.*np.pi*rho_crit*delta_overdense))**(1./3.)
     
        self.r_scale = self.r200 / self.concentration()
        
        self.rho_s = self.m200 / (4.*np.pi*self.r_scale**3.) / (-1. + self.r_scale / 
                     (self.r_scale + self.r200) + np.log((self.r_scale + self.r200) / self.r_scale))

        
    def concentration(self):
        if self.m200 > 1e10:
            az = 0.029 * self.redshift - 0.097
            bz = -110.001 / (self.redshift + 16.885) + 2469.720/(self.redshift + 16.885)**2.
            logc = az * np.log10(h_little * self.m200) + bz
            self.c = 10.**logc
            return 10.**logc
        else:
            # taken from 1312.1729 and 9908159
            if self.redshift > 5:
                scalingF = (1. + 5.)
            else:
                scalingF = (1. + self.redshift)
            clist = [37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7]
            lnfact = np.log(self.m200 * h_little)
            c200 = 0
            for i,c in enumerate(clist):
                c200 += c * lnfact**i
            return c200 / scalingF

    def density(self, r):
        return self.rho_s / (r / self.r_scale) / (1. + r / self.r_scale)**2.

    def mass_in_R(self, R):
        massR = 4.*np.pi*self.r_scale**3.*self.rho_s * (-1. + self.r_scale / 
                (self.r_scale + R) + np.log((self.r_scale + R) / self.r_scale))
        return massR

    def b_field_integral(self):
        rs = self.r_scale
        rst = 0.015 * self.r200
        
        prefactor = 4.*np.pi*self.rho_s*rs**3.*rst**2. / (rs - rst)**3.
        term1 = self.r200 * (rs - rst)*(2.*self.r200 + rs + rst) / (self.r200 + rs) / (self.r200 + rst)
        term2 = (rs + rst)*(np.log(rs/rst) + np.log((self.r200 + rst) / (self.r200 - rs)))
        return prefactor * (term1 + term2)
#        prefactor = 4.*np.pi*self.r_scale**3.*self.rho_s
#        term1 = -1. + self.r_scale / (np.exp(3./200.)*(self.r_scale * self.r200))
#        term2 = - np.exp(3.*self.r_scale/(200.*self.r200)) * (3.*self.r_scale + 200. * self.r200) * ei(-3.*self.r_scale/(200.*self.r200)) / (200.*self.r200)
#        term3 = np.exp(3.*self.r_scale/(200.*self.r200)) * (3.*self.r_scale + 200. * self.r200) * ei(-3.*(self.r_scale + self.r200)/(200.*self.r200)) / (200.*self.r200)
#        return prefactor*(term1+term2+term3)

    def FT_density(self, k):
        prefactor = 4.*np.pi*self.rho_s*self.r_scale**3
        si_1, ci_1 = sici(k * self.r_scale)
        si_2, ci_2 = sici(k * (self.r_scale + self.r200))
        bb = k * self.r_scale
        Xm = self.r200 / self.r_scale
        term1 = np.cos(bb) * (-ci_1 + ci_2)
        term2 = np.sin(bb) * (-si_1 + si_2)
        term3 = np.sin(bb * Xm) / ((1. + Xm)*bb)
        return prefactor * (term1 + term2 + term3)

    def FT_density_Bterm(self, k):

        return 4. * np.pi * self.rho_s / k * self.B_FT_integral(k)

    def B_FT_integral(self, k):
        rst = 0.015 * self.r200
        rs = self.r_scale
        
        si_1, ci_1 = sici(k * rst)
        si_2, ci_2 = sici(k * (rst + self.r200))
        si_3, ci_3 = sici(k * rs)
        si_4, ci_4 = sici(k * (rs + self.r200))
        
        prefac = rs**3. * rst**2. / (rs - rst)**3.
        term1 = k*(-rs + rst)*np.cos(k*rst)*ci_1
        term2 = k*(rs - rst)*np.cos(k*rst)*ci_2
        term3 = - (rs - rst)*(2*self.r200 +rs +rst)*np.sin(k*self.r200) / (self.r200+rs) / (self.r200 + rst)
        term4 = ci_4*(k*(rs-rst)*np.cos(k*rs)-2*np.sin(k*rs))
        term5 = ci_3*(k*(rst - rs)*np.cos(k*rs) + 2*np.sin(k*rs))
        term6 = -2.*ci_1*np.sin(k*rst) + 2.*ci_2*np.sin(k*rst)
        term7 = - k*(rs-rst)*np.sin(k*rs)*si_3
        term8 = k*(rs-rst)*np.sin(k*rs)*si_4
        term9 = 2.*np.cos(k*rs)*(-si_3 + si_4) + 2.*np.cos(k*rst)*si_1 - k*(rs-rst)*np.sin(k*rst)*si_1
        term10 = -2*np.cos(k*rst)*si_2+k*(rs-rst)*np.sin(k*rst)*si_2
        return prefac * (term1+term2+term3+term4+term5+term6+term7+term8+term9+term10)
    

class Halo_Functions(object):

    def __init__(self, Mmin, Mmax, type_HMF='ST', load_sigma=True):
        self.type_HMF = type_HMF
        P_linear = np.loadtxt('import_files/explanatory05_pk.dat')
        self.PowerLin_Dim = interp1d(np.log10(P_linear[:,0]*h_little), np.log10(P_linear[:,1]),
                            kind='cubic', bounds_error=False, fill_value='extrapolate') # Mpc
       
        self.delta_c = 1.686
        self.Mmin = Mmin
        self.Mmax = Mmax
        if load_sigma:
            sigFile = np.loadtxt('import_files/sigma_M_z0.dat')
            self.sigma_M = lambda x,z: 10.**interp1d(np.log10(sigFile[:,0]), np.log10(sigFile[:,1]), kind='cubic', fill_value=0., bounds_error=False)(np.log10(x))*(self.growth_factor(z)/self.growth_factor(0.))
        else:
            pass
            
        return

    def N_central(self, M):
        sigma_logM = 0.15
        Log_Mmin = 11.68
        return .5 * (1. + erf((np.log10(M) - Log_Mmin) / sigma_logM))

    def N_sat(self, M):
        if np.shape(M) == ():
            M = np.array([M])
        ret_array = np.zeros_like(M)
        for i,M in enumerate(M):
            Log_M0 = 11.86
            if np.log10(M) < Log_M0:
                ret_array[i] = 0
                continue
    
            Log_M1 = 13.
            alpha = 1.02
            ret_array[i] = ((M - 10.**Log_M0) / 10.**Log_M1)**alpha
        return ret_array

    def window_tophat(self, k, M):
        R = (3.*M / (4.*np.pi*Omega_M*critical_rho(0.)))**(1./3.)
        val = 3./(k*R)**3. * (np.sin(k*R) - k*R*np.cos(k*R))
        return val


    def sigma_density(self, M, z):
        prefactor = self.growth_factor(z)**2.*3.
        integ_term = quad(lambda k: k**2.*10.**self.PowerLin_Dim(np.log10(k))*self.window_tophat(k, M)**2. / (2.*np.pi**2.),
                            0., np.inf, epsrel=1e-6, limit=200)
        if integ_term[0] < 0.:
            print 'Negative sigma density??'
            print M, z, integ_term[0]
            kcheck = np.logspace(-4, 10, 10)
            for k in kcheck:
                print k, 10.**self.PowerLin_Dim(np.log10(k)), self.window_tophat(k, M)
            exit()
        return np.sqrt(integ_term[0] * prefactor)


    def growth_factor(self, z):
        a = 1. / (1.+z)
        prefactor = 5. * Omega_M / 2. * hubble(z) / H0

        integ_term = quad(lambda apr: 1./(apr * hubble(1./apr - 1.) / H0)**3., 0., a, epsrel=1e-4, limit=100)[0]
        norm = 5. * Omega_M / 2. * quad(lambda apr: 1./(apr * hubble(1./apr - 1.) / H0)**3., 0., 1., epsrel=1e-4, limit=100)[0]
        return prefactor * integ_term / norm

    def f_coll(self, sigma):
        a = 0.707
        p = 0.3
        nu = (self.delta_c / sigma)**2.
        fst = np.sqrt(a*nu/(2.*np.pi))*np.exp(-a*nu/2.)*(1. + (a*nu)**-p) * 0.3222
        return fst

    def dn_dm_tabular(self, Mmin, Mmax, z, save=False):
        mtab = np.logspace(np.log10(Mmin), np.log10(Mmax), 100)
        fcol_tab = np.zeros_like(mtab)
        sigma_m  = np.zeros_like(mtab)
        rhoM = Omega_M * critical_rho(0.) # SM / Mpc^3
        
        for i,m in enumerate(mtab):
            sigma_m[i] = self.sigma_M(m, z)
            fcol_tab[i]  = rhoM / m**2. * self.f_coll(sigma_m[i])
        if save:
            np.savetxt('outputs/fcoll_z_{:.2f}.dat'.format(z), np.column_stack(((self.delta_c / sigma_m)**2., fcol_tab)))
        
        dnDM = self.delta_c * fcol_tab[1:] * np.abs(np.diff(np.log(sigma_m)) / np.diff(np.log(mtab)))
        return np.column_stack((mtab[1:], dnDM))

    def avg_n_gal(self, z, dn_dm_tab=None):
        try:
            dn_dm = interp1d(np.log10(dn_dm_tab[:,0]), np.log10(dn_dm_tab[:,1]), kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        except:
            dn_dm_tab = self.dn_dm_tabular(self.Mmin, self.Mmax, z)
            dn_dm = interp1d(np.log10(dn_dm_tab[:,0]), np.log10(dn_dm_tab[:,1]), kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        integ_val = quad(lambda M: 10.**dn_dm(np.log10(M))*(self.N_central(M) + self.N_sat(M)), self.Mmin, self.Mmax, epsrel=1e-4, limit=50)
        return integ_val[0]


