import numpy as np
from cross_correlate import *
from multiprocessing import Pool
import time

Mmin = 1e1
Mmax = 1e16
zmax = 10

types = ['axion', 'galaxy']
galaxy_survey = '2MXSC'
modules = np.zeros(2, dtype=object)
windows = np.zeros(2, dtype=object)
avgI = np.zeros(2, dtype=object)
bias = np.zeros(2, dtype=object)
FT = np.zeros(2, dtype=object)
tags = np.zeros(2, dtype=str)
for i in range(len(types)):
    if types[i] == 'axion':
        mass = 1e-5
        gval = 1e-10
        stim_e = 1
        synch = 1
        modules[i] = Axion_Decay(mass, gval, Mmin, Mmax, stim_e=stim_e, synch=synch)
        windows[i] = modules[i].window_prime
        avgI[i] = modules[i].averageI
        bias[i] = modules[i].bias
        FT[i] = modules[i].FT_gfunc
        tags[i] = '_axion_mass_{:.1e}_'
        if stim_e == 1:
            tags[i] += '_StimE_'
        if synch == 1:
            tags[i] += '_Synch_'

    elif types[i] == 'galaxy':
        modules[i] = Galaxy_Survey(galaxy_survey, Mmin, Mmax)
        windows[i] = modules[i].window_prime
        avgI[i] = modules[i].averageI
        bias[i] = modules[i].bias
        FT[i] = modules[i].FT_gfunc
        tags[i] = '_Gsurvey_' + galaxy_survey + '_'

    else:
        print 'Have not yet included this type yet...'
        raise ValueError

fname = 'AngularPowerSpectrum_' + tags[0] + tags[1] + '.dat'
cross = Cross_Corr(windows, avgI, FT, bias, fname, zmax=zmax, Mmin=Mmin, Mmax=Mmax)
#cross.compute_angularPS()

ell_list = np.logspace(0., 4, 40)
process_Num = 10
#
#t_start = time.time()
#print cross.comoving_integrate(1e2)
#t_end = time.time()
#print t_end - t_start
#print cross.comoving_integrate(1e3)
#exit()

def anulgarPS(ell_v):
    soln = cross.comoving_integrate(ell_v)
    return soln

pool = Pool(processes=process_Num)
clHold = pool.map(anulgarPS, ell_list)
pool.close()
pool.join()
np.savetxt('outputs/' + cross.fname, clHold)


