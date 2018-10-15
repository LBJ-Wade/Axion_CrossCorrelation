import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)
from axion_decay import *
from dm_profiles import *
from cosmology import *
from scipy.interpolate import interp1d

mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=14
mpl.rcParams['ytick.labelsize']=14

test_plots = 'test_plots/'


def stim_emis_Halos_z_dep(m_a, halo_masses, fs=16):
    # Simulated emission at the center of halos of fixed mass as a function of redshift
    # halo_masses should be a list
    file_name = 'Stim_Emis_HaloCenter_ma_{:.1e}_redshift_dependence.pdf'.format(m_a)
    axionC = Axion_Decay(m_a, 1e-11, 1e6, 1e16)
    #zlist = np.logspace(-1, 1, 100)
    zlist = np.linspace(0, 10, 100)
    
    color_list = ['#442B48', '#726E60', '#98B06F', '#B6DC76', '#DBFF76']
    pl.figure()
    ax = pl.gca()
    for j,MH in enumerate(halo_masses):
        cmb = np.zeros_like(zlist)
        stime = np.zeros_like(zlist)
        for i,z in enumerate(zlist):
            cmb[i], stime[i] = axionC.gamma_phasespace(MH, 0., z)
        pl.plot(zlist, cmb, color_list[j], lw=1, label='Halo Mass: {:.0e}'.format(MH))
        pl.plot(zlist, stime, color_list[j], ls='-.', lw=1)

    #plt.tight_layout()
    ax.set_xlabel('Redshift', fontsize=fs)
    ax.set_ylabel(r'$f\gamma $', fontsize=fs)
    #ax.set_xscale("log")
    ax.set_xlim([0., 10.])
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.savefig(test_plots + file_name)
    return

def stim_emis_Volume_avg_z_dep(m_a_list, fs=16):
    file_name = 'Stim_Emis_VolumeAvg_redshift_dependence.pdf'
    zlist = np.linspace(0, 10, 100)
    axionC = np.zeros(len(m_a_list), dtype=object)
    for i in range(len(m_a_list)):
        axionC[i] = Axion_Decay(m_a_list[i], 1e-11, 1e6, 1e16)
    
    color_list = ['#39A2AE', '#561F37', '#EF6461']
    pl.figure()
    ax = pl.gca()

    for j in range(len(m_a_list)):
        cmb = np.zeros_like(zlist)
        synch = np.zeros_like(zlist)
        norm = term0 = Omega_DM * critical_rho(0)
        for i,z in enumerate(zlist):
            cmb[i] = axionC[j].gamma_phasespace(1e12, 0., z)[0] - 1.
            dn_dm_tab = axionC[j].haloF.dn_dm_tabular(1e6, 1e16, z)
            dn_dm_tab = dn_dm_tab[dn_dm_tab[:,1] > 0]
            dn_dm = interp1d(np.log10(dn_dm_tab[:,0]), np.log10(dn_dm_tab[:,1]), kind='linear',
                                bounds_error=False, fill_value=-100)
            
            term_int = quad(lambda m: 10.**dn_dm(np.log10(m)) * axionC[j].SFR(m, z)**1.77 / m**0.77 * NFW(m, z).b_field_integral() * axionC[j].beta_coeff(z), 1e6, 1e16)
        
            synch[i] = term_int[0] / norm
        
        pl.plot(zlist, cmb,  color_list[j], lw=1)
        pl.plot(zlist, synch, color_list[j], ls='-.', lw=1, label=r'$m_a =$ {:.1e}'.format(m_a_list[j]))

    #plt.tight_layout()
    ax.set_xlabel('Redshift', fontsize=fs)
    ax.set_ylabel(r'$<f \gamma> / (\Omega_c \rho_c)$', fontsize=fs)
    ax.set_xlim([0., 10.])
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.savefig(test_plots + file_name)
    return


def window_normalized(m_a_list, zmax=10, fs=16):
    file_name = 'Window_Axions.pdf'
    color_list = ['#39A2AE', '#561F37', '#EF6461']
    pl.figure()
    ax = pl.gca()
    
    axionC = np.zeros(len(m_a_list), dtype=object)
    for i in range(len(m_a_list)):
        axionC[i] = Axion_Decay(m_a_list[i], 1e-11, 1e6, 1e16)
    
    for j in range(len(m_a_list)):
        wind = axionC[j].mean_window(zmax)
        pl.plot(wind[:,0], wind[:,1],  color_list[j], lw=1, label=r'$m_a =$ {:.1e}'.format(m_a_list[j]))

    #plt.tight_layout()
    ax.set_xlabel('Redshift', fontsize=fs)
    ax.set_ylabel(r'$<I_\nu>$', fontsize=fs)
    ax.set_xlim([0., zmax])
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.savefig(test_plots + file_name)

    return

def SFR_HaloMass_Dep(zlist, fs=16):
    file_name = 'HaloMasses_Dependence.pdf'
    color_list = ['#39A2AE', '#561F37', '#EF6461']
    pl.figure()
    ax = pl.gca()
    
    axionC = Axion_Decay(1e-5, 1e-11, 1e6, 1e16)
    Mhlist = np.logspace(7, 14, 30)
    for i,z in enumerate(zlist):
        sfr = np.zeros_like(Mhlist)
#        sfrM = np.zeros_like(Mhlist)
        for j in range(len(Mhlist)):
            sfr[j] = axionC.SFR(Mhlist[j], z)**1.77 / Mhlist[j]**0.77
        
        pl.plot(Mhlist, sfr,  color_list[i], lw=1, label=r'z = {:.0f}'.format(z))
    
    ax.set_xlabel('Halo Mass', fontsize=fs)
    ax.set_ylabel(r'SFR', fontsize=fs)
    ax.set_xlim([1e7, 1e14])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.savefig(test_plots + file_name)
    return

def FT_Mass_Dep(kvals, z, fs=16):
    file_name = 'FT_Mass_Dependence.pdf'
    color_list = ['#39A2AE', '#561F37', '#EF6461']
    pl.figure()
    ax = pl.gca()
    
    axionC = Axion_Decay(1e-5, 1e-11, 1e1, 1e16, stim_e=0., synch=0.)
    Mhlist = np.logspace(1, 16, 30)
    for i,k in enumerate(kvals):
        tot, arrhold, arrhold2 = axionC.autoPS_1H(k, z, return_mass_dep=True)
#        print arrhold
#        print arrhold2
        pl.plot(arrhold[:,0], arrhold[:,1],  color_list[i], lw=1, label=r'k = {:.1e}'.format(k))
        pl.plot(arrhold2[:,0], arrhold2[:,1], color_list[i], ls='--', lw=1)

    ax.set_xlabel('Halo Mass', fontsize=fs)
    ax.set_ylabel(r'FT', fontsize=fs)
    ax.set_xlim([1e7, 1e16])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.savefig(test_plots + file_name)

    return

def Plot_AutoCorrelation_Axion(massL, fs=16):
    file_name = 'Axion_AutoCorrelation.pdf'
    color_list = ['#39A2AE', '#561F37', '#EF6461']
    pl.figure()
    ax = pl.gca()
    
    for j,ma in enumerate(massL):
        tags = ['_Stim_E_{:.1e}_'.format(ma), '_CMB_{:.1e}_'.format(ma)]
        for i,tag in enumerate(tags):
            dataLd = np.loadtxt('outputs/Axion_AutoCorre_PS' + tag + '.dat')
            if i == 0:
                lst = '-'
            else:
                lst = '--'
            pl.plot(dataLd[:,0], dataLd[:,1], color_list[j], ls=lst, lw=1, label=tag)

    ax.set_xlabel('k [$Mpc^{-1}$]', fontsize=fs)
    ax.set_ylabel(r'P(k)', fontsize=fs)
    ax.set_xlim([1e-3, 1e2])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(loc=1, frameon=True, framealpha=0.5, fontsize=9, ncol=1, fancybox=True)
    plt.savefig(test_plots + file_name)
    return
