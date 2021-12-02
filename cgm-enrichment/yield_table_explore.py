#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from JINAPyCEE import omega_plus as op
from scipy.ndimage import gaussian_filter1d

halo = 8508
cosmo = FlatLambdaCDM(H0=69.5, Om0=0.285, Ob0=0.0461) # FOGGIE I

atomic_number = {
    'H' : 1,  'He': 2,  'Li': 3,
    'Be': 4,  'B' : 5,  'C' : 6,
    'N' : 7,  'O' : 8,  'F' : 9,
    'Ne': 10, 'Na': 11, 'Mg': 12,
    'Al': 13, 'Si': 14, 'P' : 15,
    'S' : 16, 'Cl': 17, 'Ar': 18,
    'K' : 19, 'Ca': 20, 'Sc': 21,
    'Ti': 22, 'V' : 23, 'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27,
    'Ni': 28, 'Cu': 29, 'Zn': 30}

solar_abundance = np.array([1.00e+00, 1.00e-01, 2.04e-09,
    2.63e-11, 6.17e-10, 2.45e-04,
    8.51e-05, 4.90e-04, 3.02e-08,
    1.00e-04, 2.14e-06, 3.47e-05,
    2.95e-06, 3.47e-05, 3.20e-07,
    1.84e-05, 1.91e-07, 2.51e-06,
    1.32e-07, 2.29e-06, 1.48e-09,
    1.05e-07, 1.00e-08, 4.68e-07,
    2.88e-07, 2.82e-05, 8.32e-08,
    1.78e-06, 1.62e-08, 3.98e-08])


########################
# Setup table variations
########################

yield_names = {"table":"Massive & AGB Stellar",
               "nsmerger_table":"NS Merger",
               "pop3_table":"Pop 3 Stellar",
               "sn1a_table":"SNe 1a"}

# First entry is default for each table
yield_vars = {"table":{"R18+F12":"yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt",
                       "K10+LC18":"yield_tables/agb_and_massive_stars_K10_LC18_Ravg.txt",
                       "C15+N13":"yield_tables/agb_and_massive_stars_C15_N13_0_0_HNe.txt"},
              "nsmerger_table":{"A07":"yield_tables/r_process_arnould_2007.txt",
                                "R14":"yield_tables/r_process_rosswog_2014.txt"},
              "pop3_table":{"H10":"yield_tables/popIII_heger10.txt",
                            "N13":"yield_tables/popIII_N13.txt"},
              "sn1a_table":{"T86":"yield_tables/sn1a_t86.txt",
                            "T03":"yield_tables/sn1a_t03.txt",
                            "I99+W7":"yield_tables/sn1a_i99_W7.txt",
                            "S12":"yield_tables/sn1a_ivo12_stable_z.txt"}}


######################################
# Extract SFH and DM evolution on root
######################################

foggie_dir = os.path.join(os.environ.get("HOME"), "foggie", "foggie", "halo_infos")

rvir_masses = Table.read(os.path.join(foggie_dir,
                                      "00" + str(halo),
                                      "nref11c_nref9f",
                                      "rvir_masses.hdf5"))
sfr = np.genfromtxt(os.path.join(foggie_dir,
                                 "00" + str(halo),
                                 "nref11c_nref9f",
                                 "sfr"),
                    dtype=[('snaptype',"|U2"), ('z',float), ('sfr',float)])

# Save SFR to file
sfr_smooth = np.zeros((sfr.size, 2))
sfr_smooth[1:,0] = cosmo.age(sfr['z'][:-1]).to('yr')
sfr_smooth[ :,1] = gaussian_filter1d(sfr['sfr'], 10)

np.savetxt(f"smoothed_sfr_fine_halo00{halo:d}.txt", sfr_smooth)

# Generate array of DM evolution
mass_sort = np.argsort(rvir_masses['redshift'])[-1:0:-1]
DM_array_len = mass_sort.size+1
t_end = sfr_smooth[-1,0]

DM_array = np.zeros((mass_sort.size+1, 2), dtype=np.double)

DM_array[1:, 0] = cosmo.age(rvir_masses['redshift'][mass_sort])
DM_array[1:, 1] = rvir_masses['dm_mass'][mass_sort]
DM_array[0 , 1] = DM_array[1, 1]

# Confirm monotonically increasing age
assert (np.diff(DM_array[:,0]) > 0).all()


##########################
# Default model parameters (global)
##########################

control = {'special_timesteps':0,
           'dt':1e7,
           'tend':t_end,
          }

DM_evolution = {'DM_array':DM_array,
                'omega_0':0.285,
                'omega_b_0':0.0461,
                'H_0':69.5
               }

# IMPORTANT NOTE: By default, only stars between 1 and 30 Msun will eject yields.
# Stars above and below this limit will eject nothing.
# You can change this assumption with the "imf_yields_range" option
yields = {'imf_yields_range':[1,30],
          #'table':
         }

# t_star will override f_dyn
# With DM_evolution, mass_loading corresponds to end of sim when using time- and redshift- dependence
sf_fb = {'sfh_file':f'../smoothed_sfr_fine_halo00{halo:d}.txt',
         'imf_type':'chabrier', # default: kroupa
         'imf_bdys':[0.1,100.], # default: [0.1, 100]
        }

# exp_ml describes the dependence of outflow mass loading on DM mass (gets divided by 3) & z (divided by 2)
# redshift_f is final redshift
# exp_infall, t_inflow, f_t_ff, and m_inflow_in are mutually exclusive
# f_t_ff describes gas infall timescale as a fraction of free-fall time (=1.0, instant cooling)
flow = {'DM_outflow_C17':True,
           'C17_eta_z_dep':True, # try turning off and on
           'redshift_f':0.0,
           'mass_loading':1, # default: 1.0
           'exp_ml':2,
           'f_t_ff':1, # normalization
           't_ff_index':1 # redshift dependence; -3*param/2 
          } 

for table_param, table_dict in yield_vars.items():

    abundances = np.zeros((len(table_dict), len(atomic_number)),
                          dtype=np.double)
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    for row, (table_name, table) in enumerate(table_dict.items()):

        model = op.omega_plus(m_DM_0=DM_array[-1,1], mgal=1e1,
                              **control,
                              **DM_evolution,
                              **flow,
                              **sf_fb,
                              **yields,
                              Grackle_on=True,
                              **{table_param:table}
                                     )

        for i, isotope in enumerate(model.inner.history.isotopes):

            try:
                atom = atomic_number[isotope.split('-')[0]] - 1
            except KeyError:
                continue

            abundances[row, atom] += model.ymgal_outer[-1][i]

        abundances[row] /= abundances[row, 0]
    
        # Plot current abundance
        ax1.plot(np.log10(abundances[row])-np.log10(solar_abundance), label=table_name)
        
        if row > 0:
            ax2.semilogy(np.abs(abundances[row] -  abundances[0]), 
                         marker='.', label=table_name)
            
    ax1.axhline(0, color='gray', ls='--')
    ax1.legend()
    ax1.set_title(f"Altering {yield_names[table_param]} Yields")
    ax1.set_xlabel("Atomic Number")
    ax1.set_ylabel("Abundance [X/H]")
    #ax1.set_ylim(-8, 3)
    ax1.set_xlim(0,30)
    
    fig1.tight_layout()
    fig1.savefig(yield_names[table_param].replace(' ','_').lower() + '_xh.png')
    plt.close(fig1)
    del fig1

    ax2.grid(axis='y')
    ax2.legend()
    ax2.set_xlabel("Atomic Number")
    ax2.set_ylabel("Difference from Default")
    ax2.set_xlim(0,30)
    
    fig2.tight_layout()
    fig2.savefig(yield_names[table_param].replace(' ','_').lower() + '_var.png')
    plt.close(fig2)
    del fig2
