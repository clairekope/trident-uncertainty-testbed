#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from JINAPyCEE import omega_plus as op
from scipy.ndimage import gaussian_filter1d

halo = 8508
cosmo = FlatLambdaCDM(H0=69.5, Om0=0.285, Ob0=0.0461) # FOGGIE I

target_redshifts = [1.5,2.0,2.5]

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


###########################################
# Setup ABG & massive star yield variations
###########################################


with open("yield_list_AGB_massive.txt","r") as f:
    yield_tables = f.read().split("\n")
    if '' in yield_tables:
        yield_tables.remove('') # remove end of file newline


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


###########################
# Model parameters (global)
###########################

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
           'exp_ml':2, # default: 2.0
           'f_t_ff':1, # normalization
           't_ff_index':1 # redshift dependence; -3*param/2 
          } 

######################
# Calculate abundances
######################

abundances = np.zeros((len(target_redshifts), len(yield_tables), len(atomic_number)),
                      dtype=np.double)
    
for row, table_name in enumerate(yield_tables):

    model = op.omega_plus(m_DM_0=DM_array[-1,1], mgal=1e1,
                          **control,
                          **DM_evolution,
                          **flow,
                          **sf_fb,
                          **yields,
                          **{"table":"yield_tables/"+table_name},
                          Grackle_on=True)

    ages = model.inner.history.age * u.yr

    for z_ind, z in enumerate(target_redshifts):
        print("Redshift:", z)
        
        this_abun = abundances[z_ind]

        # Find the age index most closely corresponding to this redshift
        target_age = cosmo.age(z)
        age_ind = np.where(ages < target_age)[0][-1] # age below

        # Is the next age index closer?
        if (ages[age_ind+1] - target_age) < np.abs(ages[age_ind] - target_age):
          age_ind += 1

        # Iterate over all isotopes to get abundances at this redshift
        for i, isotope in enumerate(model.inner.history.isotopes):

            try:
                atom = atomic_number[isotope.split('-')[0]] - 1
            except KeyError:
                continue

            this_abun[row, atom] += model.ymgal_outer[age_ind][i]

        this_abun[row] /= this_abun[row, 0]

#########################
# Save abundances to file
#########################

header = ''
for a in atomic_number.keys(): # order isn't guarenteed but nothing modified dict
    header += a + ' '

# Lazy check b/c techincally, dict ordering is not ensured.
assert "H He Li" in header, "Order of atom labels is not as expected!"

for z_ind, z in enumerate(target_redshifts):
    np.savetxt(f"abundances_AGB_massive_yields_halo{halo}_z{z}.txt", 
                np.row_stack((solar_abundance, abundances[z_ind])),
                delimiter=' ', header=header, comments="")

