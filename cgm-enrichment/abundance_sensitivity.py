#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from JINAPyCEE import omega_plus as op
from mpi4py import MPI


#########
# Globals (excluding default model params; see below)
#########

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

nsamples = size # easy to divide work
halo = 8508

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

param_guide = {'mass_loading': 0,
               'exp_ml':       1,
               'f_t_ff':       2,
               #'t_ff_index':   3, # takes really fucking long
              }

# Only used by root but easily modified if up top
bounds = {'mass_loading': (0.1, 10),
          'exp_ml':       (1, 4),    # will only choose integers
          'f_t_ff':   (1, 10),
         }


######################################
# Extract SFH and DM evolution on root
######################################

if rank == 0:
    from astropy.table import Table
    from astropy.cosmology import FlatLambdaCDM
    from scipy.ndimage import gaussian_filter1d

    cosmo = FlatLambdaCDM(H0=69.5, Om0=0.285, Ob0=0.0461) # FOGGIE I
    
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

# First, broadcast shape of DM_array to children to prepare buffers
# Also, end time
else:
    DM_array_len = None
    t_end = None

DM_array_len = comm.bcast(DM_array_len, root=0)
t_end = comm.bcast(t_end, root=0)

# Second, broadcast DM_array that was prepped on root
if rank == 0:
    DM_array = np.zeros((mass_sort.size+1, 2), dtype=np.double)
    
    DM_array[1:, 0] = cosmo.age(rvir_masses['redshift'][mass_sort])
    DM_array[1:, 1] = rvir_masses['dm_mass'][mass_sort]
    DM_array[0 , 1] = DM_array[1, 1]

    # Confirm monotonically increasing age
    assert (np.diff(DM_array[:,0]) > 0).all()
    
else:
    DM_array = np.empty((DM_array_len,2), dtype=np.double)
    
comm.Bcast(DM_array, root=0)

        
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


#################
# Model Variation
#################

param_var = None

# Fill param_var on root. Recall nsamples = size
if rank == 0:

    param_var = np.empty((nsamples, (len(param_guide))), dtype=np.double)
    
    rng = np.random.default_rng()
    
    param_var[:, param_guide['mass_loading']] = rng.uniform(*bounds['mass_loading'], nsamples)
    param_var[:, param_guide['exp_ml']] = rng.integers(*bounds['exp_ml'], nsamples)
    param_var[:, param_guide['f_t_ff']] = rng.uniform(*bounds['f_t_ff'], nsamples)
    #param_var[:, param_guide['t_ff_index']] = rng.uniform(1/3,1, nsamples)

# Each process runs one model of each varied parameter
# Scatter will break up rows of the send buffer
my_params = np.empty(len(param_guide), dtype=np.double)
comm.Scatter(param_var,  my_params, root=0)


############
# Run Models
############

# Record abundances. Use first column to preserve modified parameter
my_abund = np.zeros((len(param_guide), len(atomic_number)+1), dtype=np.double)

for param, col in param_guide.items():
    
    flow[param] = my_params[col]

    model = op.omega_plus(m_DM_0=DM_array[-1,1], mgal=1e1,
                          **control,
                          **DM_evolution,
                          **flow,
                          **sf_fb,
                          **yields,
                          Grackle_on=True,
                         )

    # Sum isotopal abundances to get elemental abundances
    for j, isotope in enumerate(model.inner.history.isotopes):

        # Get the atomic number of the isotope, counting from 1
        try:
            atom = atomic_number[isotope.split('-')[0]]
        except KeyError:
            continue

        my_abund[col][atom] += model.ymgal_outer[-1][j]

    my_abund[col] /= my_abund[col][1] # Normalize to hydrogen
    my_abund[col][0] = flow[param] # Preserve modified param

# Gather model results
abundance_var = None
if rank == 0:
    abundance_var = np.empty((size, *my_abund.shape), dtype=np.double)
comm.Gather(my_abund, abundance_var, root=0)

if rank == 0:
    
    elements = sorted(atomic_number.keys(), key=lambda x: atomic_number[x])
    
    # Save a file of abundances for each parameter varied
    for param, col in param_guide.items():

        header = 'param'
        for e in elements:
            header += ' ' + e
        
        np.savetxt(f"abundance_var_{param}.txt", 
                   abundance_var[:,col,:], 
                   header=header)
