#!/usr/bin/env python
# coding: utf-8

from JINAPyCEE import omega_plus as op
import numpy as np
from astropy.cosmology import Planck18

nsamples = 50

logt_z2 = np.log10(Planck18.age(2).to('yr').value)

def tanh(x, x0, y0, k1, k2):
    return (y0+k1) + k1*np.tanh(k2*(x-x0))

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

abundances = np.empty((nsamples+1, len(atomic_number)))

popt = [8,9,1.75,1]

# Set up time steps for dark matter evolution
log_DM_times = np.array(
      [ 6.        ,  6.14186012,  6.28372023,  6.42558035,  6.56744046,
        6.70930058,  6.85116069,  6.99302081,  7.13488092,  7.27674104,
        7.41860116,  7.56046127,  7.70232139,  7.8441815 ,  7.98604162,
        8.12790173,  8.26976185,  8.41162197,  8.48831869,  8.55348208,
        8.63017881,  8.6953422 ,  8.77203892,  8.83720231,  8.88980767,
        8.93672394,  8.97906243,  9.03166779,  9.07858406,  9.12092254,
        9.1735279 ,  9.22044417,  9.26278266,  9.30282191,  9.33947938,
        9.37328211,  9.40464277,  9.44468203,  9.4813395 ,  9.51514222,
        9.54650289,  9.58654214,  9.62319962,  9.65700234,  9.68836301,
        9.72840226,  9.76505973,  9.79886246,  9.83022312,  9.87026238,
        9.90691985,  9.94072257,  9.97208324, 10.01212249, 10.04877996,
       10.08258269, 10.11394335])

DM_times = np.concatenate(([0], 10**log_DM_times))

# Set model parameters, including best fits to MW values
static_params = {'mgal':1.0, 
                 #'m_DM_0':1.0e12,
                 'DM_outflow_C17':True,
                 'C17_eta_z_dep':True,
                 'redshift_f':0.0,
                 'f_t_ff':10,
                 "imf_yields_range":[1,30],
                 'imf_type':'kroupa', # default
                 'imf_bdys':[0.1,100.], # default
                 'sfe_m_dep':False,
                 'Grackle_on':True,
                 'special_timesteps':150,
                }

fit_params = {"t_star":1.0,
              'sfe':5.0e-10,
              'mass_loading':2,
              'exp_ml':3
             }

# Run the fiducial model, who's DM evolution is the one we're modifying from
DM_mass = 10**tanh(log_DM_times, # ignore time zero; bad math
                popt[0], 
                popt[1], 
                popt[2],
                popt[3])
DM_mass = np.concatenate(([DM_mass[0]], DM_mass)) # Add zero mass by hand

DM_array = np.column_stack((DM_times, DM_mass))

model = op.omega_plus(DM_array=DM_array.tolist(), m_DM_0=DM_mass[-1],
                      **static_params, **fit_params)

# Save CGM abundances, normalized to H, in order of atomic number up to Zn
this_abun = np.zeros(len(atomic_number))
for iso, isotope in enumerate(model.inner.history.isotopes): 
    try:
        atom = atomic_number[isotope.split('-')[0]] - 1
    except KeyError:
        continue
    this_abun[atom] += model.ymgal_outer[-1][iso]

abundances[0] = this_abun[:30]/this_abun[0]
np.savetxt(f"z2_2e12/dm_ev/cgm_dm_fid.txt",
           DM_mass)

# Randomly perturb the dark matter evolution (within constraints)
rng = np.random.default_rng(65468974635) # the new numpy random

for i in range(nsamples):

    # Randomly select horizontal offset and steepness adjustments
    # baseline is popt
    x0_add = rng.uniform(-0.5, 0.5)
    k2_mult = rng.uniform(0.5, 2.0)
    k1_mult = 1

    # Adjust k2 multiplier (steepness of transition) so that the LHS is close to target
    while (tanh(6, 
                popt[0]+x0_add, 
                popt[1], 
                popt[2]*k1_mult,
                popt[3]*k2_mult
               ) - 9)/9 > 0.01:
        #print("Adjusting k2 multiplier")
        k2_mult += 0.1

    # Adjust k1 multiplier (vertical stretch) so that RHS is  close to target
    rhs_err = (tanh(np.log10(1.3e10), 
                    popt[0]+x0_add, 
                    popt[1],
                    popt[2]*k1_mult, 
                    popt[3]*k2_mult
                   ) - 12)/12
    while np.abs(rhs_err) > 0.001:
        #print("Adjusting k1 multiplier")
        if rhs_err > 0: # overshooting
            k1_mult *= 0.9
        elif rhs_err < 0: # undershooting
            k1_mult *= 1.1
        rhs_err = (tanh(logt_z2, 
                        popt[0]+x0_add,
                        popt[1],
                        popt[2]*k1_mult,
                        popt[3]*k2_mult
                       ) - 12.3)/12.3

    # be looser re: LHS requirement now that I've fiddled with the function params a bunch
    assert np.abs(tanh(6,                
                       popt[0]+x0_add,
                       popt[1],
                       popt[2]*k1_mult,
                       popt[3]*k2_mult
                      ) - popt[1])/popt[1] < 0.01

    assert np.abs(tanh(np.log10(1.3e10), 
                       popt[0]+x0_add, 
                       popt[1], 
                       popt[2]*k1_mult,
                       popt[3]*k2_mult
                      ) - 12.3)/12.3 < 0.1
    
    DM_mass = 10**tanh(log_DM_times,
                    popt[0]+x0_add, 
                    popt[1], 
                    popt[2]*k1_mult,
                    popt[3]*k2_mult)
    DM_mass = np.concatenate(([DM_mass[0]], DM_mass))
    
    DM_array = np.column_stack((DM_times, DM_mass))
    
    model = op.omega_plus(DM_array=DM_array.tolist(), m_DM_0=DM_mass[-1],
                          **static_params, **fit_params)
    
    # Save CGM abundances, normalized to H, in order of atomic number
    this_abun = np.zeros(len(atomic_number))
    for iso, isotope in enumerate(model.inner.history.isotopes): 
        try:
            atom = atomic_number[isotope.split('-')[0]] - 1
        except KeyError:
            continue
        this_abun[atom] += model.ymgal_outer[-1][iso]
    
    abundances[i+1] = this_abun[:30]/this_abun[0]
    np.savetxt(f"z2_2e12/dm_ev/cgm_dm_{i:03d}.txt",
               DM_mass)

header = '#'
for a in atomic_number.keys(): # order isn't guarenteed but nothing modified dict
    header += ' ' + a
    
np.savetxt("z2_2e12/cgm_abundances.txt", abundances, header=header)
