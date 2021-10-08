#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    folder = input("Enter a folder: ")
else:
    folder = sys.argv[1]
    
if folder[-1] == '/':
    folder = folder[:-1]


data = np.genfromtxt(folder+"/cgm_abundances.txt")
ref = data[0,:]

for i in range(1, data.shape[0]):
    plt.semilogy(np.abs(data[i,:]-ref)/ref*100, lw=1)
    
plt.ylabel("Abundance Deviation  [%]")
plt.xlabel("Atomic Number")
plt.title("Internal Deviation")
plt.savefig(folder+"/internal_deviation.png")
plt.clf()


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

for i in range(0, data.shape[0]):
    plt.semilogy(np.abs(data[i,:]-solar_abundance)/solar_abundance*100, lw=1)
    
plt.ylabel("Abundance Deviation  [%]")
plt.xlabel("Atomic Number")
plt.title("Deviation from Solar")
plt.savefig(folder+"/solar_deviation.png")
plt.clf()


for i in range(0, data.shape[0]):
    plt.semilogy(data[i,:], lw=1)
plt.semilogy(solar_abundance, c='k', ls=":", lw="1")

plt.ylim(1e-22, 1)
plt.xlim(0, data.shape[1])
plt.ylabel("Abundance")
plt.xlabel("Atomic Number")
plt.title("Abundance Patterns")
plt.savefig(folder+"/abundance_patterns.png")
plt.clf()


fid = np.genfromtxt(folder+"/dm_ev/cgm_dm_fid.txt")
plt.semilogy(np.linspace(0, 13, fid.size), fid, ls = '--')

for i in range(0, data.shape[0]-1):
    data = np.genfromtxt(folder+f"/dm_ev/cgm_dm_{i:03d}.txt")
    plt.semilogy(np.linspace(0, 13, data.size), data, ls="-")
    
plt.xlim(0,13)
plt.ylabel("DM Mass")
plt.xlabel("Time [Gyr]")
plt.savefig(folder+"/dm_hist.png")

