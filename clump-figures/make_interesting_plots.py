import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from scipy import stats

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

halo_marker_dict = {'2392' : 'o',
                    '4123' : 'x',
                    '5016' : 'v',
                    '5036' : '*',
                    '8508' : 'D'} ##set which marker I want for each halo

halo_names_dict = {'2392'  :  'Hurricane',
                   #'2878'  :  'Cyclone', # unused for now
                   '4123'  :  'Blizzard',
                   '5016'  :  'Squall',
                   '5036'  :  'Maelstrom',
                   '8508'  :  'Tempest'} ##set FOGGIE names

halo_colors = {"Hurricane":"olive",
               "Blizzard":"steelblue",
               "Squall":"lightcoral",
               "Maelstrom":"orchid",
               "Tempest":"seagreen"}

rs_lis = [2.0, 2.5] ##list of rs to analyze, true

subplot_params = dict(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8,10), gridspec_kw = {'wspace':0.1, 'hspace':0.1})

ad = {'C':(0,0),
      'N':(0,1),
      'O':(1,0),
      'Mg':(1,1),
      'Si':(2,0),
      'S':(2,1)}

ion_dict = {#'C_I' : 'silver',
            'C II' : 'darkgrey',
            'C III' : 'grey',
            'C IV' : 'dimgrey',
            #'Si I' : 'lightcoral',
            'Si II' : 'indianred',
            'Si III' : 'firebrick',
            'Si IV' : 'darkred',
            # 'Fe I' : 'salmon',
            # 'Fe II' : 'tomato',
            # 'Fe III' : 'maroon',
            #'N I' : 'lightgreen',
            'N II' : 'limegreen',
            'N III' :'forestgreen' ,
            'N IV' :'green' ,
            'N V' : 'darkgreen' ,
            #'O I' : 'cornflowerblue',
            'O II' : 'royalblue',
            'O III' :'blue',
            'O IV' : 'mediumblue',
            'O V' : 'darkblue',
            'O VI' : 'navy',
            #'Mg I' : 'magenta',
            'Mg II' : 'darkmagenta',
            #'S I' : 'cornsilk',
            'S II' : 'lemonchiffon',
            'S III' : 'khaki',
            'S IV' : 'gold',
            'S V' : 'goldenrod',
            'S VI' : 'darkgoldenrod'} ##dictionary of ions to colors

##set the legends to be the way I want them
halo_legend = []
for halo, marker in halo_marker_dict.items():
    halo_legend.append(Line2D([0], [0], lw= 0, color = 'k', marker = marker, label = halo_names_dict[halo]))

legend_dict = {'C':[],
               'N':[],
               'O':[],
               'Mg':[],
               'Si':[],
               'S':[]}
for ion in ion_dict.keys():
    legend_patch = Patch(fc = ion_dict[ion], label = ion)
    legend_dict[ion.split()[0]].append(legend_patch)

# Assume 3 rows and 2 cols
def set_outer_labels(axes, xlabel, ylabel):

    axes[2,0].set_xlabel(xlabel)
    axes[2,1].set_xlabel(xlabel)

    axes[0,0].set_ylabel(ylabel)
    axes[1,0].set_ylabel(ylabel)
    axes[2,0].set_ylabel(ylabel)

fig1, axes1 = plt.subplots(**subplot_params)
for rs in rs_lis: ##we want one graph per redshift

    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Median Col Dens
    for halo in halo_marker_dict.keys(): ##all the halos
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys(): ##all the ions
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0: ##only use data where there actually is something
                    med_col_dens = ds["median_col_desnity"] ##plot the spread of the median vs the median col dens
                    col_dens_spread = ds["mad_for_col_desnity"]
                    ax.scatter(med_col_dens, col_dens_spread, c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError: ##handles if ion was not done for some reason
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
    
    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=3)

    #fig.suptitle(f"MAD Column Density vs Column Density, Redshift {rs}") 
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "Median $\log_{10}(N_{X_i})$", "MAD$\,(\log_{10}\ N_{X_i})$")
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(12.4,19.1)
    axes[0,0].xaxis.set_minor_locator(MultipleLocator(0.5))
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'./mad_vs_med_z{rs}.png', bbox_inches='tight', dpi = 600) ##make sure it saves to the right place with the right name
    plt.close() ##so the rest doesn't plot on top of this data
    ##a simmilar process is used for the rest of these plots, simply with different quantities
        
    fig, axes = plt.subplots(**subplot_params)
    # Temperature vs Median Col Dens
    for halo in halo_marker_dict.keys():
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    med_col_dens = ds["median_col_desnity"] ##temperature and median column density
                    temp = ds["temperature"]
                    ax.scatter(temp, med_col_dens, c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue

    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=1)

    #fig.suptitle(f"Temperature vs Column Density, Redshift {rs}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "Temperature [K]", "Median $\log_{10}(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlim(1e1,1e7)
    axes[0,0].set_ylim(12.4,19.1)
    axes[0,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    fig.savefig(f'median/temp_vs_med_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # Median Col Dens vs Distance
    for halo in halo_marker_dict.keys():
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    med_col_dens = ds["median_col_desnity"] ##distance from galaxy vs median col dens
                    dist = ds["distance_from_galaxy"]/3.086e21 # cm to kpc
                    ax.scatter(dist, med_col_dens, c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=3)
      
    #fig.suptitle(f"Distance vs Column Density, Redshift {rs}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "Distance from Galaxy [kpc]", "Median $\log_{10}(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlim(4e2,4e4)
    axes[0,0].set_ylim(12.4,19.1)
    axes[0,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    fig.savefig(f'median/dist_vs_med_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()

    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Median Column Density
    for halo in halo_marker_dict.keys():
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    col_dens_spread = ds["mad_for_col_desnity"] ##MAD col dens vs density
                    dens = ds["density"]
                    ax.scatter(dens, col_dens_spread, c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=3)
      
    #fig.suptitle(f"Density vs Spread of Column Density , Redshift {rs}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "Gas Density [g cm$^{-3}$]", "MAD$\,(\log_{10}\ N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(1e-29,1e-22)
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'MAD/dens_vs_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Temperature
    for halo in halo_marker_dict.keys():
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    col_dens_spread = ds["mad_for_col_desnity"] ##MAD col dons vs temperature
                    temp = ds["temperature"]
                    ax.scatter(temp, col_dens_spread, c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=3)
      
    #fig.suptitle(f"Temperature vs Spread of Column Density, Redshift {rs}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "Temperature [K]", "MAD$\,(\log_{10}\ N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(1e1,1e7)
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'MAD/temp_vs_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Distance
    for halo in halo_marker_dict.keys():
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    col_dens_spread = ds["mad_for_col_desnity"] ##MAD col dens vs distance from galaxy
                    dist = ds["distance_from_galaxy"]/3.086e21
                    ax.scatter(dist, col_dens_spread, c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=3)
      
    #fig.suptitle(f"Distance vs Spread of Column Density, Redshift {rs}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "Distance from Galaxy [kpc]", "MAD$\,(\log_{10}\ N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(4e2,4e4)
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'MAD/dist_vs_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # Absorber MAD vs Abundance MAD
    for halo in halo_marker_dict.keys():
        name = halo_names_dict.get(halo)
        abun = np.genfromtxt(f"/home/claire/trident_uncertainty/mods/abundances/abundances_AGB_massive_yields_halo{halo}_z{rs}.txt",
                             skip_header=1)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    elem_mad = ds["mad_of_element"]/np.median(abun[:,atomic_number[elem]-1]) 
                    col_dens_spread = ds["mad_for_col_desnity"]/ds["median_col_desnity"] 
                    ax.scatter(elem_mad[col_dens_spread>0], col_dens_spread[col_dens_spread>0], c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue

    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=3)

    #fig.suptitle(f"MAD Elemental Abundance vs MAD Column Density, Redshift {rs}; n={len(all_abun)}, r={corr[0]:.2f}, p={corr[1]:.2e}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "MAD$\,(A_X)\ /\ \mathrm{Med}(A_{X})$", "MAD$\,(\log_{10}\ N_{X_i})\ /\ \mathrm{Med}(\log_{10}\ N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(1e-2,1e0)
    axes[0,0].set_ylim(5e-5,1e-1)
    fig.savefig(f'MAD/mad_vs_elem_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    
    ##this one is different because it's a histogram and not a scatter plot
    diff_lists = {'C':[],
                  'N':[],
                  'O':[],
                  'Mg':[],
                  'Si':[],
                  'S':[]} ##initalize list to be put into a histogram
    nan_counter = {'C':0,
                  'N':0,
                  'O':0,
                  'Mg':0,
                  'Si':0,
                  'S':0}

    for halo in halo_marker_dict.keys():

        # collect data into lists by element 
        name = halo_names_dict.get(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    diff = -1*ds["diff_from_solar_abun"] ##get the data from data tables
                    for value in diff:
                        if pd.isna(value) == False:
                            diff_lists[elem].append(value) ##make the list of values
                            #print('added!')
                        else:
                            nan_counter[elem] += 1
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
        
    # plot all halos as one
    total = 0
    xax = np.linspace(-2.0, 2.0, num = 500)
    for elem, data in diff_lists.items():
        ax = axes1[ad[elem]]
        kernel = stats.gaussian_kde(data)
        ax.plot(xax, kernel.evaluate(xax), label=f"z={rs}")#, color=halo_colors[name]) ##make the histogram; color by halo
        #data = []
        if rs == 2.5:
            ax.axvline(0, color='gray', ls='--')
            ax.text(-2, 2, elem, verticalalignment='center', fontsize='x-large')
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            
        print("NaN fraction for", elem, "is", nan_counter[elem]/(len(data)+nan_counter[elem]))
        
    #fig.suptitle(f"Diffrence between Solar Abundance and Median Column Density, Redshift {rs}") ##needs a better title but I can't think of one right now
    set_outer_labels(axes1, "Med$(\log_{10}\ N_{X_i}) - \log_{10}\ N_\odot$ [dex]", "Relative Frequency") ##has to be in dex(order of magnitude) because of how sal the super snake works
axes1[0,0].legend()
fig1.savefig(f'./solar_diff_all.png', bbox_inches='tight', dpi = 600)


print('All done!')
