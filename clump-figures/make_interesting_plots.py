import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

halo_marker_dict = {'2392' : 'o',
                    '4123' : 'x',
                    '5016' : 'v',
                    '5036' : '*',
                    '8508' : 'D'} ##set which marker I want for each halo
halo_names_dict = {'2392'  :  'Hurricane',
                   '2878'  :  'Cyclone',
                   '4123'  :  'Blizzard',
                   '5016'  :  'Squall',
                   '5036'  :  'Maelstrom',
                   '8508'  :  'Tempest'} ##set FOGGIE names

def get_halo_names(num): ##get the FOGGIE name for each halo ID 
    if str(num) in halo_names_dict.keys():
        halo_name = halo_names_dict[str(num)]
    return halo_name

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
halo_legend = [Line2D([0], [0], lw= 0, color = 'k', marker = 'o', label = 'Hurricane'),
              Line2D([0], [0],  lw= 0, color = 'k', marker = 'x', label = 'Cyclone'),
              Line2D([0], [0],  lw= 0, color = 'k', marker = 'v', label = 'Squall'),
              Line2D([0], [0], lw= 0, color = 'k', marker = '*', label = 'Maelstrom'),
              Line2D([0], [0], lw= 0, color = 'k', marker = 'D', label = 'Tempest')]

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

for rs in rs_lis: ##we want one graph per redshift

    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Median Col Dens
    for halo in halo_marker_dict.keys(): ##all the halos
        name = get_halo_names(halo)
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
    set_outer_labels(axes, "Median $\log(N_{X_i})$", "MAD $\log(N_{X_i})$")
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
        name = get_halo_names(halo)
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
    set_outer_labels(axes, "Temperature [K]", "Median $\log(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlim(1e1,1e7)
    axes[0,0].set_ylim(12.4,19.1)
    axes[0,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    fig.savefig(f'median/temp_vs_med_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # Median Col Dens vs Distance
    for halo in halo_marker_dict.keys():
        name = get_halo_names(halo)
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
    set_outer_labels(axes, "Distance from Galaxy [kpc]", "Median $\log(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlim(4e2,4e4)
    axes[0,0].set_ylim(12.4,19.1)
    axes[0,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    fig.savefig(f'median/dist_vs_med_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()

    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Median Column Density
    for halo in halo_marker_dict.keys():
        name = get_halo_names(halo)
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
    set_outer_labels(axes, "Gas Density [g cm$^{-3}$]", "MAD $\log(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(1e-29,1e-22)
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'MAD/dens_vs_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Temperature
    for halo in halo_marker_dict.keys():
        name = get_halo_names(halo)
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
    set_outer_labels(axes, "Temperature [K]", "MAD $\log(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(1e1,1e7)
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'MAD/temp_vs_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # MAD vs Distance
    for halo in halo_marker_dict.keys():
        name = get_halo_names(halo)
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
    set_outer_labels(axes, "Distance from Galaxy [kpc]", "MAD $\log(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlim(4e2,4e4)
    axes[0,0].set_ylim(1e-3,1e0)
    fig.savefig(f'MAD/dist_vs_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    fig, axes = plt.subplots(**subplot_params)
    # Absorber MAD vs Abundance MAD
    for halo in halo_marker_dict.keys():
        name = get_halo_names(halo)
        for ion in ion_dict.keys():
            elem = ion.split()[0]
            ax = axes[ad[elem]]
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion.replace(" ","_")}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    elem_mad = ds["mad_of_element"] ##MAD elemental abundance vs MAD the col dens
                    col_dens_spread = ds["mad_for_col_desnity"]
                    ax.scatter(elem_mad[col_dens_spread>0], col_dens_spread[col_dens_spread>0], c = ion_dict[ion], marker = halo_marker_dict[halo])
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue

    for elem in ad.keys():
        ax = axes[ad[elem]]
        ax.legend(handles=legend_dict[elem], ncol=1)

    #fig.suptitle(f"MAD Elemental Abundance vs MAD Column Density, Redshift {rs}; n={len(all_abun)}, r={corr[0]:.2f}, p={corr[1]:.2e}")
    fig.legend(handles = halo_legend, ncol = 5, bbox_to_anchor=(0.5, 0.93), bbox_transform=fig.transFigure, loc="upper center")
    set_outer_labels(axes, "MAD $A_X$", "MAD $\log(N_{X_i})$")
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    fig.savefig(f'MAD/mad_vs_elem_mad_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()
    
    # ##this one is different because it's a histogram and not a scatter plot
    # diff_list = [] ##initalize list to be put into a histogram
    # for halo in halo_marker_dict.keys():
    #     name = get_halo_names(halo)
    #     for ion in ion_dict.keys():
    #         try:
    #             ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
    #             if len(ds['density']) != 0:
    #                 diff = ds["diff_from_solar_abun"] ##get the data from data tables
    #                 for value in diff:
    #                     if pd.isna(value) == False:
    #                         diff_list.append(value) ##make the list of values
    #                         print('added!')
    #             else:
    #                 print(f'No {ion} in halo {halo} z{rs}')
    #         except FileNotFoundError:
    #             print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
    #             continue
    # kernel = stats.gaussian_kde(diff_list)
    # xax = np.linspace(-2.0, 2.0, num = 500)
    # ax.plot(xax, kernel.evaluate(xax)) ##make the histogram
    # fig.suptitle(f"Diffrence between Solar Abundance and Median Column Density, Redshift {rs}") ##needs a better title but I can't think of one right now
    # ax.set_ylabel("relative frequency")
    # params = {'mathtext.default': 'regular' }          
    # ax.rcParams.update(params)
    # ax.set_xlabel("$log(N_{\odot}) - log(N_{med}) [dex]$") ##has to be in dex(order of magnitude) because of how sal the super snake works
    # fig.savefig(f'./solar_diff_z{rs}.png', bbox_inches='tight', dpi = 600)
    # plt.close()

print('All done!')