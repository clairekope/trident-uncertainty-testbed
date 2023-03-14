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

def get_halo_names(num): ##get the FOGGIE name for each halo ID 
    if str(num) in halo_names_dict.keys():
        halo_name = halo_names_dict[str(num)]
    return halo_name

rs_lis = [2.0, 2.5] ##list of rs to analyze, true

subplot_params = dict(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8,10), gridspec_kw = {'wspace':0.1, 'hspace':0.1})


halo_marker_dict = {'2392' : 'o',
                    '4123' : 'x',
                    '5016' : 'v',
                    '5036' : '*',
                    '8508' : 'D'}

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

for halo in halo_marker_dict.keys():
    name = get_halo_names(halo)

    for rs in rs_lis:
    
        diff_list = [] ##initalize list to be put into a histogram
    
        for ion in ion_dict.keys():
            try:
                ds = pd.read_csv(f'./data/halo{halo}/redshift{rs}/stats/{halo}_z{rs}_{ion}_abun_all-model-families_all-clumps.csv', delim_whitespace = True)
                if len(ds['density']) != 0:
                    diff = ds["diff_from_solar_abun"] ##get the data from data tables
                    for value in diff:
                        if pd.isna(value) == False:
                            diff_list.append(value) ##make the list of values
                            print('added!')
                else:
                    print(f'No {ion} in halo {halo} z{rs}')
            except FileNotFoundError:
                print(f'This halo, {halo_names_dict[halo]}, had something wierd going on')
                continue
    kernel = stats.gaussian_kde(diff_list)
    xax = np.linspace(-2.0, 2.0, num = 500)
    ax.plot(xax, kernel.evaluate(xax)) ##make the histogram
    fig.suptitle(f"Diffrence between Solar Abundance and Median Column Density, Redshift {rs}") ##needs a better title but I can't think of one right now
    ax.set_ylabel("relative frequency")
    params = {'mathtext.default': 'regular' }          
    ax.rcParams.update(params)
    ax.set_xlabel("$log(N_{\odot}) - log(N_{med}) [dex]$") ##has to be in dex(order of magnitude) because of how sal the super snake works
    fig.savefig(f'./solar_diff_z{rs}.png', bbox_inches='tight', dpi = 600)
    plt.close()

print('All done!')
