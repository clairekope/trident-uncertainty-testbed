# coding: utf-8
import yt
import trident
import salsa
import pandas as pd
import pdb

ds = yt.load("/mnt/research/galaxies-REU/sims/FOGGIE/halo_002392/nref11c_nref9f/RD0020/RD0020")
abun_tab_args = {"filename":"cgm_abundances.txt",
                 "select_rows":1,
                 "ratios":False}

center = ds.arr([23876.757358761424, 23842.452527236022, 22995.717805638298], 'kpc')
other_fields = ['density','temperature','metallicity']

# Make rays if they do not already exist
# !!!! IMPORTANT !!!!
# the ion list for generate_lrays should ONLY consist of 'H I' and the fields list
# MUST contain metallicity. If the ion list contains any metals yt will not let
# the fields be re-calculated using different abundance tables
if not salsa.utils.check_rays('test_rays/', 1, []):
    salsa.generate_lrays(ds, center.to('code_length'), 1, 15, ion_list=['H I'], fields=other_fields, out_dir='test_rays')

# some rad science fails?
#abundances = trident.some_rad_science(**abun_tab_args)

abun = pd.read_csv('cgm_abundances.txt', header=0, delimiter=' ')
abundances = abun.iloc[0].to_dict()

# Solar abundances
print("USING SOLAR ABUNDANCES")
absext = salsa.AbsorberExtractor(ds, "test_rays/ray0.h5", ion_name="O VI", calc_missing=True)
df_ovi = salsa.get_absorbers(absext, ["test_rays/ray0.h5"], method='spice', fields=[])
print(df_ovi)

# Modified abudances
print("USING NON-SOLAR ABUNDANCES")
absext_mod = salsa.AbsorberExtractor(ds, "test_rays/ray0.h5", ion_name="O VI", 
                                     abundance_table=abundances, calc_missing=True)
df_ovi_mod = salsa.get_absorbers(absext_mod, ["test_rays/ray0.h5"], method='spice', fields=[])
print(df_ovi_mod)

