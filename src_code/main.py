import pandas as pd
import numpy as np
from PLSR_modeling import data_extraction, seasonal_data_extraction, random_CV, spatial_CV, leave_one_out_CV,\
    site_extrapolation, cross_PFT_CV, leave_one_PFT_out, cross_sites_PFTs, PFT_extrapolation, random_temporal_CV,\
        temporal_CV,leave_one_season_out_CV,across_sensors
        
dataset = "../1_datasets/Paired leaf traits and leaf spectra dataset.csv"
trait_name = ['Chla+b','Ccar','EWT','LMA']
for tr in trait_name:
    X, y = data_extraction(tr, dataset)
    
    """
    Spatial analysis
    """
    # (1) Random 10 fold cross-validation
    a = random_CV(X,y,tr,10,100)
    # (2) Spatial 10 fold cross-validation
    b = spatial_CV(X,y,tr,10,100)
    # (3) Leave one site out for training
    c = leave_one_out_CV(X,y,tr)
    # (4) Site extrapolation
    d = site_extrapolation(X,y,tr)
    
    """
    PFTs analysis
    """
    # (1) 5 fold cross PFTs validation
    f = cross_PFT_CV(X,y,tr,5,100)
    # (2) Leave one PFT out for training
    g = leave_one_PFT_out(X,y,tr)
    # (3) cross sites for each PFT
    h = cross_sites_PFTs(X,y,tr)
    # (4) PFT extrapolation
    i = PFT_extrapolation(X,y,tr)
    
"""
Temporal analysis
"""
## Saperate seasonal datasets
leaf_spectra, leaf_traits = seasonal_data_extraction(dataset)
ds = ['Dataset#3','Dataset#4','Dataset#8']
for dataset_num in ds:
    if dataset_num == 'Dataset#8':
        trait_name = ['LMA']
    else:
        trait_name = ['Chla+b','Ccar','LMA']  
    for tr in trait_name:
        print(dataset_num,tr)
        traits = leaf_traits[(leaf_traits['Dataset ID']==dataset_num)&(leaf_traits[tr]>0)]

        refl = leaf_spectra.iloc[traits.index]
        refl.reset_index(drop = True,inplace = True)
        traits.reset_index(drop = True,inplace = True)
        
        X = refl
        y = traits
        
        # (1) Random 10 fold cross-validation
        a = random_temporal_CV(X,y,tr,5,dataset_num,100)
        # (2) Spatial 10 fold cross-validation
        b = temporal_CV(X,y,tr,5,dataset_num,100)
        # (3) Leave one site out for training
        c = leave_one_season_out_CV(X,y,tr,dataset_num)
        
## All seasonal datasets together
trait_name = ['Chla+b','Ccar','LMA']
for tr in trait_name:
    print(tr)
    traits = leaf_traits[leaf_traits[tr]>0]
    refl = leaf_spectra.iloc[traits.index]
    refl.reset_index(drop = True,inplace = True)
    traits.reset_index(drop = True,inplace = True)
    
    X = refl
    y = traits
    dataset_num = 'All_datasets'
    # (1) Random 10 fold cross-validation
    aa = random_temporal_CV(X,y,tr,5,dataset_num,100)
    # (2) Spatial 10 fold cross-validation
    bb = temporal_CV(X,y,tr,5,dataset_num,100)
    # (3) Leave one site out for training
    cc = leave_one_season_out_CV(X,y,tr,dataset_num)
    
"""
Across sensors analysis
"""
dataset = "../1_datasets/Paired leaf traits and leaf spectra dataset.csv"
tr = "LMA"
X, y = data_extraction(tr, dataset)
trait = y[y["Dataset ID"]=="Dataset#23"]
refl = X.iloc[trait.index]
refl.reset_index(drop = True, inplace = True)
trait.reset_index(drop = True, inplace = True)

for pft in trait["PFT"].unique():
    trait_data = trait[(trait["PFT"]==pft)]
    refl_data = refl.iloc[trait_data.index]
    refl_data.reset_index(drop = True, inplace = True)
    trait_data.reset_index(drop = True, inplace = True)
    X, y = refl_data, trait_data
    
    a = across_sensors(X,y,tr,pft)