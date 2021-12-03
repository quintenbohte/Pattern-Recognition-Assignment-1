# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:11:32 2021

@author: quint
"""

################################# PACKAGES #################################
from pathlib import Path
import os

################################ FUNCTIONS #####################
def load_dict(path_to_dict):
    import pickle
    with open(path_to_dict, 'rb') as config_dictionary_file:
     
        # Step 3
        dictionary = pickle.load(config_dictionary_file)
     
        return dictionary

################################# LOAD RESULTS #################################

parent_dir = str(Path(os.path.abspath(__file__)).parents[1])

results_lr_lasso = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_LASSONone.dictionary')
results_mlp = load_dict(parent_dir + '/Data/Output/Results_Neural_NetworkNone.dictionary')
results_svm  = load_dict(parent_dir + '/Data/Output/Results_support_vector_machineNone.dictionary')
results_lr_ink_features = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_ink_feature.dictionary')
results_lr_row_features = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_row_feature.dictionary')
results_lr_both_features = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_both_featutes.dictionary')





