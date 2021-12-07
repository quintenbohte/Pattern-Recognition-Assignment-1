# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:11:32 2021

@author: quint
"""

################################# PACKAGES #################################
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
waitAfterShow = True

################################ FUNCTIONS #####################
def load_dict(path_to_dict):
    import pickle
    with open(path_to_dict, 'rb') as config_dictionary_file:
     
        # Step 3
        dictionary = pickle.load(config_dictionary_file)
     
        return dictionary

def print_heatMap(confusionMatrix, classFrom, classTo):
    class_names=[classFrom,classTo] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    # plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if(waitAfterShow):
        plt.show()

def Accuracy_per_class(Conv_matr):
    
    total_cases = np.sum(Conv_matr, axis = 1)
    
    diagonal = np.diagonal(Conv_matr)
    
    Classes_correctly = pd.DataFrame(diagonal/total_cases)
    
    Classes_correctly['classes'] = np.arange(0,10,1)
    
    return Classes_correctly

def Mistakes_between_two_digits(confusion_matrix):
    
    pairs_list = []
    mistakes_list = []
    
    for i in range(10):   #row
        for j in range(10):    #column
            if i==j:
                continue
            
            mistakes = confusion_matrix[i,j] + confusion_matrix[j,i]
            
            pairs_list.append((i,j))
            mistakes_list.append(mistakes)
            
            df = pd.DataFrame({'pairs':pairs_list, 'mistakes':mistakes_list })
            
    return df

#%%


################################# LOAD RESULTS #################################

parent_dir = str(Path(os.path.abspath(__file__)).parents[1])

results_lr_lasso = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_LASSONone.dictionary')
results_mlp = load_dict(parent_dir + '/Data/Output/Results_Neural_NetworkNone.dictionary')
results_svm  = load_dict(parent_dir + '/Data/Output/Results_support_vector_machineNone.dictionary')
results_lr_ink_features = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_ink_feature.dictionary')
results_lr_row_features = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_row_feature.dictionary')
results_lr_both_features = load_dict(parent_dir + '/Data/Output/Results_Logistic_regression_both_featutes.dictionary')


##################################### MAKE HEAT MAPS #############################


Conv_matr_svm = results_svm['Confusion matrix']
Conv_matr_mlp = results_mlp['Confusion matrix']
Conv_matr_lr = results_lr_lasso['Confusion matrix']


# print_heatMap(Conv_matr_svm, 1, 9)
print_heatMap(Conv_matr_mlp, 1, 9)


##################################### ACCURACY PER CLASS #############################

acc_per_class_svm = Accuracy_per_class(Conv_matr_svm)
acc_per_class_mlp = Accuracy_per_class(Conv_matr_mlp)
acc_per_class_lr = Accuracy_per_class(Conv_matr_lr)


##################################### MISTAKES PER PAIR #############################

lr_pairs = Mistakes_between_two_digits(Conv_matr_lr)
svm_pairs = Mistakes_between_two_digits(Conv_matr_svm)
mlp_pairs = Mistakes_between_two_digits(Conv_matr_mlp)
         
        










