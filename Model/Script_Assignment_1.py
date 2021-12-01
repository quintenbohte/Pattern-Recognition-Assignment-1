# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:03:13 2021

@author: quint
"""

########################################## LOADING PACKAGES ###############################################

from numpy.lib.index_tricks import diag_indices
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier


skipHistogram = False
smallList = False
waitAfterShow = True

## TODO: Maybe we can look at an heatmap of all the combined numbers and filter the input data

#%%
########################################## FUNCTIONS ###############################################

def conf_matrix_two_classes(class1, class2, calculated_confusion_matrix):
    
    confusion_matrix_two_classes3 = np.zeros((2,2))
    
    confusion_matrix_two_classes3[0,0] = calculated_confusion_matrix[class1,class1]
    confusion_matrix_two_classes3[0,1] = calculated_confusion_matrix[class1,class2]
    confusion_matrix_two_classes3[1,0] = calculated_confusion_matrix[class2,class1]
    confusion_matrix_two_classes3[1,1] = calculated_confusion_matrix[class2,class2]
    
    return confusion_matrix_two_classes3
    

def print_heatMap(confusionMatrix, classFrom, classTo):
    class_names=[classFrom,classTo] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(confusionMatrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if(waitAfterShow):
        plt.show()
    #Text(0.5,257.44,'Predicted label')

def logistic_regression(xVar, yVar):
    ########################################### CONSTRUCT TEST AND TRAINING SET ######################
    # This splits every entry in here in 2 a training and a test set
    X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2, random_state=42)


    ########################################### FIT MULTINOMIAL LOGISTIC REGRESSION MODEL ######################

    #define the model
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    #fit the model
    lr.fit(X_train, y_train)

    #make predictions
    y_pred = lr.predict(X_test)

    #construct confusion matrix
    calculated_confusion_matrix = confusion_matrix(y_test, y_pred)

    #construct transition matrix for two classes. From this we can see how well the model is able to
    #distuingish these two classes
    confusion_matrix_two_classes = conf_matrix_two_classes(0,5, calculated_confusion_matrix)
    print(confusion_matrix_two_classes)

    print_heatMap(calculated_confusion_matrix, 1, 9)

            

def get_best(model, param, X_train, y_train):
    GridS = GridSearchCV(model, param, cv=5, n_jobs=-1, scoring='accuracy')
    GridS.fit(X_train, y_train)
    return GridS

def logistic_regression_LASSO(params, X_train, X_test, y_train, y_test):
    
    #find best value for lambda using cross validation. For this we use the grid_search method from sklearn
    lr_param = params
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
    grid_search = get_best(model, lr_param, X_train, y_train)
    best_C = grid_search.best_params_['C']
    
    #fit final model, using the best value for C found using cross-validation
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', C = best_C, max_iter = 10000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    #compute performances
    accuracy = accuracy_score(y_test, lr_pred)
    precision = precision_score(y_test, lr_pred, average='micro')
    recall = recall_score(y_test, lr_pred, average='micro')

    
    print('LOGISTIC REGRESSION LASSO')
    print('--Accuracy:', accuracy)
    print('--Precision:', precision)
    print('--Recall:', recall)
    print('')
    
    return accuracy

def SVM(params, X_train, X_test, y_train, y_test):
    
    #find best value for lambda using cross validation. For this we use the grid_search method from sklearn
    svm_param = params
    model = svm.SVC(decision_function_shape='ovo')
    grid_search = get_best(model, svm_param, X_train, y_train)
    best_C = grid_search.best_params_['C']
    best_kernel = grid_search.best_params_['kernel']
    
    #fit final model, using the best value for C found using cross-validation
    svm_model = svm.SVC(decision_function_shape='ovo', kernel = best_kernel, C = best_C)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    
    #compute performances
    accuracy = accuracy_score(y_test, svm_pred)
    precision = precision_score(y_test, svm_pred, average='micro')
    recall = recall_score(y_test, svm_pred, average='micro')

    print('Support Vector Machine')
    print('-- best C:', best_C)
    print('-- best Kernel:', best_kernel)
    print('--Accuracy:', accuracy)
    print('--Precision:', precision)
    print('--Recall:', recall)
    print('')
    
    
    return accuracy

#%%
"INTRODUCTION QUESTIONS"

########################################## LOAD DATA ###############################################

parent_dir = str(Path(os.path.abspath(__file__)).parents[1])
mnist_data = pd.read_csv(parent_dir + '\data\mnist.csv').values

########################################## EXAMPLE IMAGE OF DATA ###############################################

labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]

# For debugging purposes (Histogram generation takes a long time)
if(smallList):
    amount = 1000
    labels = labels[0:amount]
    digits = digits[0:amount]

img_size = 28
plt.figure()
plt.imshow(digits[0].reshape(img_size, img_size))
if(waitAfterShow):
    plt.show()


#%%
"QUESTION 1"
"-- Do an exploratory data analysis."
########################################## EXPLORATORY ANALYSIS ###############################################

#Compute and plot Class Distribution
labels_df = pd.DataFrame({'labels':labels})
labels_df = labels_df.groupby(['labels']).size().reset_index(name = 'count')
labels_df['percentage'] = labels_df['count']/42000
plt.figure()
plt.bar(np.arange(0,10), labels_df['count'] )
plt.xticks(np.arange(0,10))
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
if(waitAfterShow):
    plt.show()


#%%
"QUESTION 2"
"-- Derive the ink feature."

"-- Compute the mean and std of the ink feature for each class. Can we distuinghish classes based on the"
"   ink feature means of each class?"

"-- Fit a multinomial logistic regression model using only the ink feature. Use the confusion matrix to see"
"   what classes can be distuinished good from each other and which can't, using only the ink feature."
########################################### COMPUTE INK FEATURE ####################################

ink_pixels = np.count_nonzero(digits, axis = 1)
total_ink = np.sum(digits, axis = 1)
average_ink_per_pixel = total_ink/ink_pixels

########################################### CONSTRUCT DATAFRAME FOR ANALYSIS ######################

df = pd.DataFrame({'labels': labels, 'Ink_Quantity': total_ink})

################################# COMPUTE MEAN AND STD OF INK QUANTITY FOR EACH CLASS ######################

df_mean_std = df.groupby('labels').agg({'Ink_Quantity': ['mean', 'std']})
df_mean_std = df_mean_std.reset_index('labels')
df_mean_std.columns = ['labels', 'mean', 'std']

plt.figure()
plt.bar(np.arange(0,10), df_mean_std['mean'])
plt.errorbar(np.arange(0,10), df_mean_std['mean'], yerr=df_mean_std['std'], fmt='none', ecolor='black', barsabove=True)
plt.xticks(np.arange(0,10))
plt.xlabel('Class')
plt.ylabel('Mean')
plt.title('Class Ink Quantity Mean with Standard Deviation')
if(waitAfterShow):
    plt.show()

plt.figure()
plt.bar(np.arange(0,10), df_mean_std['std'])
plt.xticks(np.arange(0,10))
plt.xlabel('Class')
plt.ylabel('Standard Deviation')
plt.title('Class Ink Quantity Standard Deviation')
if(waitAfterShow):
    plt.show()

########################################### SCALE INK QUANTITY FEATURE ###############################

df['Ink_Quantity_scaled'] = scale(df['Ink_Quantity']).reshape(-1, 1)
mean_of_scaled_ink_quantity = df['Ink_Quantity_scaled'].mean()
std_of_scaled_ink_quantity = df['Ink_Quantity_scaled'].std()

print('mean of scaled ink quantity is', np.round(mean_of_scaled_ink_quantity,2), 'this should be 0' )
print('std of scaled ink quantity is', np.round(std_of_scaled_ink_quantity,2), 'this should be 1' )

logistic_regression(df[['Ink_Quantity_scaled']], df['labels'])


#%%
"QUESTION 3"
"-- Think of a new feature and do the same analysis as done above"




rowHistArray = []
columnHistArray = []
imageCounter = 0


# every 28th array[::28]
# Make a row and column histogram for each image
for image in digits:
    # counter initializations
    counter = 0
    tempSum = 0
    colCounter = 0

    # Array initializations to 0
    tempArray = np.zeros((28))
    tempColArray = np.zeros((28))
    
    # Histogram generation
    if(skipHistogram):
        rowHistArray.append(tempArray)
        columnHistArray.append(tempColArray)
        continue
    for x in range(784):
        # Row histogram generation
        if(x % 28 == 0 and x != 0):
            counter += 1
        tempArray[counter] += image[x]

        # Column histogram generation
        tempColArray[colCounter] += image[x]
        colCounter += 1
        if(colCounter >= 28):
            colCounter = 0

    rowHistArray.append(tempArray)
    columnHistArray.append(tempColArray)
    if(imageCounter % 100 == 0):
        print("Done image ", imageCounter)
    imageCounter += 1
    


################################# COMPUTE MEAN AND STD OF INK QUANTITY FOR EACH CLASS ######################

rowFeatureArrays = []
colFeatureArrays = []

for x in range(28):
    rowFeatureArrays.append([])
    colFeatureArrays.append([])

# We make the assumption here that the length of both the arrays are the same
# Make feature list
for x in range(len(rowHistArray)):
    currentRow = rowHistArray[x]
    currentCol = columnHistArray[x]

    for y in range(28):
        rowFeatureArrays[y].append(currentRow[y])
        colFeatureArrays[y].append(currentCol[y])

# Doing this seperate for readability of the output
dfRow = pd.DataFrame({'labels': labels})
dfCol = pd.DataFrame({'labels': labels})
dfCombined = df

for x in range(28):
    ########################################### CONSTRUCT DATAFRAME FOR ANALYSIS ######################

    rowName = 'Row_' + str(x)

    dfRow[rowName] = rowFeatureArrays[x]

    

    dfRow_mean_std = dfRow.groupby('labels').agg({rowName: ['mean', 'std']})
    dfRow_mean_std = dfRow_mean_std.reset_index('labels')
    dfRow_mean_std.columns = ['labels', 'mean', 'std']

    ########################################### SCALE INK QUANTITY FEATURE ###############################
    scaledRowName = rowName + '_histogram_scaled'
    dfRow[scaledRowName] = scale(dfRow[rowName]).reshape(-1, 1)
    dfCombined[scaledRowName] = scale(dfRow[rowName]).reshape(-1, 1)
    mean_of_scaled_Row_histogram = dfRow[scaledRowName].mean()
    std_of_scaled_Row_histogram = dfRow[scaledRowName].std()

    meanString = 'mean of scaled '+ rowName + ' is'
    stdString = 'std of scaled '+ rowName + ' is'

    print(meanString, np.round(mean_of_scaled_Row_histogram,2), 'this should be 0' )
    print(stdString, np.round(std_of_scaled_Row_histogram,2), 'this should be 1' )


for x in range(28):
    ########################################### CONSTRUCT DATAFRAME FOR ANALYSIS ######################


    colName = 'Col_' + str(x)

    dfCol[colName] = colFeatureArrays[x]

    

    dfCol_mean_std = dfCol.groupby('labels').agg({colName: ['mean', 'std']})
    dfCol_mean_std = dfCol_mean_std.reset_index('labels')
    dfCol_mean_std.columns = ['labels', 'mean', 'std']

    ########################################### SCALE INK QUANTITY FEATURE ###############################
    scaledColName = colName + '_histogram_scaled'
    dfCol[scaledColName] = scale(dfCol[colName]).reshape(-1, 1)
    dfCombined[scaledColName] = scale(dfCol[colName]).reshape(-1, 1)
    mean_of_scaled_Col_histogram = dfCol[scaledColName].mean()
    std_of_scaled_Col_histogram = dfCol[scaledColName].std()

    meanString = 'mean of scaled '+ colName + ' is'
    stdString = 'std of scaled '+ colName + ' is'

    print(meanString, np.round(mean_of_scaled_Col_histogram,2), 'this should be 0' )
    print(stdString, np.round(std_of_scaled_Col_histogram,2), 'this should be 1' )






feature_cols_row = ['Row_0_histogram_scaled','Row_1_histogram_scaled','Row_2_histogram_scaled','Row_3_histogram_scaled',
    'Row_4_histogram_scaled','Row_5_histogram_scaled','Row_6_histogram_scaled','Row_7_histogram_scaled',
    'Row_8_histogram_scaled','Row_9_histogram_scaled','Row_10_histogram_scaled','Row_11_histogram_scaled',
    'Row_12_histogram_scaled','Row_13_histogram_scaled','Row_14_histogram_scaled','Row_15_histogram_scaled',
    'Row_16_histogram_scaled','Row_17_histogram_scaled','Row_18_histogram_scaled','Row_19_histogram_scaled',
    'Row_20_histogram_scaled','Row_21_histogram_scaled','Row_22_histogram_scaled','Row_23_histogram_scaled',
    'Row_24_histogram_scaled','Row_25_histogram_scaled','Row_26_histogram_scaled','Row_27_histogram_scaled']


feature_cols_col = ['Col_0_histogram_scaled','Col_1_histogram_scaled','Col_2_histogram_scaled','Col_3_histogram_scaled',
    'Col_4_histogram_scaled','Col_5_histogram_scaled','Col_6_histogram_scaled','Col_7_histogram_scaled',
    'Col_8_histogram_scaled','Col_9_histogram_scaled','Col_10_histogram_scaled','Col_11_histogram_scaled',
    'Col_12_histogram_scaled','Col_13_histogram_scaled','Col_14_histogram_scaled','Col_15_histogram_scaled',
    'Col_16_histogram_scaled','Col_17_histogram_scaled','Col_18_histogram_scaled','Col_19_histogram_scaled',
    'Col_20_histogram_scaled','Col_21_histogram_scaled','Col_22_histogram_scaled','Col_23_histogram_scaled',
    'Col_24_histogram_scaled','Col_25_histogram_scaled','Col_26_histogram_scaled','Col_27_histogram_scaled']

feature_cols_ink = ['Ink_Quantity_scaled']


xVarRow = dfRow[feature_cols_row]
xVarCol = dfCol[feature_cols_col]
xVarRowAndCol = dfCombined[feature_cols_row + feature_cols_col] 
yVarRow = dfRow['labels'] 
yVarCol = dfCol['labels'] 
yVarRowAndCol = dfCombined['labels'] 


logistic_regression(xVarRow, yVarRow)
#logistic_regression(xVarCol, yVarCol)
#logistic_regression(xVarRowAndCol, yVarRowAndCol)


#%%
"QUESTION 4"
"-- Fit a multinomial logistic regression model on both features and see if it out performs the model"
"   that used only the ink feature"

xVarRowAndInk = dfCombined[feature_cols_row + feature_cols_ink]
yVarRowAndInk = dfCombined['labels'] 
logistic_regression(xVarRowAndInk, yVarRowAndInk)



#%%
"QUESTION 5"
"-- Use each pixel as a feature and fit the following models:"
"   1: Logistic regression with Lasso Penalty"
"   2: Support Vector Machine"
"   3: Multilayer Perceptions Classifier (MLP Classifier"

"-- Use cross validation to select the optimal paramerer values and use these values to fit a final model"


#################################### CONSTRUCT DATAFRAME WITH EACH PIXEL AS FEATURE ######################
#construct train and test set
X_train_pix, X_test_pix, y_train_pix, y_test_pix = train_test_split(digits, labels, test_size=0.88, random_state=42)


#################################### FIT LOGISTIC REGRESSION LASSO ######################
#dictionary with the different parameters we want to consider and the value range. 
lr_param = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}

#fit models for all parameters and pick best parameter values and use these to fit final model.
lr_accuracy = logistic_regression_LASSO(lr_param, X_train_pix, X_test_pix, y_train_pix, y_test_pix)


#################################### FIT SUPPORT VECTOR MACHINE ######################

#dictionary with the different parameters we want to consider and the value range. 
#Another parameter we could include is the gamma parameter. 
svm_param = {'kernel':['linear', 'poly', 'rbf'], 'C':[0.1,0.5,1,5,10,50,100]}

#fit models for all parameters and pick best parameter values and use these to fit final model.
svm_accuracy = SVM(svm_param,X_train_pix, X_test_pix, y_train_pix, y_test_pix)

#################################### FIT SUPPORT VECTOR MACHINE ######################

#%%

"QUESTION 6"
"-- What models performs best on the classification problem?"
"-- Do an statistical test to see whether there are significant differences between the model accuracies"

#I would suggest a 































