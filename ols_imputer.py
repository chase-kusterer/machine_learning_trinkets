# required packages
import numpy as np # mathematical essentials
import pandas as pd # data science essentials
import statsmodels.api as sm # OLS linear regression with p-values
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # OLS linear regression


# optimal_neighbors function
#########################
# optimal_neighbors
#########################
def optimal_neighbors(X_data,
                      y_data,
                      standardize = True,
                      pct_test=0.25,
                      seed=802,
                      response_type='reg',
                      max_neighbors=20,
                      show_viz=True):
    """
Exhaustively compute training and testing results for KNN across
[1, max_neighbors]. Outputs the maximum test score and (by default) a
visualization of the results.

PARAMETERS
----------
X_data        : explanatory variable data

y_data        : response variable

standardize   : whether or not to standardize the X data, default True

pct_test      : test size for training and validation from (0,1), default 0.25

seed          : random seed to be used in algorithm, default 802

response_type : type of neighbors algorithm to use, default 'reg'
    Use 'reg' for regression (KNeighborsRegressor)
    Use 'class' for classification (KNeighborsClassifier)

max_neighbors : maximum number of neighbors in exhaustive search, default 20

show_viz      : display or surpress k-neigbors visualization, default True
"""    
    
    
    if standardize == True:
        # optionally standardizing X_data
        scaler             = StandardScaler()
        scaler.fit(X_data)
        X_scaled           = scaler.transform(X_data)
        X_scaled_df        = pd.DataFrame(X_scaled)
        X_data             = X_scaled_df



    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size = pct_test,
                                                        random_state = seed)


    # creating lists for training set accuracy and test set accuracy
    training_accuracy = []
    test_accuracy = []
    
    
    # setting neighbor range
    neighbors_settings = range(1, max_neighbors + 1)


    for n_neighbors in neighbors_settings:
        # building the model based on response variable type
        if response_type == 'reg':
            clf = KNeighborsRegressor(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)
            
        elif response_type == 'class':
            clf = KNeighborsClassifier(n_neighbors = n_neighbors)
            clf.fit(X_train, y_train)            
            
        else:
            print("Error: response_type must be 'reg' or 'class'")
        
        
        # recording the training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
    
        # recording the generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))


    # optionally displaying visualization
    if show_viz == True:
        # plotting the visualization
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()
    
    
    # returning optimal number of neighbors
    print(f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy))+1}")
    return test_accuracy.index(max(test_accuracy))+1


# required packages

import pandas as pd # data science essentials
import statsmodels.api as sm # OLS linear regression with p-values
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # OLS linear regression


#########################
# ols_imputer
#########################
# ols_imputer function
def ols_imputer(X_data,
                y_data,
                AVST = True,
                pct_test=0.25,
                seed=802,):
    
    """
Uses stepwise selection to determine which explanatory variables are
significant based on p-values. Outputs list of test set predictions.

PARAMETERS
----------
X_data        : explanatory variable data

y_data        : response variable

AVST          : whether or not to use stepwise selection, default True

pct_test      : test size for training and validation from (0,1), default 0.25

seed          : random seed to be used in algorithm, default 802

response_type : type of neighbors algorithm to use, default 'reg'
    Use 'reg' for regression (KNeighborsRegressor)
    Use 'class' for classification (KNeighborsClassifier)

max_neighbors : maximum number of neighbors in exhaustive search, default 20

show_viz      : display or surpress k-neigbors visualization, default True
"""    

    
    ####################
    # automated variable selection technique (stepwise) 
    ####################
    if AVST == True:
        def stepwise_selection(X, y, 
                             initial_list=[], 
                             threshold_in=0.01, 
                             threshold_out = 0.05, 
                             verbose=True):
            """
Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS

PARAMETERS
----------
X - pandas.DataFrame with candidate features

y - list-like with the target

initial_list - list of features to start with (column names of X)

threshold_in - include a feature if its p-value < threshold_in

threshold_out - exclude a feature if its p-value > threshold_out

verbose - whether to print the sequence of inclusions and exclusions


Returns: list of selected features 

Always set threshold_in < threshold_out to avoid infinite looping.
See https://en.wikipedia.org/wiki/Stepwise_regression for the details"""
            
            included = list(initial_list)
            while True:
                changed=False
                # forward step
                excluded = list(set(X.columns)-set(included))
                new_pval = pd.Series(index=excluded)
                for new_column in excluded:
                    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                    new_pval[new_column] = model.pvalues[new_column]
                best_pval = new_pval.min()
                if best_pval < threshold_in:
                    best_feature = new_pval.argmin()
                    included.append(best_feature)
                    changed=True
                    if verbose:
                        print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                # backward step
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
                # use all coefs except intercept
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max() # null if pvalues is empty
                if worst_pval > threshold_out:
                    changed=True
                    worst_feature = pvalues.argmax()
                    included.remove(worst_feature)
                    if verbose:
                        print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                if not changed:
                    break
            return included



        # running AVST (stepwise_selection)
        stepwise_result = stepwise_selection(X = X_data,
                                             y = y_data)

        X_data = X_data[stepwise_result]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y_data,
        test_size    = pct_test,
        random_state = seed)

    
    ####################
    # OLS linear regression
    ####################
    # instantiating, fitting, and scoring a model object
    lr = LinearRegression()
    lr_fit = lr.fit(X_train, y_train)
    print('Training Score:', lr.score(X_train, y_train).round(4))
    print('Testing Score:',  lr.score(X_test, y_test).round(4))
    print('*' * 40)
    print('Coefficients:', lr.coef_)


    model_df = pd.concat([pd.Series(lr.coef_),
                          pd.Series(stepwise_result)],
                          axis = 1)

    
    return model_df
