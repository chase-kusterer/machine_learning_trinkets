# required libraries
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def optimal_neighbors(X_train, y_train, X_test, y_test,
                      response_type = 'reg',
                      max_neighbors = 20,
                      show_viz = True):
    """
Exhaustively compute training and testing results for KNN across
[1, max_neighbors]. Outputs the maximum test score and (by default) a
visualization of the results.

PARAMETERS
----------
X_train       : training data for explanatory variables, default X_train

y_train       : training data for response variable, default y_train

X_test        : testing data for explanatory variables, default X_test

y_test        : testing data for response variable, default y_test

response_type : type of neighbors algorithm to use, default 'reg'
    Use 'reg' for regression (KNeighborsRegressor)
    Use 'class' for classification (KNeighborsClassifier)

max_neighbors : maximum number of neighbors in exhaustive search, default 20

show_viz      : display or surpress k-neigbors visualization, default True
"""
    
    # creating lists for training set accuracy and test set accuracy
    training_accuracy = []
    test_accuracy = []


    # building a visualization of 1 to 50 neighbors
    neighbors_settings = range(1, max_neighbors + 1)


    for n_neighbors in neighbors_settings:
        # building the model
        
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


    if show_viz == True:
        # plotting the visualization
        fig, ax = plt.subplots(figsize=(12,8))
        plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()
    
    
    print(f"""The optimal number of neighbors is {test_accuracy.index(max(test_accuracy)) + 1}""")
    return test_accuracy.index(max(test_accuracy)) + 1
