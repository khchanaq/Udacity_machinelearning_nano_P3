#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:02:44 2017

@author: khchanaq
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import visuals as vs

# Import supplementary visualizations code visuals.py

# Pretty display for notebooks
#%matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    

# Display a description of the dataset
display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [3,15,100]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.copy()
new_data.drop(['Milk'], axis = 1, inplace = True)
X = new_data
y = data.iloc[:,1]

# TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#####################################################################################################

outliers_entry = pd.DataFrame()
    
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    #print Q1
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    #print Q3
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = Q3 - Q1
    #print step
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    outliers_entry = pd.concat([outliers_entry, (log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])])

#Drop outliers with more than 2 features which is outlining
outliers_entry.drop_duplicates(keep = False, inplace = True)

outliers = outliers_entry.index.values
# OPTIONAL: Select the indices for data points you wish to remove
    
# Remove the outliers, if any were specified
# Remove the data point with only 1 feature which is outlier
good_data = log_data.copy()
good_data = good_data.drop(log_data.index[outliers])

#####################################################################################################
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
good_data = pca.fit_transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = log_samples.copy()
pca_samples = pca.transform(pca_samples)

# Generate PCA results plot
pca_results = vs.pca_results(pd.DataFrame(pca_samples), pca)

#####################################################################################################

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

#####################################################################################################

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components = 2)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.fit_transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(pca_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
#####################################################################################################

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

# Create a DataFrame for the pca_samples
good_data = pd.DataFrame(good_data)

#####################################################################################################

# Create a biplot
vs.biplot(good_data, reduced_data, pca)

#####################################################################################################

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

silhouettescore = []
init_cluster = 2;
for i in range(init_cluster,20):
    clusterer = GaussianMixture(n_components = i, random_state = 0)
    clusterer.fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    silhouettescore.append(silhouette_score(reduced_data, preds, random_state = 0))

best_cluster = silhouettescore.index(max(silhouettescore)) + init_cluster

clusterer = GaussianMixture(n_components = best_cluster, random_state = 0)
clusterer.fit(reduced_data)
preds = clusterer.predict(reduced_data)

centers = clusterer.means_
pca_samples = pd.DataFrame(pca_samples).values
sample_preds = clusterer.predict(pca_samples)

score = max(silhouettescore)

#####################################################################################################

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)

#####################################################################################################

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

outliers = outliers.values
