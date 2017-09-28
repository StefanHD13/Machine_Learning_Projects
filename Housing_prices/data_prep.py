import pandas
import numpy as np
import random
from sklearn.decomposition import PCA


#prep housing price data into training and testing features and labels

#load in the data
data = pandas.read_csv('train.csv')

#this function converts all columns with the category or object datatype into dummy variables (one hot encoding)
data = pandas.get_dummies(data)

#This function fills all NA values with the average from that column
data = data.fillna(data.mean())

#convert into labels and features
features = np.concatenate((np.array(data.loc[:,'MSSubClass':'YrSold']),np.array(data.loc[:,'MSZoning_C (all)':'SaleCondition_Partial'])),1)
labels = np.array(data.loc[:,'SalePrice'])


#seperate out 10% of the samples into testing features and labels
rows,cols = np.shape(features)
test_samples = round(rows/10)

#normalise each column to between 0 and 1
normal = features.max(axis=0)
features = features/normal

#select a random 10% of the rows for testing
random_nums = random.sample(range(0,rows),test_samples)

features_test = features[random_nums,:]
labels_test = labels[random_nums]

#remove those rows to form the training sets
features_train = np.delete(features,random_nums,0)
labels_train = np.delete(labels,random_nums,0)

#could maybe do some pca analysis here
pca = PCA(n_components=50)
pca.fit(features_train)

exit()
#save each of the datasets to be used in the training script
np.save('features_test.npy',features_test)
np.save('labels_test.npy',labels_test)
np.save('features_train.npy',features_train)
np.save('labels_train.npy',labels_train)