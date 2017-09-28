#compare the accuracy of using just a count vectorizer or a tfidf representation,
#	as well as how many of the k best features affect the accuracy and time.

import pandas
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load in the data and store in lists
features_train_store = [0] * 10
features_test_store = [0] * 10
#load labels
labels_train = list(np.load('labels_train.npy'))
labels_test = list(np.load('labels_test.npy'))
#load training features
features_train_store[0] = np.load('features_count_train_k10.npy')
features_train_store[1] = np.load('features_count_train_k50.npy')
features_train_store[2] = np.load('features_count_train_k100.npy')
features_train_store[3] = np.load('features_count_train_k200.npy')
features_train_store[4] = np.load('features_tfidf_train_k10.npy')
features_train_store[5] = np.load('features_tfidf_train_k50.npy')
features_train_store[6] = np.load('features_tfidf_train_k100.npy')
features_train_store[7] = np.load('features_tfidf_train_k200.npy')
#load testing features
features_test_store[0] = np.load('features_count_test_k10.npy')
features_test_store[1] = np.load('features_count_test_k50.npy')
features_test_store[2] = np.load('features_count_test_k100.npy')
features_test_store[3] = np.load('features_count_test_k200.npy')
features_test_store[4] = np.load('features_tfidf_test_k10.npy')
features_test_store[5] = np.load('features_tfidf_test_k50.npy')
features_test_store[6] = np.load('features_tfidf_test_k100.npy')
features_test_store[7] = np.load('features_tfidf_test_k200.npy')

features_train_store[8] = np.load('features_count_train.npy')
features_train_store[9] = np.load('features_tfidf_train.npy')

features_test_store[8] = np.load('features_count_test.npy')
features_test_store[9] = np.load('features_tfidf_test.npy')

#create a table to store results
results = pandas.DataFrame(index=['count_k10','count_k50','count_k100','count_k200','tfidf_k10','tfidf_k50','tfidf_k100','tfidf_k200','count_all','tfidf_all'],columns=['training_time','accuracy'])

#loop through and calculate accuracy of naive bayes classifier for each dataset and store in results table
for i in range(0,10):
	
	#create classifier
	clf = GaussianNB()
	#fit the classifier
	start = time.time()
	clf.fit(features_train_store[i], labels_train)
	end = time.time()
	train_time = end-start
	# predict the test data
	labels_predicted = clf.predict(features_test_store[i])
	# determine the accuracy
	acc = accuracy_score(labels_test,labels_predicted)
	#store results
	results.set_value(i,0,train_time,takeable=True)
	results.set_value(i,1,acc,takeable=True)

results.to_csv(path_or_buf='features_results.csv')
print(results)
