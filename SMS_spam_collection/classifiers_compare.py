import pandas
import numpy as np
import random
import time

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import Stef_nn as snn

features_train = np.load('features_count_train_k50.npy')
features_test = np.load('features_count_test_k50.npy')
#load labels
labels_train = list(np.load('labels_train.npy'))
labels_test = list(np.load('labels_test.npy'))
#load labels for neural network
labels_train_nn = np.atleast_2d(np.load('labels_train.npy'))
labels_train_nn = labels_train_nn.T
labels_test_nn = np.atleast_2d(np.load('labels_test.npy'))
labels_test_nn = labels_test_nn.T


#####   CREATE A TABLE TO STORE RESULTS  #####
results = pandas.DataFrame(index=['Naive Bayes','Decision Trees','AdaBoost','Random Forest','GA Neural Network'],columns=['Training Time','Prediction Time','Accuracy'])


#####   USING NAIVE BAYES   #######

#create classifier
clf = GaussianNB()
#fit the classifier
start = time.time()
clf.fit(features_train, labels_train)
end = time.time()
train_time = end-start
# predict the test data
start = time.time()
labels_predicted = clf.predict(features_test)
end = time.time()
pred_time = end-start
# determine the accuracy
NBacc = accuracy_score(labels_test,labels_predicted)
#store results
results.set_value('Naive Bayes','Training Time',train_time)
results.set_value('Naive Bayes','Prediction Time',pred_time)
results.set_value('Naive Bayes','Accuracy',NBacc)



#### USING DECISION TREES   ######

#create the classifier
clf = tree.DecisionTreeClassifier(min_samples_split=50)
#fit the classifier
start = time.time()
clf = clf.fit(features_train, labels_train)
end = time.time()
train_time = end-start
# predict the test data
start = time.time()
labels_predicted = clf.predict(features_test)
end = time.time()
pred_time = end-start
# determine the accuracy
DTacc = accuracy_score(labels_test,labels_predicted)

results.set_value('Decision Trees','Training Time',train_time)
results.set_value('Decision Trees','Prediction Time',pred_time)
results.set_value('Decision Trees','Accuracy',DTacc)


####  USING ADABOOST   #######

#create the classifier
clf = AdaBoostClassifier()
#fit the classifier
start = time.time()
clf = clf.fit(features_train, labels_train)
end = time.time()
train_time = end-start
# predict the test data
start = time.time()
labels_predicted = clf.predict(features_test)
end = time.time()
pred_time = end-start
# determine the accuracy
ADABacc = accuracy_score(labels_test,labels_predicted)

results.set_value('AdaBoost','Training Time',train_time)
results.set_value('AdaBoost','Prediction Time',pred_time)
results.set_value('AdaBoost','Accuracy',ADABacc)



####   USING RANDOM FOREST   ######

#create the classifier
clf = RandomForestClassifier()
#fit the classifier
start = time.time()
clf = clf.fit(features_train, labels_train)
end = time.time()
train_time=end-start
# predict the test data
start = time.time()
labels_predicted = clf.predict(features_test)
end = time.time()
pred_time=end-start
# determine the accuracy
RFacc = accuracy_score(labels_test,labels_predicted)

results.set_value('Random Forest','Training Time',train_time)
results.set_value('Random Forest','Prediction Time',pred_time)
results.set_value('Random Forest','Accuracy',RFacc)


#### USING GA NEURAL NETOWRK ####
start = time.time()
nets,fitness = snn.nn_train_stoch_bank_fraud(features_train,labels_train_nn,1,[6],52,100,100,20)
end = time.time()
train_time=end-start

rows,cols = np.shape(nets)
#get best row and put it into a list
index = np.argmin(nets[:,cols-1])
x = nets[index,:(cols-1)]
weights = x.tolist()

start = time.time()
labels_pred = snn.nn_predict(features_test,weights,[6,1])
end = time.time()
pred_time=end-start

rows,cols = np.shape(labels_pred)
correct_count = 0
total_count =0
for i in range(0,rows):
	for j in range(0,cols):
		pred = labels_pred[i,j]
		real = labels_test_nn[i,j]
		total_count = total_count+1

		if pred >= 0.5:
			pred = 1
		elif pred < 0.5:
			pred = 0

		if pred==real:
			correct_count = correct_count+1

accuracy = (correct_count/total_count)

results.set_value('GA Neural Network','Training Time',train_time)
results.set_value('GA Neural Network','Prediction Time',pred_time)
results.set_value('GA Neural Network','Accuracy',accuracy)

results.to_csv(path_or_buf='class_results.csv')
#
print(results)