# this script will attemp to detect card fraud with naive bayes, svm, decision treed.
#	can later do it with k-clustering, eta boosting, random forest once ive learnt them.

#The data is skewed and so would give a high accuracy even if it wasnt detecting fraud since it would detect the non-fraudalent transactions.

# There fore use an equal number of fraudalent and non-fraudalent transactions in the test data.

#use 10% of the fraudalent transactions as test data and then an equal number of non-fraudalent transactions


import pandas
import numpy as np
import random
import Stef_nn as snn
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load in the data
data = pandas.read_csv('creditcard.csv')

#seperate data that is fraudalent
fraud_data = data.loc[data['Class'] == 1]
# There are 492 fraudalent transactions

#seperate data that is non-fraudalent
legal_data = data.loc[data['Class']==0]
# There are 284,315 non-fraudalent transactions (the data is largely skewed)


#The data are sorted by transaction time, so randomly sort each data set and take the first 50 data of each for testing data

#create blank data frames to store the test fraud data
fraud_test = pandas.DataFrame(columns=data.columns.values)

#create blank dataframe to store test legal data
legal_test = pandas.DataFrame(columns=data.columns.values)

#loop through and remove a random 50 rows from fraud_data and store them in fraud_test
for i in range(0,50):
	rows,cols = fraud_data.shape
	#generate a random number between 0 and the number of rows
	num = random.randint(0,(rows-1))
	#add the row of that number to fraud_test
	row_values = fraud_data.iloc[num]
	fraud_test = fraud_test.append(row_values, ignore_index=True)
	#remove that row from fraud_data
	fraud_data = fraud_data.drop(fraud_data.index[num])

#loop through and remove a random 50 rows from legal_data and store them in legal_test
for i in range(0,50):
	rows,cols = legal_data.shape
	#generate a random number between 0 and the number of rows
	num = random.randint(0,(rows-1))
	#add the row of that number to legal_test
	row_values = legal_data.iloc[num]
	legal_test = legal_test.append(row_values, ignore_index=True)
	#remove that row from legal_data
	legal_data = legal_data.drop(legal_data.index[num])



# append the legal and fraud test data together for a test data set
test_data = legal_test.append(fraud_test, ignore_index=True)

# append the fraud and legal train data together for a training data set
train_data = legal_data.append(fraud_data, ignore_index=True)

#seperate both datasets into features and labels
labels_test = np.array(test_data.loc[:,'Class'],ndmin=2)
labels_test = labels_test.T
features_test = np.array(test_data.loc[:,'V1':'V28'])

labels_train = np.array(train_data.loc[:,'Class'],ndmin=2)
labels_train = labels_train.T
features_train = np.array(train_data.loc[:,'V1':'V28'])


#####   CREATE A TABLE TO STORE RESULTS  #####
results = pandas.DataFrame(index=['Naive Bayes','Decision Trees','K Nearest Neighbours','AdaBoost','Random Forest','GA Neural Network'],columns=['Training Time','Prediction Time','Accuracy'])


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


####  USING K NEAREST NEIGHBOURS  ######

#create the classifier
clf = KNeighborsClassifier(n_neighbors=50)
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
KNNacc = accuracy_score(labels_test,labels_predicted)

results.set_value('K Nearest Neighbours','Training Time',train_time)
results.set_value('K Nearest Neighbours','Prediction Time',pred_time)
results.set_value('K Nearest Neighbours','Accuracy',KNNacc)

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

####   USING MY NEURAL NETWORK  ####
start = time.time()
nets,fitness = snn.nn_train_stoch_bank_fraud(features_train,labels_train,1,[6],52,100,100,20)
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
		real = labels_test[i,j]
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

results.to_csv(path_or_buf='results.csv')

print(results)