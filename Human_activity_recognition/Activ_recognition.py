import pandas
import numpy as np
from sklearn import preprocessing
import random
import time

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

#load in the data
train_data = pandas.read_csv('train.csv')
test_data = pandas.read_csv('test.csv')

#seperate into features and labels
features_train = np.array(train_data.loc[:,'tBodyAcc-mean()-X':'angle(Z,gravityMean)'])
labels_train = np.array(train_data.loc[:,'Activity'])

features_test = np.array(test_data.loc[:,'tBodyAcc-mean()-X':'angle(Z,gravityMean)'])
labels_test = np.array(test_data.loc[:,'Activity'])

#need to encode the labels to numbers
le = preprocessing.LabelEncoder()
le.fit(labels_train)
labels_train = le.transform(labels_train)
labels_test = le.transform(labels_test)
enc = OneHotEncoder()
enc.fit(labels_train.reshape(-1, 1))
labels_train_OH = enc.transform(labels_train.reshape(-1, 1))
labels_test_OH = enc.transform(labels_test.reshape(-1, 1))


#####   CREATE A TABLE TO STORE RESULTS  #####
results = pandas.DataFrame(index=['Naive Bayes','Decision Trees','AdaBoost','Random Forest'],columns=['Training Time','Prediction Time','Accuracy'])


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


results.to_csv(path_or_buf='results.csv')

#
print(results)