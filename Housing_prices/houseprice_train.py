import numpy as np
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import time
import pandas

#load in the data
features_train=np.load('features_train.npy')
features_test=np.load('features_test.npy')
labels_train=np.load('labels_train.npy')
labels_test=np.load('labels_test.npy')

#use cross validation to find suitable values for regularisation parameters

#### USING RIDGE REGRESSION  ######
ridge_model = linear_model.Ridge(alpha=10)
start=time.time()
ridge_model.fit(features_train,labels_train)
end=time.time()
train_time=end-start
labels_pred = ridge_model.predict(features_test)

#create array to store results
length = len(labels_pred)
x = np.zeros((length,4))
#1st column is the saleprice
x[:,0] = labels_test
#second column is the predicted saleprice
x[:,1] = labels_pred

dif = [0]*length
abdif = [0]*length
for i in range(0,length):
	real = labels_test[i]
	pred = labels_pred[i]
	d = abs(real-pred)
	perc = (d/pred)*100
	dif[i] = perc
	abdif[i] = d

#3rd column is absolute error
x[:,2] = abdif
#4th column is the percentage error
x[:,3] = dif
#np.set_printoptions(suppress=True)

#Store the numpy array in a pandas table with column headings
full_preds = pandas.DataFrame(data=x,columns=['Sale price ($)','Predicted sale price ($)','Absolute error ($)','% error'])
#Calculate Mean Absolute Error ($) and Mean percentage error and store in seperate table
mean_abs_error = np.mean(x[:,2])
mean_perc_error = np.mean(x[:,3])
results = pandas.DataFrame(data=np.array([train_time,mean_abs_error,mean_perc_error],ndmin=2),columns=['Training Time (s)','Mean Absolute Error ($)','Mean Percentage Error (%)'])
#save the tables 
results.to_csv(path_or_buf='results.csv',index=False)
full_preds.to_csv(path_or_buf='full_preds.csv',index=False)

#plt.scatter(x[:,0],x[:,1])
#plt.show()

