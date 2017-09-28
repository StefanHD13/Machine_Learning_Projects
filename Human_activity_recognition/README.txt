##################################################################
################## HUMAN ACTIVITY RECOGNITION ####################
##################################################################

##################################################################
SUMMARY:

-The Human Activity Recognition database was built from the record
 ings of 30 study participants performing activities of daily livi
 ng (ADL) while carrying a waist-mounted smartphone with embedded 
 inertial sensors.
-Various classifiers were fit to a training dataset in order to co
 mpare their training time.
-The classifiers were used to identify one of six activities perfo
 rmed in each of the test data, comparing their accuracy.
##################################################################

##################################################################
FILES:

train.csv,test.csv - The Human Activity Recognition database was 
					 built from the recordings of 30 study 
					 participants performing activities of daily 
					 living (ADL) while carrying a waist-mounted 
					 smartphone with embedded inertial sensors. 
					 The experiments were carried out with a group
					 of 30 volunteers within an age bracket of 
					 19-48 years. Each person performed six 
					 activities (WALKING, WALKING_UPSTAIRS, 
					 WALKING_DOWNSTAIRS, SITTING, STANDING, 
					 LAYING).

		   			 The dataset was obtained from: 
https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones/data

Activ_recognition.py - This python script first loads the testing
					   and training data. The labels are given as
					   strings (eg WALKING, STANDING) and so I 
					   then encoded them to numerical values.
					   Various classifiers (Naive Bayes,Decision 
					   Trees,AdaBoost,Random Forest) are then fit
					   to the data. The resulting training times,
					   prediction times, and prediction 
					   accuracies can be found in results.csv,
					   which can be viewed by running results.py.

results.csv - A table containing the results of 
			  Activ_recognition.py. The table can be easily 
			  viewed by running the results.py script.

results.py - Python script which displays the contents of 
			 results.csv when run.		   
##################################################################