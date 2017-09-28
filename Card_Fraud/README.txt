##################################################################
###################### C A R D   F R A U D #######################
##################################################################

##################################################################
SUMMARY:

-The dataset contains transactions made by credit cards in Septemb
 er 2013 by european cardholders.
-The dataset contains 492 frauds out of 284,807 transactions, and 
 so it is largely skewed.
-The features are principle components which have already been det
 ermined.
-Various classifiers were fit to a training dataset in order to co
 mpare their training time.
-The classifiers were used to determine fraudalent transactions in
 the test dataset with their accuracy compared.
##################################################################

##################################################################
FILES:

creditcard.csv - Dataset containing transactions made by credit 
 				 cards in September 2013 by european cardholders. 
 				 This dataset presents transactions that occurred
 				 in two days, where there are 492 frauds out of 
 				 284,807 transactions.

 				 The dataset was obtained from:
 			   https://www.kaggle.com/dalpozz/creditcardfraud/data

cardfraud.py - This python script sorts the data into training and
			   testing data, before fitting with classifiers for
			   prediction.
			   The data is largely skewed, and so the
			   script ensures that there are an equal amount of
			   fraudalent and non-fraudalent transactions in the 
			   testing data. The testing data accounts for 10% of
			   the total data.
			   The data was fit with a selection of classifiers
			   (Naive Bayes,Decision Trees,K Nearest Neighbours,
			   AdaBoost,Random Forest) as well as the Genetic
			   Algorithm neural network I created.
			   The training times, prediction times, and accuracy
			   of test label prediction can be found in the
			   results table (results.csv).

results.csv - A table containing the results of cardfraud.py. The
			  table can be easily viewed by running the results.py
			  script.

results.py - Python script which displays the contents of
			 results.csv when run.

Stef_nn.py - A set of functions I created which are used in
			 cardfraud.py for the GA neural network's training and
			 prediction.
##################################################################