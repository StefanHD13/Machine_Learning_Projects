##################################################################
####################### SMS SPAM DETECTION #######################
##################################################################

##################################################################
SUMMARY:

-The SMS Spam Collection is a set of SMS tagged messages that have
 been collected for SMS Spam research. It contains one set of SMS 
 messages in English of 5,574 messages, tagged acording being ham 
 (legitimate) or spam.
-I first tokenized the words in each string while removing punctua
 tion, before removing stop words and reducing each word to its st
 em (using the python natural language toolkit).
-The resulting words were then transformed into both a Bag of Word
 s representation and TF-IDF representation (so that the two metho
 ds could later be compared).
-I then selected the k best features (k ranging from 10 to 200) in
 order to again compare resulting accuracies and training times.
-Classifiers were fit to the training data and used to determine s
 pam messages in the testing data.
##################################################################

##################################################################
FILES:

spam.csv - The data contains a set of SMS tagged messages that 
		   have been collected for SMS Spam research. It contains 
		   one set of SMS messages in English of 5,574 messages, 
		   tagged acording being ham (legitimate) or spam. 

		   The dataset was obtained from: 
	 https://www.kaggle.com/uciml/sms-spam-collection-dataset/data

data_prep.py - Python script which prepares the data (remove
			   stopwords, stemming, transforming into bag-of-words
			   and TF-IDF representation, selecting K best
			   features).

features_compare.py - This python script fits a naive bayes
					  classifier to training data that has been 
					  represented by both bag-of-words and TF-IDF,
					  as well as having varying numbers of best
					  selected features.
					  The results are saved in 
					  features_results.csv and can be viewed by
					  running results.py.

classifiers_compare.py - This python script fit the data with a
 						 selection of classifiers (Naive Bayes,
 						 Decision Trees, AdaBoost,Random Forest) 
 						 as well as the Genetic Algorithm neural 
 						 network I created.
			   			 The training times, prediction times, and 
			   			 accuracy of test label prediction can be 
			   			 found in the results table 
			   			 (class_results.csv) and can be viewed by
			   			 running results.py

features_results.csv - A table containing the results of 
					   features_compare.py. The table can be 
					   easily viewed by running the results.py 
					   script.

class_results.csv - A table containing the results of 
					features_compare.py. The table can be easily 
					viewed by running the results.py script.

results.py - Python script which displays the contents of 
			 features_results.csv and class_results.csv when run

Stef_nn.py - A set of functions I created which are used in
			 classifiers_compare.py for the GA neural network's 
			 training and prediction.

features_count_train_kXX.npy - training data represented using
							   bag-of-words, where the number 
							   replacing XX indicates the number
							   of best selected features.

features_count_test_kXX.npy - testing data represented using 
							  bag-of-words, where the number 
							  replacing XX indicates the number of 
							  best selected features.

features_tfidf_train_kXX.npy - training data represented using
							   TF-IDF, where the number replacing 
							   XX indicates the number of best 
							   selected features.

features_tfidf_test_kXX.npy - testing data represented using 
							  TF-IDF, where the number replacing 
							  XX indicates the number of best 
							  selected features.

labels_train.npy - training labels indicating whether a given SMS
				   message is spam.

labels_test.npy - testing labels indicating whether a given SMS
				  message is spam.			   
##################################################################