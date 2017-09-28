
##################################################################
######################## P R O J E C T S #########################
##################################################################

(MORE DETAILED INFORMATION CAN BE FOUND IN THE README.txt FILES IN
EACH OF THE PROJECT FOLDERS)

##################################################################
Card_Fraud:

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
Human_activity_recognition:

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
SMS_spam_collection:

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
Housing_prices:

-The dataset contains 79 variables describing (almost) every aspec
 t of residential homes in Ames, Iowa, along with the property's s
 ale price.
-I converted categorical features into dummy variables (one hot en
 coding), and replaced any blank features with the average from th
 at column.
-A ridge regression model was fit to the training data and used to
 predict the sale prices in the test data.
##################################################################

##################################################################
neural_network:

-I created a set of functions which can be used to generate a popu
 lation of neural networks, which can be fit to data using a genet
 ic algorithm.
-using these functions you just need to supply training and testin
 g data as well as your desired structure for the neural network. 
 The optimal weights for the neural network are then gradually det
 ermined as the networks evolve.
-I have used the neural network with the Card_fraud data and the 
 SMS_spam_collection data, the results of which are in each respec
 tive folder along with the results using other classifiers.
##################################################################