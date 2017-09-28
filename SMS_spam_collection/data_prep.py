import pandas
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.sparse import csr_matrix
#load in the data
data = pandas.read_csv('spam.csv',encoding = "latin-1")
#get rid of unwanted columns
data = data.loc[:,'v1':'v2']
#get size of the DataFrame
rows,cols = data.shape
#get stopwords
stop_words = set(stopwords.words('english'))
#initiate porter stemmer
ps = PorterStemmer()
#set up translator to remove punctuation
translator = str.maketrans('', '', string.punctuation)
#loop through and tokenize the strings, remove stopwords, and perform stemming
for i in range(0,rows):
	sms = data.iloc[i,1]
	#tokenize the string while removing punctuation
	words = word_tokenize(sms.translate(translator))
	#remove stopwords and stem
	filtered_words = []
	for w in words:
		if w not in stop_words:
			filtered_words.append(ps.stem(w))
	
	#rejoin the list into a string
	w = " ".join(filtered_words)

	data.set_value(i,1,w,takeable=True)

#convert to numpy arrays
features = np.array(data.loc[:,'v2'])
labels = np.array(data.loc[:,'v1'])
l = len(labels)
spam_index = [] #list containing the index for each spam sms
ham_index = [] #list containing the index for each non spam sms
for i in range(0,l):
	if labels[i] == 'ham':
		labels[i] = 0
		ham_index.append(i)
	elif labels[i] == 'spam':
		labels[i] = 1
		spam_index.append(i)
	else:
		print('UNIDENTIFIED LABEL AT INDEX: '+str(i))


#count the occurance of each word
CV = CountVectorizer()
TF = TfidfTransformer()
#fit and transform the features
features_count = CV.fit_transform(features) #contains a count of each word in each sms
TF.fit(features_count)
features_tfidf = TF.transform(features_count) #TfIdf representation

#convert to dense arrays in order to seperate into test and training data and perform feature selection
features_count = features_count.toarray()
features_tfidf = features_tfidf.toarray()

#seperate into traiing and testing data
num = round(rows/20)
spam_num = len(spam_index)
ham_num = len(ham_index)
#select 10% of the rows for testing
random_ham_index = random.sample(range(0,ham_num),num)
random_spam_index = random.sample(range(0,spam_num),num)

hams = [ham_index[i] for i in random_ham_index]
spams = [spam_index[i] for i in random_spam_index]

random_nums = hams+spams #the test data will have an equal number of spam and non-spam messages

features_count_test = features_count[random_nums,:]
features_tfidf_test = features_tfidf[random_nums,:]
labels_test = labels[random_nums]


#remove those rows to form the training sets
features_count_train = np.delete(features_count,random_nums,0)
features_tfidf_train = np.delete(features_tfidf,random_nums,0)
labels_train = np.delete(labels,random_nums,0)

#Use Univariate feature selection to select the k best features
SK_count_10 = SelectKBest(chi2, k=10)
SK_count_50 = SelectKBest(chi2, k=50)
SK_count_100 = SelectKBest(chi2, k=100)
SK_count_200 = SelectKBest(chi2, k=200)

SK_tfidf_10 = SelectKBest(chi2, k=10)
SK_tfidf_50 = SelectKBest(chi2, k=50)
SK_tfidf_100 = SelectKBest(chi2, k=100)
SK_tfidf_200 = SelectKBest(chi2, k=200)

#fit and transform the training features
features_count_train_k10 = SK_count_10.fit_transform(features_count_train,list(labels_train)) #need labels as list as it fixes the unkown label type error
features_count_train_k50 = SK_count_50.fit_transform(features_count_train,list(labels_train))
features_count_train_k100 = SK_count_100.fit_transform(features_count_train,list(labels_train))
features_count_train_k200 = SK_count_200.fit_transform(features_count_train,list(labels_train))

features_tfidf_train_k10 = SK_tfidf_10.fit_transform(features_tfidf_train,list(labels_train))
features_tfidf_train_k50 = SK_tfidf_50.fit_transform(features_tfidf_train,list(labels_train))
features_tfidf_train_k100 = SK_tfidf_100.fit_transform(features_tfidf_train,list(labels_train))
features_tfidf_train_k200 = SK_tfidf_200.fit_transform(features_tfidf_train,list(labels_train))

#transform the testing features
features_count_test_k10 = SK_count_10.transform(features_count_test)
features_count_test_k50 = SK_count_50.transform(features_count_test)
features_count_test_k100 = SK_count_100.transform(features_count_test)
features_count_test_k200 = SK_count_200.transform(features_count_test)

features_tfidf_test_k10 = SK_tfidf_10.transform(features_tfidf_test)
features_tfidf_test_k50 = SK_tfidf_50.transform(features_tfidf_test)
features_tfidf_test_k100 = SK_tfidf_100.transform(features_tfidf_test)
features_tfidf_test_k200 = SK_tfidf_200.transform(features_tfidf_test)

#save the arrays to files
np.save('features_count_test.npy',features_count_test)
np.save('features_tfidf_test.npy',features_tfidf_test)
np.save('features_count_train.npy',features_count_train)
np.save('features_tfidf_train.npy',features_tfidf_train)
np.save('labels_test.npy',labels_test)
np.save('labels_train.npy',labels_train)

np.save('features_count_train_k10.npy',features_count_train_k10)
np.save('features_count_train_k50.npy',features_count_train_k50)
np.save('features_count_train_k100.npy',features_count_train_k100)
np.save('features_count_train_k200.npy',features_count_train_k200)

np.save('features_tfidf_train_k10.npy',features_tfidf_train_k10)
np.save('features_tfidf_train_k50.npy',features_tfidf_train_k50)
np.save('features_tfidf_train_k100.npy',features_tfidf_train_k100)
np.save('features_tfidf_train_k200.npy',features_tfidf_train_k200)

np.save('features_count_test_k10.npy',features_count_test_k10)
np.save('features_count_test_k50.npy',features_count_test_k50)
np.save('features_count_test_k100.npy',features_count_test_k100)
np.save('features_count_test_k200.npy',features_count_test_k200)

np.save('features_tfidf_test_k10.npy',features_tfidf_test_k10)
np.save('features_tfidf_test_k50.npy',features_tfidf_test_k50)
np.save('features_tfidf_test_k100.npy',features_tfidf_test_k100)
np.save('features_tfidf_test_k200.npy',features_tfidf_test_k200)
