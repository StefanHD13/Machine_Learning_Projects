##################################################################
######################## HOUSING PRICES ##########################
##################################################################

##################################################################
SUMMARY:

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
FILES:

train.csv - The dataset contains 79 variables describing (almost) 
			every aspect of residential homes in Ames, Iowa, along
			with the property's sale price.

 			The dataset was obtained from:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

data_prep.py - This python script prepares the data by converting
			   categorical data into dummy variables and by 
			   filling blank values with the average for that 
			   column. The script then seperates the data into 
			   training and testing data (the testing data
			   accounts for 10% of the data).

houseprice_train.py - This python script fits a Ridge Regression
					  model to the training data and predicts
					  saleprices for the test samples. The results
					  can be viewed by running results.py

results.py - Python script which displays results from 
			 houseprice_train.py. The option of 5 different
			 tables/plots can be viewed.
##################################################################