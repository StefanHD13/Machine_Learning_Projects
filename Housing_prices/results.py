import pandas

#ask which option you'd like to view
#1st option is to view the results
#2nd option is to view the full table of saleprice and predicted saleprice (test data)
#3rd option is to view a plot of saleprice vs predicted saleprice
#4th option is to view a plot of saleprice vs $ error
#5th option is to view a plot of saleprice vs % error

resp = input('''
What table/plot would you like to view?

1 - Table of results (training time and mean error)

2 - Plot of saleprice (test labels) vs predicted saleprice

3 - Full table of saleprice (test labels) and predicted saleprice

4 - Plot of saleprice (test labels) vs absolute error ($)

5 - Plot of saleprice (test labels) vs % error

(1/2/3/4/5) => :''')
print('\n\n')
if int(resp)==1:
	#option 1
	data = pandas.read_csv('results.csv')
	print(data)
elif int(resp)==2:
	#option 2
	import matplotlib.pyplot as plt
	data = pandas.read_csv('full_preds.csv')
	plt.scatter(data.iloc[:,0],data.iloc[:,1])
	plt.suptitle('Plot of saleprice (test labels) vs predicted saleprice')
	plt.xlabel('Sale price ($)')
	plt.ylabel('Predicted sale price ($)')
	plt.show()
elif int(resp)==3:
	#option 3
	data = pandas.read_csv('full_preds.csv')
	print(data.to_string())
elif int(resp)==4:
	#option 4
	import matplotlib.pyplot as plt
	data = pandas.read_csv('full_preds.csv')
	plt.scatter(data.iloc[:,0],data.iloc[:,2])
	plt.suptitle('Plot of saleprice (test labels) vs absolute error')
	plt.xlabel('Sale price ($)')
	plt.ylabel('Absolute error ($)')
	plt.show()
elif int(resp)==5:
	#option 5
	import matplotlib.pyplot as plt
	data = pandas.read_csv('full_preds.csv')
	plt.scatter(data.iloc[:,0],data.iloc[:,3])
	plt.suptitle('Plot of saleprice (test labels) vs Percentage error')
	plt.xlabel('Sale price ($)')
	plt.ylabel('Percentage error (%)')
	plt.show()
else:
	#put up error
	print('Error: your input of '+'"'+resp+'" '+'was invalid')
