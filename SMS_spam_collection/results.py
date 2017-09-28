import pandas
data_feat = pandas.read_csv('features_results.csv')
data_class = pandas.read_csv('class_results.csv')

print('\nSelect K best Comparison:\n')
print(data_feat)
print('\n\nClassifiers Comparison:\n')
print(data_class)
