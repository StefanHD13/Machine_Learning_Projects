#need 2 functions one for predicting and one for training
import numpy as np
from math import exp
import random

#prediction function

def nn_predict(features,weights,neurons):
	# features : an array of the features to predit labels for

	# weights : a list of the weights. the first weight for each layer is for the bias neuron (1)

	# neurons : a list containing the number of neurons in each hidden layer and output layer


	# first find the number of input features
	rows, features_num = np.shape(features)
	# get the number of hidden layers
	hidden_lay_num = (len(neurons)-1)
	# make an error check to see that the number of neurons and number of weights are matching
	### NEED TO DO THIS  ######

	#create an array of zeros to store the predicted labels
	labels = np.zeros((rows,neurons[hidden_lay_num]))

	# need to loop through the number of samples and predict labels for each of them
	for s in range(0,rows):
		#initialise the weight location
		weight_loc = 0
		#loop through each hidden layer and the output layer
		for i in range(0,(hidden_lay_num+1)):
			#get number of neurons for current layer
			current_neurons_num = neurons[i]
			#get number of neurons for previous layer and set the values for the previous layer neurons
			if i == 0:
				previous_neurons_num = features_num
				x = features[s,:]
				previous_neurons = x.tolist()

			else:
				previous_neurons_num = neurons[i-1]
				previous_neurons = current_neurons

			#calculate the values for each of the neurons in the current layer
			current_neurons = [0]*current_neurons_num
			for j in range(0,current_neurons_num):
				#add the bias to the neuron
				current_neurons[j] = 1 * weights[weight_loc]
				weight_loc = weight_loc +1
				#loop through each of the neurons in the previous layer
				for k in range(0,previous_neurons_num):
					current_neurons[j] = current_neurons[j] + (previous_neurons[k]*weights[weight_loc])
					weight_loc = weight_loc + 1

				#now need to normalise the value for this neuron
				# normalise using sigmoid function
				x = current_neurons[j]
				current_neurons[j] = exp(x)/(exp(x)+1)

		#should now have the neuron values for output layer under current_neurons
		#store them in the numpy array
		for i in range(0,neurons[hidden_lay_num]):
			labels[s,i] = current_neurons[i]


	return labels


# train a neural network using genetic algorithm

def nn_train_elite(features,labels,hidden_lay_num,neurons_num,population,iterations,mutations,batch_size):
	#features: an array of features to train from
	#labels: an array of labels to train to predict
	#hidden_lay_num: the number of hidden layers for the neural network
	#hidden_lay_neurons: a list containing the number of neurons for each hidden layer
	#population: the population of the genetic algorithm IMPORTANT: Half of the population must be an even number
	#iterations: the total number of iterations for the genetic algorithm
	#mutations: affects the chance of a gene mutating when children are created. generates a random integer between 0 and this value.
	#		if the generated integer is 0, that gene will mutate
	#batch_size: the size of the random batch in each generation. if None, will use all of the samples in each generation

	# INVALID INPUTS CHECKS
	# check that the population divided by two is an even number. if not, give an error



	#first calculate the number of weights needed
	samples_num,features_num = np.shape(features)
	samples_num,labels_num = np.shape(labels)
	neurons_num.append(labels_num)
	weights_num = (features_num+1)*neurons_num[0] #the +1 is for the bias neuron
	x = len(neurons_num)
	for i in range(1,x):
		weights_num = weights_num + (neurons_num[i]*(neurons_num[i-1]+1)) #the +1 is for the bias neuron

	#now weights_num has the number of weights for the neural network

	#create a numpy array of random numbers with columns equal to the number of weights and rows equal to the population
	networks = np.random.uniform(-2,2,(population,weights_num))
	x = np.zeros((population,1))
	networks = np.concatenate((networks,x),1) #the last column of networks is for the fitness

	#create a small random batch of features and labels
	if batch_size == None:
		labels_batch = labels
		features_batch = features
		batch_samples = samples_num

	else:
		#generating a list of random numbers determined by batch_size
		random_nums = random.sample(range(0,samples_num),batch_size)
		labels_batch = labels[random_nums,:]
		features_batch = features[random_nums,:]
		batch_samples = batch_size

	#now for these inital networks, calculate the fitness for each one
	for i in range(0,population):
		#print('Calculating fitness for initial networks: '+str(i)+'/'+str(population))
		#get the weights for the network in a list
		x = networks[i,0:weights_num]
		weights = x.tolist()

		#predict the labels using this network
		labels_pred = nn_predict(features_batch,weights,neurons_num)

		#calculate a fitness score based on the labels
		#the fitness will be the mean of the squared residuals between each label
		fitness=0
		count=0
		for j in range(0,batch_samples):
			for k in range(0,labels_num):
				#add the squared residual between the current location label and predicted label
				curr_label = labels_batch[j,k]
				curr_label_pred = labels_pred[j,k]
				fitness = fitness + ((curr_label-curr_label_pred)**2)
				count = count +1

		#now store the fitness score in the last column
		fitness = fitness/count
		networks[i,weights_num] = fitness


	#####   ITERATIONS START HERE  ######
	new_pop = round(population/2)

	#create a list to store the best fitnesses from each generation
	best_fitnesses = [0] * iterations
	for s in range(0,iterations):
		print('Beginning generation '+str(s))

		#now find 50% of the networks with the smallest fitness
		networks_best = np.zeros((new_pop,(weights_num+1)))
		for i in range(0,new_pop):
			index = np.argmin(networks[:, weights_num])
			#add that row to the new network array
			networks_best[i,:] = networks[index,:]
			#add the fitness value to the best_fitnesses list
			best_fitnesses[s] = networks[index,weights_num]
			#delete that row from the network array
			networks = np.delete(networks,index,0)

		#now need to create an array for the children
		networks_children = np.zeros((new_pop,(weights_num+1)))
		i=0
		while i < new_pop:
			#print('Creating children: '+str(i)+'/'+str(new_pop))
			#each pair will create 2 children which are the inverse of each other
			for j in range(0,weights_num):
				#generate a random integer (either 0 or 1)
				x = random.randint(0,1)
				if x==0:
					networks_children[i,j] = networks_best[i,j]
					networks_children[i+1,j] = networks_best[i+1,j]
				else:
					networks_children[i,j] = networks_best[i+1,j]
					networks_children[i+1,j] = networks_best[i,j]

				#generate two random integers based on the mutation variable. if it is 0, mutate the weight
				x = random.randint(0,mutations)
				y = random.randint(0,mutations)
				#mutate the weight by generating a random float between 0 and twice the current weight
				if x==0:
					#mutate the child at [i,j]
					gene = networks_children[i,j]
					gene = random.uniform(-(2*gene),(2*gene))
					networks_children[i,j] = gene
				if y ==0:
					#mutate the child at [i+1,j]
					gene = networks_children[i+1,j]
					gene = random.uniform(-(2*gene),(2*gene))
					networks_children[i+1,j] = gene
			i = i +2

		#networks_children contains the child networks from networks_best
		#create a small random batch of features and labels
		if batch_size == None:
			labels_batch = labels
			features_batch = features
			batch_samples = samples_num

		else:
			#generating a list of random numbers determined by batch_size
			random_nums = random.sample(range(0,samples_num),batch_size)
			labels_batch = labels[random_nums,:]
			features_batch = features[random_nums,:]
			batch_samples = batch_size

		#now just need to calculate fitnesses for networks_children
		#now calculate the fitness for each one
		for i in range(0,new_pop):
			#print('Calculating children fitnesses: '+str(i)+'/'+str(new_pop))
			#get the weights for the network in a list
			x = networks_children[i,0:weights_num]
			weights = x.tolist()

			#predict the labels using this network
			labels_pred = nn_predict(features_batch,weights,neurons_num)

			#calculate a fitness score based on the labels
			#the fitness will be the mean of the squared residuals between each label
			fitness=0
			count =0
			for j in range(0,batch_samples):
				for k in range(0,labels_num):
					#add the squared residual between the current location label and predicted label
					curr_label = labels_batch[j,k]
					curr_label_pred = labels_pred[j,k]
					fitness = fitness + ((curr_label-curr_label_pred)**2)
					count = count+1

			#now store the fitness score in the last column
			fitness = fitness/count
			networks_children[i,weights_num] = fitness

		#concenate networks_children with networks_best and repeat dat loop
		networks = np.concatenate((networks_best,networks_children),0)

		#put stuff here to show how far the training is
		print('Generation '+str(s)+' complete')
	####### ITERATIONS END HERE #######

	#best_fitnesses contains the ebst fitness score from each iteration

	#return the current list of networks
	return networks, best_fitnesses


def nn_train_stoch(features,labels,hidden_lay_num,neurons_num,population,iterations,mutations,batch_size):
	#uses a stochastic sampling method for parent selection rather than the previous elitist selection
	#uses stochastic acceptance

	#features: an array of features to train from
	#labels: an array of labels to train to predict
	#hidden_lay_num: the number of hidden layers for the neural network
	#hidden_lay_neurons: a list containing the number of neurons for each hidden layer
	#population: the population of the genetic algorithm IMPORTANT: Half of the population must be an even number
	#iterations: the total number of iterations for the genetic algorithm
	#mutations: affects the chance of a gene mutating when children are created. generates a random integer between 0 and this value.
	#		if the generated integer is 0, that gene will mutate
	#batch_size: the size of the random batch in each generation. if None, will use all of the samples in each generation

	# INVALID INPUTS CHECKS
	# check that the population divided by two is an even number. if not, give an error



	#first calculate the number of weights needed
	samples_num,features_num = np.shape(features)
	samples_num,labels_num = np.shape(labels)
	neurons_num.append(labels_num)
	weights_num = (features_num+1)*neurons_num[0] #the +1 is for the bias neuron
	x = len(neurons_num)
	for i in range(1,x):
		weights_num = weights_num + (neurons_num[i]*(neurons_num[i-1]+1)) #the +1 is for the bias neuron

	#now weights_num has the number of weights for the neural network

	#create a numpy array of random numbers with columns equal to the number of weights and rows equal to the population
	networks = np.random.uniform(-2,2,(population,weights_num))
	x = np.zeros((population,1))
	networks = np.concatenate((networks,x),1) #the last column of networks is for the fitness

	#create a small random batch of features and labels
	if batch_size == None:
		labels_batch = labels
		features_batch = features
		batch_samples = samples_num

	else:
		#generating a list of random numbers determined by batch_size
		random_nums = random.sample(range(0,samples_num),batch_size)
		labels_batch = labels[random_nums,:]
		features_batch = features[random_nums,:]
		batch_samples = batch_size

	#now for these inital networks, calculate the fitness for each one
	for i in range(0,population):
		#print('Calculating fitness for initial networks: '+str(i)+'/'+str(population))
		#get the weights for the network in a list
		x = networks[i,0:weights_num]
		weights = x.tolist()

		#predict the labels using this network
		labels_pred = nn_predict(features_batch,weights,neurons_num)

		#calculate a fitness score based on the labels
		#the fitness will be the mean of the squared residuals between each label
		fitness=0
		count=0
		for j in range(0,batch_samples):
			for k in range(0,labels_num):
				#add the squared residual between the current location label and predicted label
				curr_label = labels_batch[j,k]
				curr_label_pred = labels_pred[j,k]
				fitness = fitness + ((curr_label-curr_label_pred)**2)
				count = count +1

		#now store the fitness score in the last column
		fitness = fitness/count
		networks[i,weights_num] = fitness


	#####   ITERATIONS START HERE  ######
	new_pop = round(population/2)

	#create a numpy array to store the best,average,and worst fitnesses from each generation (four columns with first being gen number)
	fitness_store = np.zeros((iterations,4))
	for s in range(0,iterations):
		print('Beginning generation '+str(s))

		#now use stochastic acceptance to select the parents
		networks_best = np.zeros((new_pop,(weights_num+1)))
		i=0
		rows,cols = np.shape(networks)
		# find the value for the maximum fitness
		index = np.argmax(networks[:, weights_num])
		max_fitness = networks[index,weights_num]

		while (i < new_pop):
			# select a random row
			row_select = random.randint(0,(rows-1))
			#calcuate the probability of the row being accepted (fitness/max)
			fitness = networks[row_select,weights_num]
			probability = fitness/max_fitness
			#generate a random float between 0 and 1
			num = random.random()
			#accept that row if the float is more than or equal to probability (since a lower fitness is better)
			if (num>=probability):
				#add that row to the new network array
				networks_best[i,:] = networks[row_select,:]

				#delete that row from the network array
				networks = np.delete(networks,row_select,0)
				rows = rows -1
				#add one to i
				i = i+1

		#now need to create an array for the children
		networks_children = np.zeros((new_pop,(weights_num+1)))
		i=0
		while i < new_pop:
			#print('Creating children: '+str(i)+'/'+str(new_pop))
			#each pair will create 2 children which are the inverse of each other
			for j in range(0,weights_num):
				#generate a random integer (either 0 or 1)
				x = random.randint(0,1)
				if x==0:
					networks_children[i,j] = networks_best[i,j]
					networks_children[i+1,j] = networks_best[i+1,j]
				else:
					networks_children[i,j] = networks_best[i+1,j]
					networks_children[i+1,j] = networks_best[i,j]

				#generate two random integers based on the mutation variable. if it is 0, mutate the weight
				x = random.randint(0,mutations)
				y = random.randint(0,mutations)
				#mutate the weight by generating a random float between 0 and twice the current weight
				if x==0:
					#mutate the child at [i,j]
					gene = networks_children[i,j]
					gene = random.uniform(-(2*gene),(2*gene))
					networks_children[i,j] = gene
				if y ==0:
					#mutate the child at [i+1,j]
					gene = networks_children[i+1,j]
					gene = random.uniform(-(2*gene),(2*gene))
					networks_children[i+1,j] = gene
			i = i +2

		#networks_children contains the child networks from networks_best
		#create a small random batch of features and labels
		if batch_size == None:
			labels_batch = labels
			features_batch = features
			batch_samples = samples_num

		else:
			#generating a list of random numbers determined by batch_size
			random_nums = random.sample(range(0,samples_num),batch_size)
			labels_batch = labels[random_nums,:]
			features_batch = features[random_nums,:]
			batch_samples = batch_size

		#now just need to calculate fitnesses for networks_children
		#now calculate the fitness for each one
		for i in range(0,new_pop):
			#print('Calculating children fitnesses: '+str(i)+'/'+str(new_pop))
			#get the weights for the network in a list
			x = networks_children[i,0:weights_num]
			weights = x.tolist()

			#predict the labels using this network
			labels_pred = nn_predict(features_batch,weights,neurons_num)

			#calculate a fitness score based on the labels
			#the fitness will be the mean of the squared residuals between each label
			fitness=0
			count =0
			for j in range(0,batch_samples):
				for k in range(0,labels_num):
					#add the squared residual between the current location label and predicted label
					curr_label = labels_batch[j,k]
					curr_label_pred = labels_pred[j,k]
					fitness = fitness + ((curr_label-curr_label_pred)**2)
					count = count+1

			#now store the fitness score in the last column
			fitness = fitness/count
			networks_children[i,weights_num] = fitness

		#concenate networks_children with networks_best and repeat dat loop
		networks = np.concatenate((networks_best,networks_children),0)

		# put the best, average and worst fitnesses from the generation in fitness_store
		fitness_best = np.max(networks[:,weights_num]) #find the best fitness
		fitness_average = np.mean(networks[:,weights_num]) #find the average fitness
		fitness_worst = np.min(networks[:,weights_num]) #find the worst fitness
		fitness_store[s,0] = s
		fitness_store[s,1] = fitness_best
		fitness_store[s,2] = fitness_average
		fitness_store[s,3] = fitness_worst
		#put stuff here to show how far the training is
		print('Generation '+str(s)+' complete')
	####### ITERATIONS END HERE #######

	#fitness_store contains the best, average, and worst fitness from each generation

	#return the current list of networks
	return networks, fitness_store


def nn_train_stoch_bank_fraud(features,labels,hidden_lay_num,neurons_num,population,iterations,mutations,batch_size):
	#THIS IS SPECIFICALLY FOR THE BANK FRAUD DATA
	#selects and equal number of fraudalent and non fraudalent samples in each batch
	#BATCH SIZE MUST BE EVEN

	#uses a stochastic sampling method for parent selection rather than the previous elitist selection
	#uses stochastic acceptance

	#
	#features: an array of features to train from
	#labels: an array of labels to train to predict
	#hidden_lay_num: the number of hidden layers for the neural network
	#hidden_lay_neurons: a list containing the number of neurons for each hidden layer
	#population: the population of the genetic algorithm IMPORTANT: Half of the population must be an even number
	#iterations: the total number of iterations for the genetic algorithm
	#mutations: affects the chance of a gene mutating when children are created. generates a random integer between 0 and this value.
	#		if the generated integer is 0, that gene will mutate
	#batch_size: the size of the random batch in each generation. if None, will use all of the samples in each generation

	# INVALID INPUTS CHECKS
	# check that the population divided by two is an even number. if not, give an error



	#first calculate the number of weights needed
	samples_num,features_num = np.shape(features)
	samples_num,labels_num = np.shape(labels)
	neurons_num.append(labels_num)
	weights_num = (features_num+1)*neurons_num[0] #the +1 is for the bias neuron
	x = len(neurons_num)
	for i in range(1,x):
		weights_num = weights_num + (neurons_num[i]*(neurons_num[i-1]+1)) #the +1 is for the bias neuron

	#now weights_num has the number of weights for the neural network

	#create a numpy array of random numbers with columns equal to the number of weights and rows equal to the population
	networks = np.random.uniform(-2,2,(population,weights_num))
	x = np.zeros((population,1))
	networks = np.concatenate((networks,x),1) #the last column of networks is for the fitness

	#create a small random batch of features and labels
	if batch_size == None:
		labels_batch = labels
		features_batch = features
		batch_samples = samples_num

	else:
		#create an array of fraud features and an array for those labels
		x = []
		for i in range(0,samples_num):
			y = labels[i,0]
			if y==1:
				x.append(i)
		features_fraud = features[x,:]
		labels_fraud = labels[x,:]
		fraud_samples_num,x = np.shape(labels_fraud)
		#generating a list of random numbers determined by batch_size
		random_nums = random.sample(range(0,samples_num),round(batch_size/2))
		random_nums_fraud = random.sample(range(0,fraud_samples_num),round(batch_size/2))
		labels_batch = np.concatenate((labels[random_nums,:],labels_fraud[random_nums_fraud,:]),0)
		features_batch = np.concatenate((features[random_nums,:],features_fraud[random_nums_fraud,:]),0)
		batch_samples = batch_size

	#now for these inital networks, calculate the fitness for each one
	for i in range(0,population):
		#print('Calculating fitness for initial networks: '+str(i)+'/'+str(population))
		#get the weights for the network in a list
		x = networks[i,0:weights_num]
		weights = x.tolist()

		#predict the labels using this network
		labels_pred = nn_predict(features_batch,weights,neurons_num)

		#calculate a fitness score based on the labels
		#the fitness will be the mean of the squared residuals between each label
		fitness=0
		count=0
		for j in range(0,batch_samples):
			for k in range(0,labels_num):
				#add the squared residual between the current location label and predicted label
				curr_label = labels_batch[j,k]
				curr_label_pred = labels_pred[j,k]
				fitness = fitness + ((curr_label-curr_label_pred)**2)
				count = count +1

		#now store the fitness score in the last column
		fitness = fitness/count
		networks[i,weights_num] = fitness


	#####   ITERATIONS START HERE  ######
	new_pop = round(population/2)

	#create a numpy array to store the best,average,and worst fitnesses from each generation (four columns with first being gen number)
	fitness_store = np.zeros((iterations,4))
	for s in range(0,iterations):
		print('Beginning generation '+str(s))

		#now use stochastic acceptance to select the parents
		networks_best = np.zeros((new_pop,(weights_num+1)))
		i=0
		rows,cols = np.shape(networks)
		# find the value for the maximum fitness
		index = np.argmax(networks[:, weights_num])
		max_fitness = networks[index,weights_num]

		while (i < new_pop):
			# select a random row
			row_select = random.randint(0,(rows-1))
			#calcuate the probability of the row being accepted (fitness/max)
			fitness = networks[row_select,weights_num]
			probability = fitness/max_fitness
			#generate a random float between 0 and 1
			num = random.random()
			#accept that row if the float is more than or equal to probability (since a lower fitness is better)
			if (num>=probability):
				#add that row to the new network array
				networks_best[i,:] = networks[row_select,:]

				#delete that row from the network array
				networks = np.delete(networks,row_select,0)
				rows = rows -1
				#add one to i
				i = i+1

		#now need to create an array for the children
		networks_children = np.zeros((new_pop,(weights_num+1)))
		i=0
		while i < new_pop:
			#print('Creating children: '+str(i)+'/'+str(new_pop))
			#each pair will create 2 children which are the inverse of each other
			for j in range(0,weights_num):
				#generate a random integer (either 0 or 1)
				x = random.randint(0,1)
				if x==0:
					networks_children[i,j] = networks_best[i,j]
					networks_children[i+1,j] = networks_best[i+1,j]
				else:
					networks_children[i,j] = networks_best[i+1,j]
					networks_children[i+1,j] = networks_best[i,j]

				#generate two random integers based on the mutation variable. if it is 0, mutate the weight
				x = random.randint(0,mutations)
				y = random.randint(0,mutations)
				#mutate the weight by generating a random float between 0 and twice the current weight
				if x==0:
					#mutate the child at [i,j]
					gene = networks_children[i,j]
					gene = random.uniform(-(2*gene),(2*gene))
					networks_children[i,j] = gene
				if y ==0:
					#mutate the child at [i+1,j]
					gene = networks_children[i+1,j]
					gene = random.uniform(-(2*gene),(2*gene))
					networks_children[i+1,j] = gene
			i = i +2

		#networks_children contains the child networks from networks_best
		#create a small random batch of features and labels
		if batch_size == None:
			labels_batch = labels
			features_batch = features
			batch_samples = samples_num

		else:
			#generating a list of random numbers determined by batch_size
			random_nums = random.sample(range(0,samples_num),round(batch_size/2))
			random_nums_fraud = random.sample(range(0,fraud_samples_num),round(batch_size/2))
			labels_batch = np.concatenate((labels[random_nums,:],labels_fraud[random_nums_fraud,:]),0)
			features_batch = np.concatenate((features[random_nums,:],features_fraud[random_nums_fraud,:]),0)
			batch_samples = batch_size

		#now just need to calculate fitnesses for networks_children
		#now calculate the fitness for each one
		for i in range(0,new_pop):
			#print('Calculating children fitnesses: '+str(i)+'/'+str(new_pop))
			#get the weights for the network in a list
			x = networks_children[i,0:weights_num]
			weights = x.tolist()

			#predict the labels using this network
			labels_pred = nn_predict(features_batch,weights,neurons_num)

			#calculate a fitness score based on the labels
			#the fitness will be the mean of the squared residuals between each label
			fitness=0
			count =0
			for j in range(0,batch_samples):
				for k in range(0,labels_num):
					#add the squared residual between the current location label and predicted label
					curr_label = labels_batch[j,k]
					curr_label_pred = labels_pred[j,k]
					fitness = fitness + ((curr_label-curr_label_pred)**2)
					count = count+1

			#now store the fitness score in the last column
			fitness = fitness/count
			networks_children[i,weights_num] = fitness

		#concenate networks_children with networks_best and repeat dat loop
		networks = np.concatenate((networks_best,networks_children),0)

		# put the best, average and worst fitnesses from the generation in fitness_store
		fitness_best = np.max(networks[:,weights_num]) #find the best fitness
		fitness_average = np.mean(networks[:,weights_num]) #find the average fitness
		fitness_worst = np.min(networks[:,weights_num]) #find the worst fitness
		fitness_store[s,0] = s
		fitness_store[s,1] = fitness_best
		fitness_store[s,2] = fitness_average
		fitness_store[s,3] = fitness_worst
		#put stuff here to show how far the training is
		print('Generation '+str(s)+' complete')
	####### ITERATIONS END HERE #######

	#fitness_store contains the best, average, and worst fitness from each generation

	#return the current list of networks
	return networks, fitness_store