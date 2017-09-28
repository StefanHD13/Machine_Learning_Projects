##################################################################
################ GENETIC ALGORITHM NEURAL NETWORK ################
##################################################################

##################################################################
SUMMARY:

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

##################################################################
FILES:

Stef_nn.py - contains functions for training the genetic algorithm
			 and predicting labels.

			 The neural network does not train on all of the data 
			 in every generation as this resulted in very long 
			 training times. Instead a random batch of the data is
			 selected in each generation (with the size of the 
			 batch having been defined). If the batch_size 
			 parameter is set to 'None', then the network will 
			 train on all the data in each generation.

			 The first training function i created 
			 (nn_train_elite()) used an elitist selection method
			 where only the top 50% of networks survive to the
			 next generation, however this almost always resulted
			 in overfitting.

			 The next training function (nn_train_stoch) overcomes
			 this problem by giving networks with lower fitness
			 scores a chance to survive and continue evolving,
			 resulting in a more diverse set of networks. Each
			 network is given a probability of surviving depending
			 on its fitness compared to the maximum fitness of the
			 population. A random network is selected and will
			 either be accepted or rejected, depending on its 
			 probability. This repeats until the desired number of
			 networks have been selected.
			 
			 I created the nn_train_stoch_bank_fraud() training 
			 function in order to overcome a problem caused by
			 the largely skewed card_fraud data. Since the 
			 proportion of fraudalent transactions in the data is
			 so small, they would rarely be selected in the random
			 batch for each generation. This resulted in a poor
			 accuracy when attempting to detect fraudalent
			 transactions. Therefore this function ensures that
			 an equal number of samples from each label are
			 selected for each batch, overcoming the problem.

The results of using my GA neural network can be seen in both the
Card_Fraud project and the SMS_spam_collection project.
##################################################################
