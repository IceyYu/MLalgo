from __future__ import division
import numpy as np
import re

class Bayes(object):
	def __init__(self):
	    self.posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
             ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	    self.class_vec = [0,1,0,1,0,1]
	    self.num = len(self.class_vec)    

	def vocab_list(self):
		vocab = set([])
		for item in self.posting_list:
			vocab = vocab | set(item)
		return list(vocab)

	def word_to_vec(self, test = None):
		vocab = self.vocab_list()
		if test == None:
			vec = np.zeros((self.num, len(vocab)))
			for i in range(self.num):
				for item in self.posting_list[i]:
					if item in vocab:
						vec[i][vocab.index(item)] = 1
		else:
			if type(test[0]) == list:
				vec = np.zeros((len(test), len(vocab)))
				for i in range(len(test)):
					for item in test[i]:
						if item in vocab:
							vec[i][vocab.index(item)] = 1
			else:
				vec = np.zeros((1,len(vocab)))
				for item in test:
					if item in vocab:
						vec[0][vocab.index(item)] = 1

		return vec

	def train(self):
		zero_num = self.class_vec.count(0)
		one_num = self.class_vec.count(1)
		zero_prob = zero_num / self.num
		one_prob = one_num / self.num
		temp = [0] * 2
		vec = self.word_to_vec()
		sum_array = np.ones((2, vec.shape[1])) #Laplace
		for i in range(self.num):
			if self.class_vec[i] == 0:
				sum_array[0] += vec[i]
			else:
				sum_array[1] += vec[i]
		prob_dict = {0:[],1:[]}
		sum_num_zero = np.sum(sum_array[0])
		sum_num_one = np.sum(sum_array[1])
		sum_num = np.sum(sum_array)
		prob_x_y_zero = sum_array[0] / (sum_num_zero + 2) #Laplace
		prob_x_y_one = sum_array[1] / (sum_num_one + 2)
		prob_dict[0] = np.log(prob_x_y_zero)	
		prob_dict[1] = np.log(prob_x_y_one)
		prob_y_zero = sum_num_zero / sum_num
		prob_y_one = sum_num_one / sum_num
		return prob_dict, prob_y_zero, prob_y_one

	def test(self, test_entry):
		prob_dict, prob_y_zero, prob_y_one = self.train()
		vec = self.word_to_vec(test=test_entry)
		prob0,prob1 = 0,0
		for i in range(len(vec)):
			if vec[0][i] == 1:
				prob0 += prob_dict[0][i]
				prob1 += prob_dict[1][i]
		p0 = prob0 + np.log(prob_y_zero)
		p1 = prob1 + np.log(prob_y_one)
		if p0 > p1:
			label = 0
		else:
			label = 1
		return label











if __name__ == '__main__':
	Bayes = Bayes()
	document = Bayes.train()
	test_entry = ['love', 'my', 'dalmaton']
	label = Bayes.test(test_entry)
	print label 


	
	