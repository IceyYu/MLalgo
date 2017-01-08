from __future__ import division
import numpy as np
import re

class Spam(object):
	def vocab_list(self, email):
		vocab_temp = set([])
		for item in email:
			vocab_temp = vocab_temp | set(item)
		return list(vocab_temp)

	def word_to_vec(self, word):
		if type(word[0]) == list:
			vec = np.zeros((len(word), len(vocab)))
			for i in range(len(word)):
				for item in word[i]:
					if item in vocab:
						vec[i][vocab.index(item)] += 1
		else:
			vec = np.zeros((1,len(vocab)))
			for item in word:
				if item in vocab:
					vec[0][vocab.index(item)] += 1
		return vec

	def split(self, string):
		string_split = re.split(r'\W*', string)
		return [item.lower() for item in string_split if len(item) > 2]



	def spam_test(self):
		global vocab
		doc_list, full_list, class_list = [], [], []
		for i in range(1, 26):
			word_list = self.split(open('email/spam/%d.txt' % i).read())
			doc_list.append(word_list)
			full_list.extend(word_list)
			class_list.append(1)
			word_list = self.split(open('email/ham/%d.txt' % i).read())
			doc_list.append(word_list)
			full_list.extend(word_list)
			class_list.append(0)
		test_index = np.random.choice(50, 10, replace = False).tolist()
		train_index = list(set(range(50)) - set(test_index))
		train_mat, train_class = [], []
		test_mat, test_class = [], []
		for index in train_index:
			train_mat.append(doc_list[index])
			train_class.append(class_list[index])
		vocab = self.vocab_list(train_mat)
		train_mat = self.word_to_vec(train_mat)
		for index in test_index:
			test_mat.append(doc_list[index])
			test_class.append(class_list[index])
		test_mat = self.word_to_vec(test_mat)
		train_set = {
						'train_mat': train_mat,
						'train_class': train_class,
						'test_mat': test_mat,
						'test_class': test_class
					}
		return train_set

	def test(self):
		train_set = self.spam_test()
		vec = train_set['train_mat']
		train_class = train_set['train_class']
		test_mat = train_set['test_mat']
		test_class = train_set['test_class']
		zero_num = train_class.count(0)
		one_num = train_class.count(1)
		zero_prob = zero_num / 40
		one_prob = one_num / 40
		temp = [0] * 2
		sum_array = np.ones((2, vec.shape[1])) #Laplace
		for i in range(40):
			if train_class[i] == 0:
				sum_array[0] += vec[i]
			else:
				sum_array[1] += vec[i]
		prob_dict = {0:[],1:[]}
		sum_num_zero = np.sum(sum_array[0])
		sum_num_one = np.sum(sum_array[1])
		sum_num = np.sum(sum_array)
		prob_x_y_zero = sum_array[0] / (sum_num_zero + len(vocab)) #Laplace
		prob_x_y_one = sum_array[1] / (sum_num_one + len(vocab))
		prob_dict[0] = np.log(prob_x_y_zero)	
		prob_dict[1] = np.log(prob_x_y_one)
		prob_y_zero = sum_num_zero / sum_num
		prob_y_one = sum_num_one / sum_num
		right_rate = 0		
		prob0,prob1 = 0,0
		for i in range(len(test_mat)):
			for j in range(len(test_mat[i])):
				if test_mat[i][j] != 0:
					prob0 += prob_dict[0][j] * test_mat[i][j]
					prob1 += prob_dict[1][j] * test_mat[i][j]
			p0 = prob0 + np.log(prob_y_zero)
			p1 = prob1 + np.log(prob_y_one)
			if p0 > p1:
				label = 0
			else:
				label = 1
			if label == test_class[i]:
				right_rate += 1/10
			prob0,prob1 = 0,0

		return right_rate


	

if __name__ == '__main__':
	Bayes = Spam()
	rate = []
	for i in range(10):
		rate.append(Bayes.test())
	print rate
	result = np.mean(rate)
	print result


