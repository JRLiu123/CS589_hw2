#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
# create a matrix of train_set
# rows are all reviews in train_set
# columns are all unique word in train_set
# table: matrix
# new_words: how manys new words in each review in test_set
class create_matrix:
    
    def __init__(self, train):
        self.train = train

    #key: all unique words
    #value: the number of times
    def collectWords(self):
    
        all_words = sum(self.train,[])
        collect_words = {}
        for word in all_words:
   
            collect_words[word] = collect_words.get(word,0) + 1
        
        return collect_words
    
    # add index to each unique word
    def index_words(self,collect_words):
        
        words_index = {}
        words_list = list(collect_words.keys())
        i = 0
        for words in words_list:
            words_index[words] = i
            i = i + 1
        
        return words_index

    def make_matrix(self,collect_words,words_index):

        X = np.zeros((len(self.train),len(words_index)))
        for i in range(len(self.train)):
            for word in self.train[i]:
                index = int(words_index.get(word))
                X[i, index] = X[i, index] + 1
        return X

    def ins_table(self,instance,words_index):
   
        table = np.zeros(len(words_index))
        
        new_words = 0
        for words in instance:
            
            if words in list(words_index.keys()):
                table[int(words_index[words])] = 1
            else:
                new_words = new_words + 1
        
        return table,new_words

